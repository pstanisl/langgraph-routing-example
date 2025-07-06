"""LangGraph Agent Handoff Example using Command.

This example demonstrates how to create a multi-agent (workflow) system where router agent handoff tasks to
specialized agents based on conversation context. The router generates a routing decision using and based on
the conditional logic, it routes the conversation to the appropriate agent.

The routing decision must be part of the prompt of the router agent, which uses a structured output
to determine the next agent to handle the conversation.
"""

from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    openai_base_url: str = Field(..., description="OpenAI base URL")
    openai_model: str = Field(..., description="LLM model name")


# Load settings
settings = Settings()


def update_current_agent(left: str, right: str) -> str:
    """Update function for current_agent - just return the new value."""
    logger.info(f"Updating current agent from {left} to {right}")
    return right


class NextAgent(BaseModel):
    """Structured output for supervisor routing decisions."""

    next_agent: Literal["support_agent", "research_agent", "manager_agent", "__end__"] = Field(
        description="The next agent to route to, or '__end__' to finish"
    )
    reasoning: str = Field(description="Brief explanation for the routing decision")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: Annotated[str, update_current_agent]


def create_agent(name: str, system_message: str, model: ChatOpenAI) -> Callable:
    """Create a simple agent that processes tasks without handoff logic."""

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]

        # Create system message
        system_msg = SystemMessage(content=f"You are {name}. {system_message}")

        # Get response from the model
        response = model.invoke([system_msg] + messages[-3:])

        # Add agent response to messages
        agent_response = AIMessage(content=f"[{name}]: {response.content}")

        return {"messages": [agent_response], "current_agent": name}

    return agent_node


def router(
    state: AgentState,
) -> Command[Literal["support_agent", "research_agent", "manager_agent", END]]:
    """Router that decides which agent to route to next based on conversation context."""
    messages = state["messages"]
    current_agent = state.get("current_agent", "support_agent")

    # Create supervisor model with structured output
    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    ).with_structured_output(NextAgent)

    system_prompt = f"""You are a supervisor routing conversations between specialized agents:

- **support_agent**: Handles general questions, basic help, and customer support
- **research_agent**: Conducts analysis, research, and provides detailed insights
- **manager_agent**: Makes decisions, handles escalations, and provides strategic guidance

Current conversation context: The user is currently being helped by {current_agent}.

Based on the conversation, determine which agent should handle the next interaction:
1. Route to **support_agent** for general help, basic questions, routine matters
2. Route to **research_agent** for research, analysis, investigation, complex topics
3. Route to **manager_agent** for decisions, escalations, strategic matters, urgent issues
4. Route to **__end__** if the conversation is clearly finished (user says goodbye, thanks, etc.)

Consider the user's latest message and the conversation flow to make the best routing decision.
"""

    # Get routing decision from supervisor
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            *messages[-3:],  # Last 3 messages for context
        ]
    )

    logger.debug(f"Supervisor decision: {response.next_agent} - {response.reasoning}")

    # Handle end condition
    if response.next_agent == "__end__":
        return Command(goto=END)

    # Route to the selected agent
    return Command(
        goto=response.next_agent, update={"current_agent": response.next_agent.replace("_", " ").title()}
    )


def create_workflow() -> Callable:
    """Create the LangGraph workflow with router agent handoffs."""
    # Initialize the language model with settings
    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.7,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    # Create specialized agents (simplified without handoff logic)
    support_agent = create_agent(
        "Support Agent",
        "You help users with general questions, provide customer support, and handle routine inquiries. "
        "Be helpful and friendly in your responses.",
        model,
    )

    research_agent = create_agent(
        "Research Agent",
        "You are a research specialist who analyzes complex topics and provides detailed insights. "
        "Focus on providing thorough, well-researched responses with analysis and context.",
        model,
    )

    manager_agent = create_agent(
        "Manager Agent",
        "You are a senior manager who handles escalated issues and makes strategic decisions. "
        "Provide authoritative guidance and make clear decisions when needed.",
        model,
    )

    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add supervisor node first
    workflow.add_node("router", router)

    # Add agent nodes
    workflow.add_node("support_agent", support_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("manager_agent", manager_agent)

    # Set entry point to router
    workflow.set_entry_point("router")

    # Agents end the conversation after responding (no automatic routing back)
    # The router will route to the appropriate agent and that agent will finish

    # Compile the graph with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def main() -> None:
    """Main function to run the agent handoff example."""
    logger.info("🤖 Multi-Agent Handoff System with Command Objects Started!")
    logger.info("💡 Try saying:")
    logger.info("  - 'I need research on AI trends' (→ Research Agent)")
    logger.info("  - 'Escalate this to manager' (→ Manager Agent)")
    logger.info("  - 'Basic help please' (→ Support Agent)")
    logger.info("Type 'quit' to exit\n")

    # Create the workflow
    app = create_workflow()

    # Initialize conversation state
    thread_id = "conversation-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            logger.info("Goodbye! 👋")
            break

        if not user_input:
            continue

        try:
            # Create state with user message
            state = {"messages": [HumanMessage(content=user_input)], "current_agent": "Support Agent"}

            # Run the workflow with recursion limit
            result = app.invoke(state, {**config, "recursion_limit": 10})

            # Print the agent responses
            if result["messages"]:
                # Show all new messages from this interaction
                for msg in result["messages"]:
                    if isinstance(msg, AIMessage):
                        logger.info(f"\n{msg.content}")

                # Show current agent info
                logger.info(f"\n💡 Current Agent: {result['current_agent']}")
                logger.info("-" * 50)

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            logger.info("Make sure you have configured your .env file with OPENAI_API_KEY")
            logger.info("See .env file for configuration options including proxy settings")


if __name__ == "__main__":
    main()
