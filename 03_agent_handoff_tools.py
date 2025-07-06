"""LangGraph Agent Handoff Example using Tool Calling with Supervisor Pattern.

This example demonstrates how to create a multi-agent system using the supervisor pattern
where agents are implemented as tools and the supervisor uses create_react_agent.

The supervisor routes tasks to specialized agents based on conversation context. The specialized agents
generatate the response and return it to the supervisor, which then returns the final response to the user.
"""

from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, create_react_agent
from loguru import logger
from pydantic import Field, SecretStr
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
    logger.debug(f"Updating current agent from {left} to {right}")
    return right


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: Annotated[str, update_current_agent]
    remaining_steps: int


# Agent tool functions using InjectedState
def support_agent(state: Annotated[dict, InjectedState]) -> str:
    """Handle general questions, provide customer support, and handle routine inquiries."""
    logger.info("🤖 Support Agent activated")

    try:
        # Get the model
        model = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        # Create system message
        system_message = """You are a Support Agent. You help users with general questions,
        provide customer support, and handle routine inquiries. Be helpful and friendly in your responses.

        Provide direct assistance for basic questions and customer support issues."""

        # Get recent messages for context (only HumanMessage and previous agent responses)
        messages = state.get("messages", [])
        relevant_messages = [
            msg
            for msg in messages
            if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
        ][-2:]

        # Invoke model with system message and recent conversation
        response = model.invoke([SystemMessage(content=system_message), *relevant_messages])

        # Update current agent in state
        state["current_agent"] = "Support Agent"

    except Exception as e:
        logger.error(f"Error in support_agent: {e}")
        state["current_agent"] = "Support Agent"
        return (
            "[Support Agent]: I apologize, but I'm experiencing technical difficulties. How can I help "
            "you today?"
        )
    else:
        return f"[Support Agent]: {response.content}"


def research_agent(state: Annotated[dict, InjectedState]) -> str:
    """Conduct analysis, research, and provide detailed insights."""
    logger.info("🔬 Research Agent activated")

    try:
        # Get the model
        model = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        # Create system message
        system_message = """You are a Research Agent. You are a research specialist who analyzes
        complex topics and provides detailed insights. Focus on providing thorough, well-researched
        responses with analysis and context.

        Conduct deep analysis and provide comprehensive research-based responses."""

        # Get recent messages for context (only HumanMessage and previous agent responses)
        messages = state.get("messages", [])
        relevant_messages = [
            msg
            for msg in messages
            if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
        ][-2:]

        # Invoke model with system message and recent conversation
        response = model.invoke([SystemMessage(content=system_message), *relevant_messages])

        # Update current agent in state
        state["current_agent"] = "Research Agent"

    except Exception as e:
        logger.error(f"Error in research_agent: {e}")
        state["current_agent"] = "Research Agent"
        return (
            "[Research Agent]: I apologize, but I'm experiencing technical difficulties. Let me help you "
            "with your research needs."
        )
    else:
        return f"[Research Agent]: {response.content}"


def manager_agent(state: Annotated[dict, InjectedState]) -> str:
    """Handle escalated issues and make strategic decisions."""
    logger.info("👔 Manager Agent activated")

    try:
        # Get the model
        model = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        # Create system message
        system_message = """You are a Manager Agent. You are a senior manager who handles escalated
        issues and makes strategic decisions. Provide authoritative guidance and make clear decisions
        when needed.

        Focus on high-level decision making and strategic guidance."""

        # Get recent messages for context (only HumanMessage and previous agent responses)
        messages = state.get("messages", [])
        relevant_messages = [
            msg
            for msg in messages
            if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
        ][-2:]

        # Invoke model with system message and recent conversation
        response = model.invoke([SystemMessage(content=system_message), *relevant_messages])

        # Update current agent in state
        state["current_agent"] = "Manager Agent"

    except Exception as e:
        logger.error(f"Error in manager_agent: {e}")
        state["current_agent"] = "Manager Agent"
        return (
            "[Manager Agent]: I apologize, but I'm experiencing technical difficulties. Let me help you "
            "with management decisions."
        )
    else:
        return f"[Manager Agent]: {response.content}"


# Tools list for the supervisor
AGENT_TOOLS = [support_agent, research_agent, manager_agent]


def post_model_hook(_state: Annotated[dict, InjectedState]) -> None:
    """Post model hook to log activation of the supervisor model."""
    logger.info("Supervisor model activated")
    return {}


def create_tool_handoff_graph() -> Callable:
    """Create the LangGraph workflow using supervisor pattern with create_react_agent."""
    # Initialize the supervisor model
    supervisor_model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,  # Lower temperature for more consistent routing decisions
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    # Use create_react_agent for the supervisor pattern
    memory = MemorySaver()
    return create_react_agent(
        supervisor_model,
        AGENT_TOOLS,
        state_schema=AgentState,
        checkpointer=memory,
        post_model_hook=post_model_hook,
    )


def main() -> None:
    """Main function to run the tool-based agent handoff example."""
    logger.info("🛠️  Multi-Agent Handoff System with Tool Calling (Supervisor Pattern) Started!")
    logger.info("💡 Try saying:")
    logger.info("  - 'I need research on AI trends' (Supervisor will route to Research Agent)")
    logger.info("  - 'Escalate this to manager' (Supervisor will route to Manager Agent)")
    logger.info("  - 'Basic help please' (Supervisor will route to Support Agent)")
    logger.info("Type 'quit' to exit\\n")

    # Create the workflow
    app = create_tool_handoff_graph()

    # Initialize conversation state
    thread_id = "supervisor-tool-conversation-1"
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
            # Create supervisor system message
            supervisor_system_msg = SystemMessage(
                content="""You are a supervisor that routes conversations between specialized agents.

Available agents (tools):
- support_agent: Handles general questions, basic help, and customer support
- research_agent: Conducts analysis, research, and provides detailed insights
- manager_agent: Makes decisions, handles escalations, and provides strategic guidance

Based on the user's message, determine which agent is best suited to handle their request:
1. Use support_agent for general help, basic questions, routine matters
2. Use research_agent for research, analysis, investigation, complex topics
3. Use manager_agent for decisions, escalations, strategic matters, urgent issues

Always call exactly one agent tool to handle each user request."""
            )

            # Create state with system message and user message
            state = {
                "messages": [supervisor_system_msg, HumanMessage(content=user_input)],
                "current_agent": "Supervisor",
                "remaining_steps": 10,
            }

            # Run the workflow
            result = app.invoke(state, {**config, "recursion_limit": 10})

            # Print the agent responses
            if not result.get("messages"):
                logger.info("No messages returned from the workflow.")
                continue
                # Find and display agent responses

            agent_responses = [
                msg.content
                for msg in result.get("messages", [])
                if hasattr(msg, "content")
                and msg.content
                and (
                    "[Support Agent]" in msg.content
                    or "[Research Agent]" in msg.content
                    or "[Manager Agent]" in msg.content
                )
            ]

            # Display agent responses
            if agent_responses:
                for response in agent_responses:
                    logger.info(response)
            else:
                # Fallback: show the last AI message
                for msg in reversed(result["messages"]):
                    if (
                        isinstance(msg, AIMessage)
                        and msg.content
                        and not msg.content.startswith("You are a supervisor")
                    ):
                        logger.info(f"\\n{msg.content}")
                        break

            # Show current agent info
            current_agent = result.get("current_agent", "Supervisor")
            logger.info(f"💡 Last Active Agent: {current_agent}")
            logger.info("-" * 50)

        except Exception as e:
            logger.exception("❌ Error:", e)
            logger.info("Make sure you have configured your .env file with OPENAI_API_KEY")


if __name__ == "__main__":
    main()
