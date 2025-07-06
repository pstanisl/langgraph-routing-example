"""LangGraph Tool-Based Handoff Router Example.

This example demonstrates tool-based handoffs where agents transfer is implemented using tools. The agent
generates the tool message with the tool call ID, and the router uses this to transfer control to the
appropriate agent.

The advantage of this approach is that it allows for more flexible way how to define the agents and their
handoffs, as well as the ability to easily add new agents and tools without changing the prompt of the router.
"""

from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
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


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str = Field(..., description="Unique name for the agent")
    description: str = Field(..., description="Description of what the agent does")
    system_message: str = Field(..., description="System message defining agent behavior")
    emoji: str = Field(default="🤖", description="Emoji representing the agent")


class AgentRegistry(BaseModel):
    """Registry of available agents."""

    agents: dict[str, AgentConfig] = Field(default_factory=dict)

    def register_agent(self, config: AgentConfig) -> None:
        """Register a new agent configuration."""
        self.agents[config.name] = config

    def get_agent_names(self) -> list[str]:
        """Get list of all registered agent names."""
        return list(self.agents.keys())

    def get_agent_config(self, name: str) -> AgentConfig | None:
        """Get configuration for a specific agent."""
        return self.agents.get(name)


# Load settings
settings = Settings()

# Initialize agent registry with default agents
agent_registry = AgentRegistry()
agent_registry.register_agent(
    AgentConfig(
        name="support_agent",
        description="Transfer to support agent for general help and customer service.",
        system_message="""You are a Support Agent providing customer service and general help.

Provide helpful, friendly support for the user's request. Focus on:
- General questions and basic support
- Account issues and routine inquiries
- Product information and guidance

Give a direct, helpful response to the user's question.""",
        emoji="🤖",
    )
)

agent_registry.register_agent(
    AgentConfig(
        name="research_agent",
        description="Transfer to research agent for analysis and detailed research.",
        system_message="""You are a Research Agent specializing in analysis and detailed insights.

Provide thorough research and analysis for the user's request. Focus on:
- In-depth research and analysis
- Complex topic investigation
- Data-driven insights and recommendations

Give a comprehensive, well-researched response to the user's question.""",
        emoji="🔬",
    )
)

agent_registry.register_agent(
    AgentConfig(
        name="manager_agent",
        description="Transfer to manager agent for escalations and strategic decisions.",
        system_message="""You are a Manager Agent handling strategic decisions and escalations.

Provide authoritative guidance for the user's request. Focus on:
- Strategic guidance and decision making
- Escalation resolution
- High-level policy and direction

Give a clear, decisive response to the user's question.""",
        emoji="👔",
    )
)


def create_handoff_tool(*, agent_name: str, description: str | None = None) -> Callable:
    """Create a handoff tool that transfers control to another agent."""
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name, update={"messages": state["messages"] + [tool_message]}, graph=Command.PARENT
        )

    return handoff_tool


def create_agent_function(agent_config: AgentConfig) -> Callable:
    """Create an agent function from configuration."""

    def agent_function(state: Annotated[dict, InjectedState]) -> dict:
        logger.info(f"{agent_config.emoji} {agent_config.name.replace('_', ' ').title()} activated")

        model = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        # Get the most recent user message
        messages = state.get("messages", [])
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]

        if not user_messages:
            return "No user message found."

        user_request = user_messages[-1]

        # Provide agent response
        response = model.invoke(
            [
                SystemMessage(content=agent_config.system_message),
                user_request,
            ]
        )

        return {"messages": [response]}

    # Set function name for LangGraph node identification
    agent_function.__name__ = agent_config.name
    return agent_function


def create_handoff_tools(registry: AgentRegistry) -> list[Callable]:
    """Create handoff tools for all registered agents."""
    tools = []
    for agent_name, agent_config in registry.agents.items():
        tool = create_handoff_tool(
            agent_name=agent_name,
            description=agent_config.description,
        )
        tools.append(tool)
    return tools


def create_router(registry: AgentRegistry, tools: list[Callable]) -> dict:
    """Create a router agent with dynamic agent awareness."""
    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    # Create dynamic system message with available agents
    agent_descriptions = "\n".join(
        [
            f"- {config.name}: {config.description.replace('Transfer to ', '').replace(' agent', '')}"
            for config in registry.agents.values()
        ]
    )

    system_message = SystemMessage(
        content=f"""You are a supervisor routing conversations between specialized agents. The transfers are
handled by tools. Do not respond directly to the user.

Available agents:
{agent_descriptions}

Consider the user's latest message and the conversation flow to make the best routing decision."""
    )

    return create_react_agent(name="router", model=model, tools=tools, prompt=system_message)


def create_workflow(registry: AgentRegistry) -> CompiledStateGraph:
    """Create workflow with dynamic agent registration."""
    # Create tools for all registered agents
    handoff_tools = create_handoff_tools(registry)

    # Create workflow
    workflow = StateGraph(MessagesState)

    # Add router with dynamic destinations
    agent_names = registry.get_agent_names()
    workflow.add_node("router", create_router(registry, handoff_tools), destinations=agent_names)

    # Add all registered agents dynamically
    for agent_name, agent_config in registry.agents.items():
        agent_function = create_agent_function(agent_config)
        workflow.add_node(agent_function)
        # Each agent ends the conversation
        workflow.add_edge(agent_name, END)

    workflow.set_entry_point("router")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def main() -> None:
    """Main function to run the tool-based handoff example."""
    logger.info("🚀 Multi-Agent Tool-Based Handoff System Started!")
    logger.info("💡 Available agents:")

    # Dynamic agent listing
    for agent_config in agent_registry.agents.values():
        agent_display_name = agent_config.name.replace("_", " ").title()
        logger.info(f"  {agent_config.emoji} {agent_display_name}")

    logger.info("Type 'quit' to exit\n")

    # Create the workflow with registered agents
    app = create_workflow(agent_registry)

    # Initialize conversation state
    thread_id = "tool-handoff-conversation-1"
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
            # Create state with system message and user message
            state = {"messages": [HumanMessage(content=user_input)]}

            # Run the supervisor agent
            result = app.invoke(state, {**config, "recursion_limit": 10})
            # logger.debug(f"🔄 Result: {result}")
            # Print the conversation
            if not result.get("messages"):
                logger.info("No messages in the conversation.")
                continue

            # Get all registered agent names for response filtering
            registered_agent_names = agent_registry.get_agent_names()

            agent_responses = [
                msg.content
                for msg in result.get("messages", [])
                if msg.name in registered_agent_names and hasattr(msg, "content") and msg.content
            ]

            # Display agent responses
            if agent_responses:
                for response in agent_responses:
                    logger.info(f"\n{response}")
            else:
                # Fallback: show the last AI message if no tool responses found
                for msg in reversed(result["messages"]):
                    if (
                        hasattr(msg, "content")
                        and msg.content
                        and not msg.content.startswith("You are a Supervisor")
                    ):
                        logger.info(f"\n{msg.content}")
                        break

            logger.info("-" * 50)

        except Exception as e:
            logger.exception("❌ Error:", e)
            logger.info("Make sure you have configured your .env file with OPENAI_API_KEY")


if __name__ == "__main__":
    main()
