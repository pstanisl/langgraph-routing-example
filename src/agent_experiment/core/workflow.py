"""Workflow creation and agent management utilities."""

from collections.abc import Callable
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from loguru import logger

from .config import AgentConfig, Settings
from .registry import AgentRegistry


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


def create_agent_function(agent_config: AgentConfig, settings: Settings) -> Callable:
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


def create_router(registry: AgentRegistry, tools: list[Callable], settings: Settings) -> dict:
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


def create_workflow(registry: AgentRegistry, settings: Settings) -> CompiledStateGraph:
    """Create workflow with dynamic agent registration."""
    # Create tools for all registered agents
    handoff_tools = create_handoff_tools(registry)

    # Create workflow
    workflow = StateGraph(MessagesState)

    # Add router with dynamic destinations
    agent_names = registry.get_agent_names()
    workflow.add_node("router", create_router(registry, handoff_tools, settings), destinations=agent_names)

    # Add all registered agents dynamically
    for agent_name, agent_config in registry.agents.items():
        agent_function = create_agent_function(agent_config, settings)
        workflow.add_node(agent_function)
        # Each agent ends the conversation
        workflow.add_edge(agent_name, END)

    workflow.set_entry_point("router")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
