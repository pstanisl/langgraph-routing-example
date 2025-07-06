"""LangGraph Tool-Based Handoff Router Example.

This example demonstrates tool-based handoffs where agents transfer is implemented using tools. The agent
generates the tool message with the tool call ID, and the router uses this to transfer control to the
appropriate agent.

The advantage of this approach is that it allows for more flexible way how to define the agents and their
handoffs, as well as the ability to easily add new agents and tools without changing the prompt of the router.
"""

from langchain_core.messages import HumanMessage
from loguru import logger

from agent_experiment.core.config import Settings
from agent_experiment.core.registry import create_default_registry
from agent_experiment.core.workflow import create_workflow

# Load settings and create registry
settings = Settings()
agent_registry = create_default_registry()


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
    app = create_workflow(agent_registry, settings)

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
