from langchain_core.messages import HumanMessage
from loguru import logger

from agent_experiment.examples.router_handoff_command import create_handoff_graph


def test_handoff() -> None:
    """Test the handoff functionality with different scenarios."""
    app = create_handoff_graph()

    # Test scenarios
    scenarios = [
        ("Basic support", "I need help with my account"),
        ("Research request", "Can you research AI trends for me?"),
        ("Manager escalation", "I need to escalate this urgent issue to a manager"),
        ("Goodbye", "Thanks, goodbye!"),
    ]

    for scenario_name, user_message in scenarios:
        logger.info(f"\n=== Testing {scenario_name} ===")
        config = {"configurable": {"thread_id": f"test-{scenario_name.lower().replace(' ', '-')}"}}

        state = {"messages": [HumanMessage(content=user_message)], "current_agent": "Support Agent"}

        try:
            result = app.invoke(state, config)

            logger.info(f"Final agent: {result['current_agent']}")
            logger.info(f"Messages: {len(result['messages'])}")

            # Show the last agent response
            if result["messages"]:
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    logger.info(f"Response: {last_msg.content[:100]}...")
        except KeyboardInterrupt:
            logger.info("Test interrupted")
            break


if __name__ == "__main__":
    test_handoff()
