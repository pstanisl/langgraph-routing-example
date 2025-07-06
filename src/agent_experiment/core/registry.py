"""Agent registry for managing multiple agents."""

from pydantic import BaseModel, Field

from .config import AgentConfig


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


def create_default_registry() -> AgentRegistry:
    """Create a registry with default agents."""
    registry = AgentRegistry()

    registry.register_agent(
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

    registry.register_agent(
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

    registry.register_agent(
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

    return registry
