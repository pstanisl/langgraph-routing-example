"""Agent configuration and settings."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
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


class AgentState(dict):
    """State for agent workflows."""

    messages: Annotated[list[BaseMessage], add_messages]


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str = Field(..., description="Unique name for the agent")
    description: str = Field(..., description="Description of what the agent does")
    system_message: str = Field(..., description="System message defining agent behavior")
    emoji: str = Field(default="🤖", description="Emoji representing the agent")
