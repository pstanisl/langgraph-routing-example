# LangGraph Agent Handoff Example

A demonstration of multi-agent systems using LangGraph's Command objects for intelligent agent handoffs with a supervisor pattern.

> In router examples, the agents are not really "agents" in the traditional sense, but rather workflows but it is easier to refer to them as agents for consistency with the rest of the project.

## Overview

This example showcases a conversational AI system with three specialized agents that can intelligently hand off tasks to each other based on conversation context:

- **Support Agent**: Handles general questions, basic help, and customer support
- **Research Agent**: Conducts analysis, research, and provides detailed insights
- **Manager Agent**: Makes decisions, handles escalations, and provides strategic guidance

## Architecture

The system uses the **supervisor pattern** where:

1. **Supervisor**: A central router that uses an LLM with structured output to intelligently decide which agent should handle each user interaction
2. **Specialized Agents**: Simple, focused agents that execute their specific tasks without handoff logic
3. **Command Objects**: LangGraph's Command class manages both state updates and control flow in a single operation

## Features

- **Intelligent Routing**: Model-driven decisions instead of hardcoded rules
- **Clean Separation**: Agents focus on their expertise without routing logic
- **Configuration Management**: Uses pydantic-settings for environment configuration
- **Proxy Support**: Configurable base URL for proxy setups
- **Memory**: Persistent conversation state using MemorySaver

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone or download this project
2. Install dependencies:

   ```bash
   uv sync
   ```

3. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Environment Configuration

Create a `.env` file with the following settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=claude-4-sonnet
# For proxy setups, change the base URL:
# OPENAI_BASE_URL=https://your-proxy-domain.com/v1
```

## Usage

### Running Examples

Run different handoff patterns:

```bash
# Supervisor pattern (tool-based agents)
uv run agent-handoff

# Tool-based handoff with ReAct
uv run router-tools

# Router pattern with direct handoffs
uv run router-command
```

Try different types of requests:

- `"I need help with my account"` � Routes to Support Agent
- `"Can you research AI trends for me?"` � Routes to Research Agent
- `"Escalate this urgent issue to a manager"` � Routes to Manager Agent
- `"Thanks, goodbye!"` � Ends conversation

### Testing

Run the test suite to see different routing scenarios:

```bash
uv run python tests/test_handoff.py
```

## How It Works

### Supervisor Pattern

Instead of agents deciding handoffs themselves, a central supervisor makes routing decisions:

```python
def supervisor(state: AgentState) -> Command[...]:
    # Use LLM with structured output to decide routing
    model = ChatOpenAI(...).with_structured_output(NextAgent)
    response = model.invoke([system_prompt, *messages])

    # Route to selected agent or end conversation
    return Command(goto=response.next_agent, update={...})
```

### Agent Implementation

Agents are simple and focused:

```python
def create_agent(name: str, system_message: str, model: ChatOpenAI):
    def agent_node(state: AgentState) -> dict:
        # Process the request
        response = model.invoke([system_msg] + messages)
        agent_response = AIMessage(content=f"[{name}]: {response.content}")

        # Return updated state (no handoff logic needed)
        return {"messages": [agent_response], "current_agent": name}

    return agent_node
```

### Graph Structures

**Supervisor Pattern:**

```
User Input → Supervisor → [Support|Research|Manager] Agent → Supervisor → End
```

**Tool-Based Pattern:**

```
User Input → Supervisor Tool Router →  Transfer Tools → Agent → End
```

**Router Pattern:**

```
User Input → Router → Agent → End
```

## Key Components

### `NextAgent` Model

Structured output for supervisor routing decisions:

```python
class NextAgent(BaseModel):
    next_agent: Literal["support_agent", "research_agent", "manager_agent", "__end__"]
    reasoning: str
```

### `AgentState`

Graph state with proper reducers:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: Annotated[str, update_current_agent]
```

### Settings Management

Configuration using pydantic-settings:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    openai_api_key: SecretStr
    openai_base_url: str = "https://api.proxy.com/v1"
    openai_model: str = "claude-4-sonnet"
```

> Proxy tested with LiteLLM

## Project Structure

```
src/agent_experiment/
├── core/
│   ├── config.py          # Pydantic settings configuration
│   ├── registry.py        # Agent registry and factory
│   └── workflow.py        # Core workflow components
├── examples/
│   ├── agent_handoff_tools.py     # Supervisor pattern
│   ├── router_handoff_tools.py    # Tool-based handoff
│   └── router_handoff_command.py  # Router pattern
└── utils/
    ├── graph_utils.py     # Graph utilities
    └── visualize_graphs.py # Graph visualization
```

## Benefits of Different Approaches

### Supervisor Pattern

1. **Centralized Control**: All routing decisions in one place
2. **Agent Simplicity**: Agents focus purely on their domain expertise
3. **Easy Debugging**: Clear routing logic and decision points

> Disadvantage: Supervisor generates the response after each agent, which can be less efficient.

### Tool-Based Pattern

1. **LangGraph Integration**: Uses prebuilt ReAct components
2. **Tool Paradigm**: Familiar tool-calling interface
3. **Reduced Boilerplate**: Less custom graph construction

> The routes with the descriptions are defined as tools and provided to the router.

### Router Pattern

1. **Direct Handoffs**: Agents can transfer directly to each other
2. **Distributed Control**: No single point of routing failure
3. **Flexible Workflows**: Complex multi-step agent interactions

> The paths needs to be defined in the router's prompt.

## Troubleshooting

### Common Issues

- **API Key**: Ensure `OPENAI_API_KEY` is set in your `.env` file
- **Proxy Issues**: Update `OPENAI_BASE_URL` for your proxy setup
- **Model Compatibility**: Ensure your model supports structured output

### Visualization

Generate graph visualizations:

```bash
uv run visualize-graphs
```

### Debug Mode

Enable debug logging to see routing decisions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This example is provided for educational purposes. Modify and use as needed for your projects.
