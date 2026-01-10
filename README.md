# Kiva SDK

A multi-agent orchestration SDK for building intelligent workflows

## Features

- **Three Workflow Patterns**: Router (simple), Supervisor (parallel), and Parliament (deliberative)
- **Automatic Complexity Analysis**: Intelligent workflow selection based on task complexity
- **Streaming Events**: Real-time execution monitoring with structured events
- **Rich Console Output**: Beautiful terminal visualization (optional)
- **Error Recovery**: Built-in error handling with recovery suggestions
- **Flexible API Levels**: High-level client, mid-level console, and low-level streaming

## Installation

```bash
uv add kiva-sdk
```

## Quick Start

### High-Level API (Simplest)

```python
from kiva import Kiva

kiva = Kiva(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model="gpt-4o",
)

# Single-tool agent using decorator
@kiva.agent("weather", "Gets weather information")
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 25°C in {city}"

# Multi-tool agent using class decorator
@kiva.agent("math", "Performs calculations")
class MathTools:
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

# Run with rich console output
kiva.run("What's the weather in Tokyo? Also calculate 15 * 8")
```

### Mid-Level API (Async with Console)

```python
import asyncio
from kiva import run_with_console, create_agent, ChatOpenAI, tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

async def main():
    model = ChatOpenAI(model="gpt-4o", api_key="...")
    agent = create_agent(model=model, tools=[search])
    agent.name = "search_agent"
    agent.description = "Searches for information"
    
    await run_with_console(
        prompt="Search for Python tutorials",
        agents=[agent],
        base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        model_name="gpt-4o",
    )

asyncio.run(main())
```

### Low-Level API (Full Control)

```python
import asyncio
from kiva import run, create_agent, ChatOpenAI, tool

@tool
def calculate(expr: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expr))

async def main():
    model = ChatOpenAI(model="gpt-4o", api_key="...")
    agent = create_agent(model=model, tools=[calculate])
    agent.name = "calculator"
    agent.description = "Performs calculations"
    
    async for event in run(
        prompt="Calculate 100 / 4",
        agents=[agent],
        base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        model_name="gpt-4o",
    ):
        match event.type:
            case "token":
                print(event.data["content"], end="", flush=True)
            case "workflow_selected":
                print(f"\nWorkflow: {event.data['workflow']}")
            case "agent_start":
                print(f"\nAgent started: {event.data.get('agent_id')}")
            case "agent_end":
                print(f"\nAgent finished")
            case "final_result":
                print(f"\n\nResult: {event.data['result']}")

asyncio.run(main())
```

## Workflow Patterns

### Router Workflow
Routes tasks to a single most appropriate agent. Best for simple, single-domain queries.

### Supervisor Workflow
Coordinates multiple agents executing in parallel. Ideal for multi-faceted tasks that can be decomposed into independent subtasks.

### Parliament Workflow
Implements iterative deliberation with conflict resolution. Designed for complex reasoning tasks requiring consensus or validation.

## Event Types

| Event | Description |
|-------|-------------|
| `token` | Streaming token from LLM |
| `workflow_selected` | Workflow and complexity determined |
| `parallel_start` | Parallel agent execution started |
| `agent_start` | Individual agent started |
| `agent_end` | Individual agent completed |
| `parallel_complete` | All parallel agents finished |
| `final_result` | Final synthesized result |
| `error` | Error occurred |

## Configuration

```python
async for event in run(
    prompt="Your task",
    agents=agents,
    model_name="gpt-4o",           # Lead agent model
    api_key="...",                  # API key
    base_url="...",                 # API base URL
    workflow_override="supervisor", # Force specific workflow
    max_iterations=10,              # Parliament max iterations
    max_parallel_agents=5,          # Max concurrent agents
):
    ...
```

## License

MIT License
