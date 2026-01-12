# Kiva SDK

> ⚠️ **Important Notice**: This project is currently in a rapid iteration/experimental phase, and the provided API may undergo disruptive changes at any time.

A multi-agent orchestration SDK for building intelligent workflows

## Features

- **Three Workflow Patterns**: Router (simple), Supervisor (parallel), and Parliament (deliberative)
- **Automatic Complexity Analysis**: Intelligent workflow selection based on task complexity
- **Modular Architecture**: AgentRouter for organizing agents across multiple files
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

### Modular Application with AgentRouter

For larger applications, use `AgentRouter` to organize agents across multiple files:

```python
# agents/weather.py
from kiva import AgentRouter

router = AgentRouter(prefix="weather")

@router.agent("forecast", "Gets weather forecasts")
def get_forecast(city: str) -> str:
    """Get weather forecast for a city."""
    return f"Sunny, 25°C in {city}"
```

```python
# agents/math.py
from kiva import AgentRouter

router = AgentRouter(prefix="math")

@router.agent("calculator", "Performs calculations")
class Calculator:
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
```

```python
# main.py
from kiva import Kiva
from agents.weather import router as weather_router
from agents.math import router as math_router

kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")

kiva.include_router(weather_router)
kiva.include_router(math_router)

kiva.run("What's the weather in Tokyo? Calculate 15 * 8")
```

See [AgentRouter Documentation](docs/agent-router.md) for more details.
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
