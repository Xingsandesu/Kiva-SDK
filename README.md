# Kiva SDK

> ⚠️ **Important Notice**: This project is currently in a rapid iteration/experimental phase, and the provided API may undergo disruptive changes at any time.

A multi-agent orchestration SDK for building intelligent workflows

## Features

- **Three Workflow Patterns**: Router (simple), Supervisor (parallel), and Parliament (deliberative)
- **Automatic Complexity Analysis**: Intelligent workflow selection based on task complexity
- **Parallel Agent Instances**: Spawn multiple instances of the same agent for parallel subtask execution
- **Modular Architecture**: AgentRouter for organizing agents across multiple files
- **Streaming Events**: Real-time execution monitoring with structured events
- **Rich Console Output**: Beautiful terminal visualization (optional)
- **Error Recovery**: Built-in error handling with recovery suggestions
- **Flexible API Levels**: High-level client, mid-level console, and low-level streaming

## Installation

```bash
uv add kiva-sdk
```

## Setup & Configuration

Before running the SDK, you need to configure your API credentials using environment variables:

```bash
export KIVA_API_BASE="http://your-api-endpoint/v1"
export KIVA_API_KEY="your-api-key"
export KIVA_MODEL="your-model-name"
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
                if event.data.get("parallel_strategy") != "none":
                    print(f"Parallel Strategy: {event.data['parallel_strategy']}")
                    print(f"Total Instances: {event.data.get('total_instances', 1)}")
            case "agent_start":
                print(f"\nAgent started: {event.data.get('agent_id')}")
            case "agent_end":
                print(f"\nAgent finished")
            # Parallel instance events
            case "instance_spawn":
                print(f"\nInstance spawned: {event.data.get('instance_id')}")
            case "instance_start":
                print(f"Instance started: {event.data.get('instance_id')}")
            case "instance_end":
                print(f"Instance completed: {event.data.get('instance_id')}")
            case "instance_complete":
                status = "success" if event.data.get("success") else "failed"
                print(f"Instance {event.data.get('instance_id')}: {status}")
            case "parallel_instances_start":
                print(f"\nStarting {event.data.get('instance_count')} parallel instances")
            case "parallel_instances_complete":
                print(f"All parallel instances completed")
            case "final_result":
                print(f"\n\nResult: {event.data['result']}")

asyncio.run(main())
```

## Workflow Patterns

### Router Workflow
Routes tasks to a single most appropriate agent. Best for simple, single-domain queries.

### Supervisor Workflow
Coordinates multiple agents executing in parallel. Supports spawning multiple instances of the same agent for parallel subtask processing. Ideal for multi-faceted tasks that can be decomposed into independent subtasks.

### Parliament Workflow
Implements iterative deliberation with conflict resolution. Designed for complex reasoning tasks requiring consensus or validation.

## Parallel Agent Instances

Kiva can spawn multiple instances of the same agent definition for parallel execution:

```python
# The planner automatically decides when to use parallel instances
# For example, this task might spawn 3 instances of a search agent:
kiva.run("Search for information about AI, blockchain, and quantum computing")
```

Each instance has:
- Isolated context/scratchpad
- Independent execution
- Results aggregated automatically

See [Parallel Instances Documentation](docs/parallel-instances.md) for details.

## Event Types

### Basic Events

| Event | Description |
|-------|-------------|
| `token` | Streaming token from LLM |
| `workflow_selected` | Workflow and complexity determined |
| `final_result` | Final synthesized result |
| `error` | Error occurred |

### Single-Agent Events

| Event | Description |
|-------|-------------|
| `parallel_start` | Parallel agent execution started |
| `agent_start` | Individual agent started |
| `agent_end` | Individual agent completed |
| `parallel_complete` | All parallel agents finished |

### Parallel Instance Events

| Event | Description | Data Fields |
|-------|-------------|-------------|
| `instance_spawn` | Agent instance created | `instance_id`, `agent_id`, `task` |
| `instance_start` | Instance beginning execution | `instance_id`, `agent_id`, `task` |
| `instance_end` | Instance completed task | `instance_id`, `agent_id`, `result` |
| `instance_complete` | Instance finished (success/error) | `instance_id`, `agent_id`, `success` |
| `instance_result` | Result from instance execution | `instance_id`, `agent_id`, `result`, `error` |
| `parallel_instances_start` | Batch of instances starting | `instance_count`, `agent_ids` |
| `parallel_instances_complete` | Batch of instances finished | `results` (list with `agent_id`, `instance_id`, `success`) |

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
    max_parallel_agents=5,          # Max concurrent agents/instances
):
    ...
```

## Documentation

- [AgentRouter - Modular Applications](docs/agent-router.md)
- [Parallel Agent Instances](docs/parallel-instances.md)
- [Execution Output Patterns](docs/execution-outputs.md)
- [E2E Testing Guide](docs/e2e-testing-guide.md)

## Testing

The SDK includes comprehensive unit and end-to-end tests:

```bash
# Unit tests (no API required)
uv run --dev pytest tests/ -v

# End-to-end tests (requires API configuration)
export KIVA_API_BASE="http://your-api-endpoint/v1"
export KIVA_API_KEY="your-api-key"
export KIVA_MODEL="your-model-name"

uv run --dev pytest tests/e2e/ -v
```

See [E2E Testing Guide](docs/e2e-testing-guide.md) for more details.

## License

MIT License
