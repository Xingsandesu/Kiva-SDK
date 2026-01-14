# Kiva SDK

> ⚠️ **Important Notice**: This project is currently in a rapid iteration/experimental phase, and the provided API may undergo disruptive changes at any time.

A multi-agent orchestration SDK for building intelligent workflows

## Features

- **Three Workflow Patterns**: Router (simple), Supervisor (parallel), and Parliament (deliberative)
- **Automatic Complexity Analysis**: Intelligent workflow selection based on task complexity
- **Parallel Agent Instances**: Spawn multiple instances of the same agent for parallel subtask execution
- **Output Verification**: Dual-layer verification system with automatic retry and self-healing
- **Custom Verifiers**: Extensible verification rules via `@kiva.verifier` decorator
- **Modular Architecture**: AgentRouter for organizing agents across multiple files
- **Rich Console Output**: Beautiful terminal visualization (optional)
- **Error Recovery**: Built-in error handling with recovery suggestions

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

### Basic Usage

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

# Silent mode - no console output
result = kiva.run("What's the weather in Beijing?", console=False)
print(result)
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

## Workflow Patterns

### Router Workflow
Routes tasks to a single most appropriate agent. Best for simple, single-domain queries.

### Supervisor Workflow
Coordinates multiple agents executing in parallel. Supports spawning multiple instances of the same agent for parallel subtask processing. Ideal for multi-faceted tasks that can be decomposed into independent subtasks.

### Parliament Workflow
Implements iterative deliberation with conflict resolution. Designed for complex reasoning tasks requiring consensus or validation.

## Output Verification

Kiva includes a dual-layer verification system that automatically validates agent outputs:

```python
from kiva import Kiva, VerificationResult, VerificationStatus

kiva = Kiva(
    base_url="...",
    api_key="...",
    model="gpt-4o",
    max_iterations=3,  # Global retry limit
)

# Custom verifier for additional validation
@kiva.verifier("length_check")
def check_length(task: str, output: str, context: dict | None = None) -> VerificationResult:
    """Ensure output meets minimum length."""
    if len(output) < 50:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason="Output too short",
            improvement_suggestions=["Provide more detailed response"],
        )
    return VerificationResult(status=VerificationStatus.PASSED)

@kiva.agent("researcher", "Researches topics", max_iterations=5)  # Per-agent limit
def research(topic: str) -> str:
    """Research a topic."""
    return f"Research results for {topic}..."

# Run with per-execution limit
result = kiva.run("Research AI trends", max_iterations=4)
```

See [Output Verification Documentation](docs/verification.md) for details.

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

## Async Support

For async contexts, use `run_async`:

```python
import asyncio
from kiva import Kiva

async def main():
    kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")
    
    @kiva.agent("weather", "Gets weather")
    def get_weather(city: str) -> str:
        return f"{city}: Sunny"
    
    result = await kiva.run_async("Weather in Tokyo?", console=False)
    print(result)

asyncio.run(main())
```

## Documentation

- [AgentRouter - Modular Applications](docs/agent-router.md)
- [Output Verification](docs/verification.md)
- [Parallel Agent Instances](docs/parallel-instances.md)
- [E2E Testing Guide](docs/e2e-testing-guide.md)

## Testing

The SDK includes comprehensive unit and end-to-end tests:

```bash
# Unit tests (no API required)
uv run --dev pytest tests/ -v --ignore=tests/e2e/

# End-to-end tests (requires API configuration)
export KIVA_API_BASE="http://your-api-endpoint/v1"
export KIVA_API_KEY="your-api-key"
export KIVA_MODEL="your-model-name"

uv run --dev pytest tests/e2e/ -v
```

See [E2E Testing Guide](docs/e2e-testing-guide.md) for more details.

## License

MIT License
