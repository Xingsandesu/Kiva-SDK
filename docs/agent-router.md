# AgentRouter - Modular Multi-file Applications

`AgentRouter` is a modular router provided by the Kiva SDK, inspired by FastAPI's `APIRouter`. It allows you to organize agents into multiple files, enabling you to build scalable large-scale applications.

## Why use AgentRouter?

As your project grows, defining all agents in a single file becomes difficult to maintain:

```python
# ❌ Not Recommended: All agents in one file
kiva = Kiva(...)

@kiva.agent("weather_forecast", "...")
def forecast(): ...

@kiva.agent("weather_alerts", "...")
def alerts(): ...

@kiva.agent("math_add", "...")
def add(): ...

@kiva.agent("math_multiply", "...")
def multiply(): ...

# ... many more agents
```

With `AgentRouter`, you can split agents into functional modules:

```
myapp/
├── main.py              # Entry point
├── agents/
│   ├── __init__.py
│   ├── weather.py       # Weather-related agents
│   ├── math.py          # Math-related agents
│   └── search.py        # Search-related agents
```

## Basic Usage

### Create a Router

```python
from kiva import AgentRouter

# Create a router with a prefix and tags
router = AgentRouter(prefix="weather", tags=["weather", "forecast"])
```

### Register an Agent

```python
# Single-tool agent - Decorating a function
@router.agent("forecast", "Get weather forecast")
def get_forecast(city: str) -> str:
    """Get the weather forecast for a city"""
    return f"{city}: Sunny, 25°C"

# Multi-tool agent - Decorating a class
@router.agent("calculator", "Mathematical calculations")
class Calculator:
    def add(self, a: int, b: int) -> int:
        """Addition"""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiplication"""
        return a * b
```

### Use in Kiva

```python
from kiva import Kiva
from agents.weather import weather_router
from agents.math import math_router

kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")

# Include routers
kiva.include_router(weather_router)
kiva.include_router(math_router)

# Run
kiva.run("How is the weather in Beijing? Also, calculate 15 * 8")
```

## Prefix Naming

The `prefix` parameter of `AgentRouter` is automatically prepended to all agent names:

```python
router = AgentRouter(prefix="weather")

@router.agent("forecast", "Weather forecast")  # Actual name: weather_forecast
def forecast(): ...

@router.agent("alerts", "Weather alerts")    # Actual name: weather_alerts
def alerts(): ...
```

You can also add an additional prefix when using `include_router`:

```python
kiva.include_router(weather_router, prefix="v2")
# weather_forecast -> v2_weather_forecast
```

## Nested Routers

Routers can be nested to achieve finer-grained modularity:

```python
# agents/weather/__init__.py
from kiva import AgentRouter
from .forecast import forecast_router
from .alerts import alerts_router

weather_router = AgentRouter(prefix="weather")
weather_router.include_router(forecast_router)
weather_router.include_router(alerts_router)

# agents/weather/forecast.py
forecast_router = AgentRouter(prefix="forecast")

@forecast_router.agent("daily", "Daily forecast")
def daily(city: str) -> str: ...

@forecast_router.agent("weekly", "Weekly forecast")
def weekly(city: str) -> str: ...
```

Final agent names: `weather_forecast_daily`, `weather_forecast_weekly`

## Full Example

### Project Structure

```
myapp/
├── main.py
└── agents/
    ├── __init__.py
    ├── weather.py
    └── math.py
```

### agents/weather.py

```python
from kiva import AgentRouter

router = AgentRouter(prefix="weather", tags=["weather"])

@router.agent("forecast", "Get weather forecast")
def get_forecast(city: str) -> str:
    """Get the weather forecast for a specific city"""
    forecasts = {
        "beijing": "Beijing: Sunny, 25°C",
        "tokyo": "Tokyo: Cloudy, 22°C",
    }
    return forecasts.get(city.lower(), f"{city}: No data available")

@router.agent("alerts", "Get weather alerts")
def get_alerts(region: str) -> str:
    """Get weather alerts for a specific region"""
    return f"{region}: No alerts"
```

### agents/math.py

```python
from kiva import AgentRouter

router = AgentRouter(prefix="math", tags=["math"])

@router.agent("calculator", "Math calculator")
class Calculator:
    def calculate(self, expression: str) -> str:
        """Evaluate a mathematical expression"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Calculation error: {e}"
    
    def add(self, a: int, b: int) -> int:
        """Addition"""
        return a + b
```

### main.py

```python
from kiva import Kiva
from agents.weather import router as weather_router
from agents.math import router as math_router

def create_app() -> Kiva:
    kiva = Kiva(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        model="gpt-4o",
    )
    
    kiva.include_router(weather_router)
    kiva.include_router(math_router)
    
    return kiva

if __name__ == "__main__":
    app = create_app()
    app.run("How is the weather in Beijing? Calculate 100 / 4")
```

## API Reference

### AgentRouter

```python
class AgentRouter:
    def __init__(
        self,
        prefix: str = "",           # Prefix for agent names
        tags: list[str] | None = None,  # Categorization tags
    ): ...
    
    def agent(
        self,
        name: str,          # Agent name
        description: str,   # Agent description
    ) -> Callable: ...
    
    def include_router(
        self,
        router: AgentRouter,  # Sub-router to include
        prefix: str = "",     # Additional prefix
    ) -> None: ...
    
    def get_agents(self) -> list[AgentDefinition]: ...
```

### Kiva.include_router

```python
def include_router(
    self,
    router: AgentRouter,  # Router to include
    prefix: str = "",     # Additional prefix
) -> Kiva: ...  # Returns self, supports method chaining
```

## Best Practices

1. **Group by functionality**: Keep related agents in the same router.
2. **Use meaningful prefixes**: Prefixes should clearly express the module's purpose.
3. **Keep routers independent**: Each router should be self-contained and not depend on other routers.
4. **Method chaining**: `include_router` returns `self`, allowing for concise setup.

```python
kiva.include_router(weather_router) \
    .include_router(math_router) \
    .include_router(search_router)
```
