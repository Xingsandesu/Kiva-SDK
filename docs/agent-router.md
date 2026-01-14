# AgentRouter (Modular Multi-File Apps)

`AgentRouter` helps you organize agents across multiple modules/files (inspired by FastAPI's `APIRouter`) so larger projects stay maintainable.

## Why AgentRouter?

As a project grows, keeping all agents in one file becomes hard to maintain:

```python
# Not recommended: all agents in one file
kiva = Kiva(...)

@kiva.agent("weather_forecast", "...")
def forecast(): ...

@kiva.agent("weather_alerts", "...")
def alerts(): ...

@kiva.agent("math_add", "...")
def add(): ...

@kiva.agent("math_multiply", "...")
def multiply(): ...

# ... more agents
```

With `AgentRouter`, you can split agents by domain:

```
myapp/
├── main.py              # entrypoint
├── agents/
│   ├── __init__.py
│   ├── weather.py       # weather agents
│   ├── math.py          # math agents
│   └── search.py        # search agents
```

## Basic Usage

### Create a Router

```python
from kiva import AgentRouter

router = AgentRouter(prefix="weather", tags=["weather", "forecast"])
```

### Register Agents

```python
# Single-tool agent - decorate a function
@router.agent("forecast", "Get weather forecasts")
def get_forecast(city: str) -> str:
    """Get a city weather forecast."""
    return f"{city}: Sunny, 25°C"

# Multi-tool agent - decorate a class
@router.agent("calculator", "Math tools")
class Calculator:
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
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
kiva.run("What's the weather in Beijing? Also calculate 15 * 8")
```

## Prefix Naming

The router `prefix` is prepended to all agent names:

```python
router = AgentRouter(prefix="weather")

@router.agent("forecast", "Weather forecast")  # actual name: weather_forecast
def forecast(): ...

@router.agent("alerts", "Weather alerts")      # actual name: weather_alerts
def alerts(): ...
```

You can add an extra prefix when including a router:

```python
kiva.include_router(weather_router, prefix="v2")
# weather_forecast -> v2_weather_forecast
```

## Nested Routers

Routers can be nested for finer modularization:

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

@router.agent("forecast", "Get weather forecasts")
def get_forecast(city: str) -> str:
    """Get a forecast for a city."""
    forecasts = {
        "beijing": "Beijing: Sunny, 25°C",
        "tokyo": "Tokyo: Cloudy, 22°C",
    }
    return forecasts.get(city.lower(), f"{city}: No data")

@router.agent("alerts", "Get weather alerts")
def get_alerts(region: str) -> str:
    """Get alerts for a region."""
    return f"{region}: No alerts"
```

### agents/math.py

```python
from kiva import AgentRouter

router = AgentRouter(prefix="math", tags=["math"])

@router.agent("calculator", "Math calculator")
class Calculator:
    def calculate(self, expression: str) -> str:
        """Evaluate a math expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Calculation error: {e}"
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
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
    app.run("What's the weather in Beijing? Calculate 100 / 4")
```

## API Reference

### AgentRouter

```python
class AgentRouter:
    def __init(
        self,
        prefix: str = "",
        tags: list[str] | None = None,
    ): ...
    
    def agent(
        self,
        name: str,
        description: str,
    ) -> Callable: ...
    
    def include_router(
        self,
        router: AgentRouter,
        prefix: str = "",
    ) -> None: ...
    
    def get_agents(self) -> list[AgentDefinition]: ...
```

### Kiva.include_router

```python
def include_router(
    self,
    router: AgentRouter,
    prefix: str = "",
) -> Kiva: ...
```

## Best Practices

1. Split by domain: keep related agents together in one router.
2. Use meaningful prefixes: prefixes should clearly reflect module intent.
3. Keep routers independent: avoid tight coupling between routers.
4. Chain includes: `include_router` returns `self` for chaining.

```python
kiva.include_router(weather_router) \
    .include_router(math_router) \
    .include_router(search_router)
```
