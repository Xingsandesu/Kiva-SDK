"""Modular Application Example - Using AgentRouter for multi-file organization.

This example demonstrates how to organize agents across multiple files
using AgentRouter, similar to FastAPI's APIRouter pattern.

Project structure for a real application:
    myapp/
    ├── main.py              # Entry point with Kiva instance
    ├── agents/
    │   ├── __init__.py
    │   ├── weather.py       # Weather-related agents
    │   ├── math.py          # Math-related agents
    │   └── search.py        # Search-related agents
"""

from kiva import AgentRouter, Kiva

API_BASE = ""
API_KEY = ""
MODEL = ""

# ============================================================
# agents/weather.py - Weather module
# ============================================================
weather_router = AgentRouter(prefix="weather", tags=["weather", "forecast"])


@weather_router.agent("forecast", "Gets weather forecasts for cities")
def get_forecast(city: str) -> str:
    """Get weather forecast for a city."""
    forecasts = {
        "beijing": "Beijing: Sunny, 25°C, light breeze",
        "tokyo": "Tokyo: Cloudy, 22°C, chance of rain",
        "london": "London: Rainy, 15°C, bring an umbrella",
    }
    return forecasts.get(city.lower(), f"{city}: Weather data unavailable")


@weather_router.agent("alerts", "Gets weather alerts for regions")
def get_alerts(region: str) -> str:
    """Get weather alerts for a region."""
    return f"No active weather alerts for {region}"


# ============================================================
# agents/math.py - Math module
# ============================================================
math_router = AgentRouter(prefix="math", tags=["math", "calculation"])


@math_router.agent("calculator", "Performs mathematical calculations")
class Calculator:
    """Multi-tool calculator agent."""

    def calculate(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Calculation error: {e}"

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


# ============================================================
# agents/search.py - Search module
# ============================================================
search_router = AgentRouter(prefix="search", tags=["search", "info"])


@search_router.agent("web", "Searches the web for information")
def web_search(query: str) -> str:
    """Search the web for information."""
    info_db = {
        "python": "Python is an elegant and concise programming language",
        "langchain": "LangChain is a framework for building LLM applications",
        "kiva": "Kiva is a multi-agent orchestration SDK",
    }
    for key, value in info_db.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


# ============================================================
# main.py - Application entry point
# ============================================================
def create_app() -> Kiva:
    """Create and configure the Kiva application."""
    kiva = Kiva(
        base_url=API_BASE,
        api_key=API_KEY,
        model=MODEL,
    )

    # Include all routers
    kiva.include_router(weather_router)
    kiva.include_router(math_router)
    kiva.include_router(search_router)

    return kiva


if __name__ == "__main__":
    app = create_app()

    # Run with rich console output
    app.run("What's the weather in Beijing? Also calculate 15 * 8")
