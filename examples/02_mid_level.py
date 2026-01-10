"""Mid-Level API Example - Using run_with_console.

For async workflows that still want rich console visualization.
"""

import asyncio

from kiva import ChatOpenAI, create_agent, run_with_console, tool

API_BASE = ""
API_KEY = ""
MODEL = ""


def create_model() -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=0.7
    )


# Define tools
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: Sunny, 25°C"


@tool
def search_info(query: str) -> str:
    """Search for relevant information."""
    info_db = {
        "python": "Python is an elegant and concise programming language",
        "langchain": "LangChain is a framework for building LLM applications",
    }
    for key, value in info_db.items():
        if key in query.lower():
            return value
    return f"No information found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"


# Create agents
def create_weather_agent():
    """Create a weather information agent."""
    agent = create_agent(model=create_model(), tools=[get_weather])
    agent.name = "weather_agent"
    agent.description = "Gets weather information for cities"
    return agent


def create_search_agent():
    """Create a search agent."""
    agent = create_agent(model=create_model(), tools=[search_info])
    agent.name = "search_agent"
    agent.description = "Searches for information"
    return agent


def create_calculator_agent():
    """Create a calculator agent."""
    agent = create_agent(model=create_model(), tools=[calculate])
    agent.name = "calculator_agent"
    agent.description = "Performs mathematical calculations"
    return agent


async def main():
    """Run the orchestration with console output."""
    agents = [
        create_weather_agent(),
        create_search_agent(),
        create_calculator_agent(),
    ]

    await run_with_console(
        prompt="What's the weather in Beijing? Also search for Python info",
        agents=agents,
        base_url=API_BASE,
        api_key=API_KEY,
        model_name=MODEL,
    )


if __name__ == "__main__":
    asyncio.run(main())
