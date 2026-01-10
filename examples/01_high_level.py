"""High-Level API Example - Using the Kiva Client.

The simplest way to use Kiva, ideal for rapid prototyping and development.
"""

from kiva import Kiva

API_BASE = ""
API_KEY = ""
MODEL = ""

# Initialize the client
kiva = Kiva(
    base_url=API_BASE,
    api_key=API_KEY,
    model=MODEL,
)


# Single-tool agent - decorate a function directly
@kiva.agent("weather", "Gets weather information for cities")
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: Sunny, 25°C"


@kiva.agent("search", "Searches for information")
def search_info(query: str) -> str:
    """Search for relevant information."""
    info_db = {
        "python": "Python is an elegant and concise programming language",
        "langchain": "LangChain is a framework for building LLM applications",
        "kiva": "Kiva is a multi-agent orchestration SDK",
    }
    for key, value in info_db.items():
        if key in query.lower():
            return value
    return f"No information found for: {query}"


# Multi-tool agent - decorate a class
@kiva.agent("math", "Performs mathematical calculations")
class MathTools:
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


if __name__ == "__main__":
    # Run with rich console output (default)
    kiva.run("What's the weather in Beijing? Also calculate 15 * 8")

    # Silent mode - no rich console output
    # result = kiva.run("Search for Python information", console=False)
    # print(result)
