"""Shared fixtures and configuration for e2e tests."""

import os
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _set_env_if_missing(key: str, value: str) -> None:
    if os.environ.get(key, "").strip():
        return
    os.environ[key] = value


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)
        if not key:
            continue
        _set_env_if_missing(key, value)


def _first_nonempty_env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return default


@pytest.fixture
def api_config():
    """Return API configuration."""
    repo_root = Path(__file__).resolve().parents[2]
    _load_env_file(repo_root / ".env.e2e")
    _load_env_file(repo_root / ".env")
    return {
        "base_url": _first_nonempty_env(
            "KIVA_API_BASE",
            "API_BASE",
            default="http://localhost:8000/v1",
        ),
        "api_key": _first_nonempty_env(
            "KIVA_API_KEY",
            "API_KEY",
            default="",
        ),
        "model": _first_nonempty_env(
            "KIVA_MODEL",
            "MODEL",
            default="gpt-4o",
        ),
    }


@pytest.fixture
def create_model(api_config):
    """Factory fixture to create ChatOpenAI model."""
    def _create_model(temperature: float = 0.7):
        return ChatOpenAI(
            model=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            temperature=temperature,
        )
    return _create_model


# Common tools for testing
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "beijing": "Beijing: Sunny, 25°C, humidity 45%",
        "tokyo": "Tokyo: Cloudy, 22°C, humidity 60%",
        "london": "London: Rainy, 15°C, humidity 80%",
        "new york": "New York: Clear, 20°C, humidity 50%",
        "paris": "Paris: Partly cloudy, 18°C, humidity 55%",
    }
    return weather_data.get(city.lower(), f"{city}: Weather data unavailable")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Only allow safe operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def search_info(query: str) -> str:
    """Search for information in a knowledge base."""
    knowledge_base = {
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "kiva": "Kiva is a multi-agent orchestration SDK for building intelligent workflows.",
        "ai": "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        "machine learning": "Machine Learning is a subset of AI that enables systems to learn from data.",
    }
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    return f"No information found for: {query}"


@tool
def translate(text: str, target_language: str) -> str:
    """Translate text to target language (mock implementation)."""
    translations = {
        ("hello", "chinese"): "你好",
        ("hello", "japanese"): "こんにちは",
        ("hello", "french"): "Bonjour",
        ("goodbye", "chinese"): "再见",
        ("goodbye", "japanese"): "さようなら",
    }
    key = (text.lower(), target_language.lower())
    return translations.get(key, f"[Translated '{text}' to {target_language}]")


@pytest.fixture
def weather_tool():
    """Return weather tool."""
    return get_weather


@pytest.fixture
def calculate_tool():
    """Return calculate tool."""
    return calculate


@pytest.fixture
def search_tool():
    """Return search tool."""
    return search_info


@pytest.fixture
def translate_tool():
    """Return translate tool."""
    return translate


@pytest.fixture
def create_weather_agent(create_model, weather_tool):
    """Factory to create a weather agent."""
    def _create():
        agent = create_agent(model=create_model(), tools=[weather_tool])
        agent.name = "weather_agent"
        agent.description = "Gets weather information for cities"
        return agent
    return _create


@pytest.fixture
def create_calculator_agent(create_model, calculate_tool):
    """Factory to create a calculator agent."""
    def _create():
        agent = create_agent(model=create_model(), tools=[calculate_tool])
        agent.name = "calculator_agent"
        agent.description = "Performs mathematical calculations"
        return agent
    return _create


@pytest.fixture
def create_search_agent(create_model, search_tool):
    """Factory to create a search agent."""
    def _create():
        agent = create_agent(model=create_model(), tools=[search_tool])
        agent.name = "search_agent"
        agent.description = "Searches for information"
        return agent
    return _create


@pytest.fixture
def create_translator_agent(create_model, translate_tool):
    """Factory to create a translator agent."""
    def _create():
        agent = create_agent(model=create_model(), tools=[translate_tool])
        agent.name = "translator_agent"
        agent.description = "Translates text between languages"
        return agent
    return _create
