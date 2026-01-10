"""Kiva SDK - Multi-Agent Orchestration Framework.

Kiva provides a flexible framework for orchestrating multiple AI agents using
LangChain and LangGraph. It supports three workflow patterns:

- Router: Routes tasks to a single agent (simple tasks)
- Supervisor: Coordinates parallel agent execution (medium complexity)
- Parliament: Iterative conflict resolution (complex reasoning)

Example:
    High-level API::

        from kiva import Kiva

        kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")

        @kiva.agent("weather", "Gets weather info")
        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        kiva.run("What's the weather in Tokyo?")

    Low-level API::

        from kiva import run, create_agent, ChatOpenAI, tool

        async for event in run(prompt="...", agents=[...]):
            print(event.type, event.data)
"""

__version__ = "0.1.0"

# Re-export LangChain essentials for convenience
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Core API
from kiva.client import Kiva
from kiva.console import run_with_console
from kiva.events import StreamEvent
from kiva.exceptions import (
    AgentError,
    ConfigurationError,
    SDKError,
    WorkflowError,
    wrap_agent_error,
)
from kiva.graph import build_orchestrator_graph, get_graph_edges, get_graph_nodes
from kiva.run import run

__all__ = [
    # Version
    "__version__",
    # High-level API
    "Kiva",
    # Core functions
    "run",
    "run_with_console",
    # Events and exceptions
    "StreamEvent",
    "SDKError",
    "ConfigurationError",
    "AgentError",
    "WorkflowError",
    "wrap_agent_error",
    # Graph utilities
    "build_orchestrator_graph",
    "get_graph_nodes",
    "get_graph_edges",
    # LangChain re-exports
    "create_agent",
    "ChatOpenAI",
    "tool",
]
