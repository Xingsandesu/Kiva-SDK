"""Kiva Client - High-level API for multi-agent orchestration.

This module provides the Kiva class, a simplified interface for creating
and running multi-agent workflows without dealing with low-level details.

Example:
    Basic usage with decorator::

        kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")

        @kiva.agent("calculator", "Performs math calculations")
        def calculate(expression: str) -> str:
            '''Evaluate a math expression.'''
            return str(eval(expression))

        result = kiva.run("What is 15 * 8?")

    Multi-tool agent with class::

        @kiva.agent("math", "Math operations")
        class MathTools:
            def add(self, a: int, b: int) -> int:
                '''Add two numbers.'''
                return a + b

            def multiply(self, a: int, b: int) -> int:
                '''Multiply two numbers.'''
                return a * b

    Modular application with routers::

        from kiva import Kiva, AgentRouter

        # In agents/weather.py
        weather_router = AgentRouter(prefix="weather")

        @weather_router.agent("forecast", "Gets forecasts")
        def get_forecast(city: str) -> str:
            return f"Sunny in {city}"

        # In main.py
        kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")
        kiva.include_router(weather_router)
        kiva.run("What's the weather?")
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from kiva.router import AgentRouter


@dataclass
class Agent:
    """Internal agent wrapper for storing agent metadata.

    Attributes:
        name: Unique identifier for the agent.
        description: Human-readable description of the agent's capabilities.
        tools: List of LangChain tools available to the agent.
    """

    name: str
    description: str
    tools: list
    _compiled: object = field(default=None, repr=False)


class Kiva:
    """High-level client for multi-agent orchestration.

    Provides a simplified API for defining agents and running orchestrated
    workflows. Supports both decorator-based and programmatic agent registration.

    Args:
        base_url: API endpoint URL for the LLM provider.
        api_key: Authentication key for the API.
        model: Model identifier (e.g., "gpt-4o", "gpt-3.5-turbo").
        temperature: Sampling temperature for model responses. Defaults to 0.7.

    Example:
        >>> kiva = Kiva(
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ... )
        >>> @kiva.agent("greeter", "Greets users")
        ... def greet(name: str) -> str:
        ...     '''Greet a person by name.'''
        ...     return f"Hello, {name}!"
        >>> kiva.run("Greet Alice")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
    ):
        """Initialize the Kiva client."""
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._agents: list[Agent] = []

    def _create_model(self) -> ChatOpenAI:
        """Create a ChatOpenAI instance with configured parameters."""
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def _to_tools(self, obj) -> list:
        """Convert a function, class, or list to LangChain tools.

        Args:
            obj: A callable function, a class with methods, or a list of functions.

        Returns:
            List of LangChain tool objects.

        Raises:
            ValueError: If obj cannot be converted to tools.
        """
        if callable(obj) and not isinstance(obj, type):
            # Single function -> single tool
            return [lc_tool(obj)]
        elif isinstance(obj, type):
            # Class -> multiple tools from methods
            instance = obj()
            tools = []
            for name in dir(instance):
                if name.startswith("_"):
                    continue
                method = getattr(instance, name)
                if callable(method) and method.__doc__:
                    tools.append(lc_tool(method))
            return tools
        elif isinstance(obj, list):
            # List of functions
            return [lc_tool(f) if not hasattr(f, "invoke") else f for f in obj]
        else:
            raise ValueError(f"Cannot convert {type(obj)} to tools")

    def agent(self, name: str, description: str) -> Callable:
        """Decorator to register an agent.

        Can decorate either a single function (becomes a single-tool agent)
        or a class with methods (each method becomes a tool).

        Args:
            name: Unique identifier for the agent.
            description: Human-readable description of the agent's purpose.

        Returns:
            Decorator function that registers the agent.

        Example:
            Single-tool agent::

                @kiva.agent("calculator", "Performs calculations")
                def calculate(expr: str) -> str:
                    '''Evaluate a math expression.'''
                    return str(eval(expr))

            Multi-tool agent::

                @kiva.agent("math", "Math operations")
                class Math:
                    def add(self, a: int, b: int) -> int:
                        '''Add two numbers.'''
                        return a + b
        """

        def decorator(obj):
            tools = self._to_tools(obj)
            self._agents.append(Agent(name=name, description=description, tools=tools))
            return obj

        return decorator

    def add_agent(self, name: str, description: str, tools: list) -> "Kiva":
        """Add an agent with an explicit tools list.

        Alternative to the decorator approach for programmatic agent registration.

        Args:
            name: Unique identifier for the agent.
            description: Human-readable description of the agent's purpose.
            tools: List of functions or LangChain tools.

        Returns:
            Self for method chaining.

        Example:
            >>> kiva.add_agent("math", "Does math", [add_func, subtract_func])
        """
        converted = self._to_tools(tools)
        self._agents.append(Agent(name=name, description=description, tools=converted))
        return self

    def include_router(self, router: "AgentRouter", prefix: str = "") -> "Kiva":
        """Include agents from an AgentRouter.

        Enables modular organization of agents across multiple files,
        similar to FastAPI's include_router pattern.

        Args:
            router: The AgentRouter containing agent definitions.
            prefix: Additional prefix to apply to all agent names.

        Returns:
            Self for method chaining.

        Example:
            >>> from agents.weather import weather_router
            >>> kiva.include_router(weather_router)
            >>> kiva.include_router(math_router, prefix="v2")
        """
        for agent_def in router.get_agents():
            name = f"{prefix}_{agent_def.name}" if prefix else agent_def.name
            tools = self._to_tools(agent_def.obj)
            self._agents.append(
                Agent(name=name, description=agent_def.description, tools=tools)
            )
        return self

    def _build_agents(self) -> list:
        """Build LangChain agents from registered agent definitions."""
        built = []
        for agent_def in self._agents:
            agent = create_agent(model=self._create_model(), tools=agent_def.tools)
            agent.name = agent_def.name
            agent.description = agent_def.description
            built.append(agent)
        return built

    async def run_async(self, prompt: str, console: bool = True) -> str | None:
        """Run orchestration asynchronously.

        Args:
            prompt: The task or question to process.
            console: Whether to display rich console output. Defaults to True.

        Returns:
            Final result string, or None if no result was produced.
        """
        agents = self._build_agents()

        if console:
            from kiva.console import run_with_console

            return await run_with_console(
                prompt=prompt,
                agents=agents,
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model,
            )
        else:
            from kiva.run import run

            result = None
            async for event in run(
                prompt=prompt,
                agents=agents,
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model,
            ):
                if event.type == "final_result":
                    result = event.data.get("result")
            return result

    def run(self, prompt: str, console: bool = True) -> str | None:
        """Run orchestration synchronously.

        Convenience wrapper around run_async for synchronous contexts.

        Args:
            prompt: The task or question to process.
            console: Whether to display rich console output. Defaults to True.

        Returns:
            Final result string, or None if no result was produced.
        """
        import asyncio

        return asyncio.run(self.run_async(prompt, console=console))
