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

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from kiva.router import AgentRouter
    from kiva.verification import VerificationResult


@dataclass
class RegisteredVerifier:
    """Internal verifier wrapper for storing verifier metadata.

    Attributes:
        name: Unique identifier for the verifier.
        priority: Execution priority (higher = earlier execution).
        func: The verifier function.
    """

    name: str
    priority: int
    func: Callable[..., "VerificationResult"]

    def verify(
        self,
        task: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> "VerificationResult":
        """Execute the verifier function.

        Args:
            task: The task that was assigned to the worker.
            output: The output produced by the worker.
            context: Optional additional context for verification.

        Returns:
            VerificationResult with status and details.
        """
        return self.func(task, output, context)


@dataclass
class Agent:
    """Internal agent wrapper for storing agent metadata.

    Attributes:
        name: Unique identifier for the agent.
        description: Human-readable description of the agent's capabilities.
        tools: List of LangChain tools available to the agent.
        max_iterations: Per-agent maximum iterations for verification retry.
            If None, uses the global default from Kiva instance.
    """

    name: str
    description: str
    tools: list
    max_iterations: int | None = None
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
        max_iterations: Global maximum iterations for verification retry.
            Defaults to 3. Can be overridden per-agent or per-run.

    Example:
        >>> kiva = Kiva(
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ...     max_iterations=5,  # Global default
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
        max_iterations: int = 3,
        worker_recursion_limit: int = 25,
    ):
        """Initialize the Kiva client."""
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.worker_recursion_limit = worker_recursion_limit
        self._agents: list[Agent] = []
        self._verifiers: list[RegisteredVerifier] = []

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

    def agent(
        self,
        name: str,
        description: str,
        max_iterations: int | None = None,
    ) -> Callable:
        """Decorator to register an agent.

        Can decorate either a single function (becomes a single-tool agent)
        or a class with methods (each method becomes a tool).

        Args:
            name: Unique identifier for the agent.
            description: Human-readable description of the agent's purpose.
            max_iterations: Per-agent maximum iterations for verification retry.
                If None, uses the global default from Kiva instance.

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

            Agent with custom max_iterations::

                @kiva.agent("complex", "Complex task", max_iterations=5)
                def complex_task(query: str) -> str:
                    '''Handle complex queries.'''
                    return query
        """

        def decorator(obj):
            tools = self._to_tools(obj)
            self._agents.append(
                Agent(
                    name=name,
                    description=description,
                    tools=tools,
                    max_iterations=max_iterations,
                )
            )
            return obj

        return decorator

    def verifier(
        self,
        name: str | None = None,
        priority: int = 0,
    ) -> Callable[
        [Callable[..., "VerificationResult"]], Callable[..., "VerificationResult"]
    ]:
        """Decorator to register a custom verifier.

        Custom verifiers are executed during output verification to provide
        additional validation logic beyond the default LLM-based verification.

        Args:
            name: Unique identifier for the verifier. If not provided,
                uses the function name.
            priority: Execution priority. Higher values execute earlier.
                Defaults to 0.

        Returns:
            Decorator function that registers the verifier.

        Example:
            Basic verifier::

                @kiva.verifier("length_check")
                def check_length(
                    task: str, output: str, context: dict | None = None
                ) -> VerificationResult:
                    '''Check that output meets minimum length.'''
                    if len(output) < 10:
                        return VerificationResult(
                            status=VerificationStatus.FAILED,
                            rejection_reason="Output too short",
                            improvement_suggestions=["Provide more detail"],
                        )
                    return VerificationResult(status=VerificationStatus.PASSED)

            Verifier with priority::

                @kiva.verifier("critical_check", priority=10)
                def critical_check(
                    task: str, output: str, context: dict | None = None
                ) -> VerificationResult:
                    '''Critical check that runs first.'''
                    # This runs before lower priority verifiers
                    return VerificationResult(status=VerificationStatus.PASSED)
        """

        def decorator(
            func: Callable[..., "VerificationResult"],
        ) -> Callable[..., "VerificationResult"]:
            verifier_name = name if name is not None else func.__name__
            # Store metadata on the function for introspection
            func._verifier_name = verifier_name  # type: ignore[attr-defined]
            func._verifier_priority = priority  # type: ignore[attr-defined]

            registered = RegisteredVerifier(
                name=verifier_name,
                priority=priority,
                func=func,
            )
            self._verifiers.append(registered)
            return func

        return decorator

    def get_verifiers(self) -> list[RegisteredVerifier]:
        """Get all registered verifiers sorted by priority (highest first).

        Returns:
            List of RegisteredVerifier instances sorted by priority in
            descending order (higher priority verifiers first).

        Example:
            >>> verifiers = kiva.get_verifiers()
            >>> for v in verifiers:
            ...     print(f"{v.name}: priority {v.priority}")
        """
        return sorted(self._verifiers, key=lambda v: v.priority, reverse=True)

    def add_agent(
        self,
        name: str,
        description: str,
        tools: list,
        max_iterations: int | None = None,
    ) -> "Kiva":
        """Add an agent with an explicit tools list.

        Alternative to the decorator approach for programmatic agent registration.

        Args:
            name: Unique identifier for the agent.
            description: Human-readable description of the agent's purpose.
            tools: List of functions or LangChain tools.
            max_iterations: Per-agent maximum iterations for verification retry.
                If None, uses the global default from Kiva instance.

        Returns:
            Self for method chaining.

        Example:
            >>> kiva.add_agent("math", "Does math", [add_func, subtract_func])
            >>> kiva.add_agent("complex", "Complex task", [task_func], max_iterations=5)
        """
        converted = self._to_tools(tools)
        self._agents.append(
            Agent(
                name=name,
                description=description,
                tools=converted,
                max_iterations=max_iterations,
            )
        )
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
                Agent(
                    name=name,
                    description=agent_def.description,
                    tools=tools,
                    max_iterations=getattr(agent_def, "max_iterations", None),
                )
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

    async def run_async(
        self,
        prompt: str,
        console: bool = True,
        max_iterations: int | None = None,
        worker_recursion_limit: int | None = None,
    ) -> str | None:
        """Run orchestration asynchronously.

        Args:
            prompt: The task or question to process.
            console: Whether to display rich console output. Defaults to True.
            max_iterations: Maximum iterations for verification retry.
                If None, uses the global default from Kiva instance.
            worker_recursion_limit: Maximum internal steps per Worker Agent execution.
                If None, uses the global default from Kiva instance.

        Returns:
            Final result string, or None if no result was produced.
        """
        agents = self._build_agents()
        effective_max_iterations = (
            max_iterations if max_iterations is not None else self.max_iterations
        )
        effective_worker_recursion_limit = (
            worker_recursion_limit
            if worker_recursion_limit is not None
            else self.worker_recursion_limit
        )

        if console:
            from kiva.console import run_with_console

            return await run_with_console(
                prompt=prompt,
                agents=agents,
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model,
                max_iterations=effective_max_iterations,
                worker_recursion_limit=effective_worker_recursion_limit,
                custom_verifiers=self._verifiers,
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
                max_iterations=effective_max_iterations,
                worker_recursion_limit=effective_worker_recursion_limit,
                custom_verifiers=self._verifiers,
            ):
                if event.type == "final_result":
                    result = event.data.get("result")
            return result

    def run(
        self,
        prompt: str,
        console: bool = True,
        max_iterations: int | None = None,
        worker_recursion_limit: int | None = None,
    ) -> str | None | Iterator:
        """Run orchestration synchronously.

        Convenience wrapper around run_async for synchronous contexts.

        Args:
            prompt: The task or question to process.
            console: Whether to display rich console output. Defaults to True.
            max_iterations: Maximum iterations for verification retry.
                If None, uses the global default from Kiva instance.
            worker_recursion_limit: Maximum internal steps per Worker Agent execution.
                If None, uses the global default from Kiva instance.

        Returns:
            When console=True: Final result string, or None if no result was produced.
            When console=False: Iterable of StreamEvent objects.
        """
        import asyncio

        if console:
            return asyncio.run(
                self.run_async(prompt, console=console, max_iterations=max_iterations)
            )

        agents = self._build_agents()
        effective_max_iterations = (
            max_iterations if max_iterations is not None else self.max_iterations
        )

        return _KivaRunStream(
            prompt=prompt,
            agents=agents,
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model,
            max_iterations=effective_max_iterations,
            worker_recursion_limit=worker_recursion_limit
            if worker_recursion_limit is not None
            else self.worker_recursion_limit,
            custom_verifiers=self._verifiers,
        )


class _KivaRunStream:
    def __init__(
        self,
        *,
        prompt: str,
        agents: list,
        base_url: str,
        api_key: str,
        model_name: str,
        max_iterations: int,
        worker_recursion_limit: int,
        custom_verifiers: list,
    ):
        self._prompt = prompt
        self._agents = agents
        self._base_url = base_url
        self._api_key = api_key
        self._model_name = model_name
        self._max_iterations = max_iterations
        self._custom_verifiers = custom_verifiers
        self._worker_recursion_limit = worker_recursion_limit

        self._events: list = []
        self._done = False
        self._final_result: str | None = None
        self._loop = None
        self._agen = None

    def __iter__(self) -> Iterator:
        import asyncio

        idx = 0
        while True:
            while idx < len(self._events):
                event = self._events[idx]
                idx += 1
                yield event

            if self._done:
                return

            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                from kiva.run import run

                self._agen = run(
                    prompt=self._prompt,
                    agents=self._agents,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    model_name=self._model_name,
                    max_iterations=self._max_iterations,
                    worker_recursion_limit=self._worker_recursion_limit,
                    custom_verifiers=self._custom_verifiers,
                )

            try:
                event = self._loop.run_until_complete(self._agen.__anext__())
            except StopAsyncIteration:
                self._done = True
                self._shutdown()
                return

            self._events.append(event)
            if getattr(event, "type", None) == "final_result":
                data = getattr(event, "data", {}) or {}
                self._final_result = data.get("result")
            yield event

    def result(self) -> str | None:
        for _ in self:
            pass
        return self._final_result

    def _shutdown(self) -> None:
        if self._agen is not None:
            try:
                self._loop.run_until_complete(self._agen.aclose())
            except Exception:
                pass
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
        self._loop = None
        self._agen = None

    def __str__(self) -> str:
        if self._done:
            return self._final_result or ""
        if self._final_result is not None:
            return self._final_result
        return ""
