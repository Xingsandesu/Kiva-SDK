"""Tests for the Kiva high-level API.

These tests validate the correctness properties for the Kiva client.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiva import AgentError, ConfigurationError, Kiva, SDKError


class TestKivaInitialization:
    """Tests for Kiva client initialization."""

    def test_create_kiva_with_required_params(self):
        """Test creating Kiva with required parameters."""
        kiva = Kiva(
            base_url="http://test",
            api_key="test-key",
            model="test-model",
        )
        assert kiva.base_url == "http://test"
        assert kiva.api_key == "test-key"
        assert kiva.model == "test-model"
        assert kiva.temperature == 0.7  # default

    def test_create_kiva_with_custom_temperature(self):
        """Test creating Kiva with custom temperature."""
        kiva = Kiva(
            base_url="http://test",
            api_key="test-key",
            model="test-model",
            temperature=0.5,
        )
        assert kiva.temperature == 0.5

    @given(
        temperature=st.floats(min_value=0.0, max_value=2.0),
    )
    @settings(max_examples=50)
    def test_temperature_range(self, temperature: float):
        """Test that various temperature values are accepted."""
        kiva = Kiva(
            base_url="http://test",
            api_key="test-key",
            model="test-model",
            temperature=temperature,
        )
        assert kiva.temperature == temperature


class TestKivaAgentRegistration:
    """Tests for agent registration via decorator and add_agent."""

    def test_register_function_agent(self):
        """Test registering a function as an agent."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        @kiva.agent("weather", "Gets weather info")
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"{city}: Sunny"

        assert len(kiva._agents) == 1
        assert kiva._agents[0].name == "weather"
        assert kiva._agents[0].description == "Gets weather info"

    def test_register_class_agent(self):
        """Test registering a class as a multi-tool agent."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        @kiva.agent("math", "Math operations")
        class MathTools:
            def add(self, a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

            def multiply(self, a: int, b: int) -> int:
                """Multiply two numbers."""
                return a * b

        assert len(kiva._agents) == 1
        assert kiva._agents[0].name == "math"
        # Class should have multiple tools
        assert len(kiva._agents[0].tools) >= 2

    def test_add_agent_method(self):
        """Test add_agent method for programmatic registration."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        def search(query: str) -> str:
            """Search for info."""
            return f"Results for: {query}"

        kiva.add_agent("search", "Searches for information", [search])

        assert len(kiva._agents) == 1
        assert kiva._agents[0].name == "search"

    def test_add_agent_method_chaining(self):
        """Test that add_agent returns self for chaining."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        def func1() -> str:
            """Func 1."""
            return "1"

        def func2() -> str:
            """Func 2."""
            return "2"

        result = kiva.add_agent("a1", "Agent 1", [func1]).add_agent("a2", "Agent 2", [func2])

        assert result is kiva
        assert len(kiva._agents) == 2

    def test_multiple_agents_registration(self):
        """Test registering multiple agents."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        @kiva.agent("weather", "Weather info")
        def weather(city: str) -> str:
            """Get weather."""
            return city

        @kiva.agent("calc", "Calculator")
        def calc(expr: str) -> str:
            """Calculate."""
            return expr

        @kiva.agent("search", "Search")
        def search(query: str) -> str:
            """Search."""
            return query

        assert len(kiva._agents) == 3
        names = [a.name for a in kiva._agents]
        assert "weather" in names
        assert "calc" in names
        assert "search" in names

    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        description=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=50)
    def test_agent_name_and_description_preserved(self, name: str, description: str):
        """Test that agent name and description are preserved."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        @kiva.agent(name, description)
        def test_func() -> str:
            """Test function."""
            return "test"

        assert kiva._agents[0].name == name
        assert kiva._agents[0].description == description


class TestKivaIncludeRouter:
    """Tests for including AgentRouter."""

    def test_include_router_basic(self):
        """Test including a router."""
        from kiva import AgentRouter

        kiva = Kiva(base_url="http://test", api_key="test", model="test")
        router = AgentRouter()

        @router.agent("test", "Test agent")
        def test_func() -> str:
            """Test function."""
            return "test"

        kiva.include_router(router)
        assert len(kiva._agents) == 1
        assert kiva._agents[0].name == "test"

    def test_include_router_with_prefix(self):
        """Test including a router with additional prefix."""
        from kiva import AgentRouter

        kiva = Kiva(base_url="http://test", api_key="test", model="test")
        router = AgentRouter(prefix="api")

        @router.agent("search", "Search")
        def search(q: str) -> str:
            """Search."""
            return q

        kiva.include_router(router, prefix="v1")
        assert kiva._agents[0].name == "v1_api_search"

    def test_include_multiple_routers(self):
        """Test including multiple routers."""
        from kiva import AgentRouter

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        weather = AgentRouter(prefix="weather")
        math = AgentRouter(prefix="math")

        @weather.agent("forecast", "Forecast")
        def forecast(city: str) -> str:
            """Get forecast."""
            return city

        @math.agent("add", "Add")
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        kiva.include_router(weather)
        kiva.include_router(math)

        assert len(kiva._agents) == 2
        names = [a.name for a in kiva._agents]
        assert "weather_forecast" in names
        assert "math_add" in names

    def test_include_router_method_chaining(self):
        """Test that include_router returns self for chaining."""
        from kiva import AgentRouter

        kiva = Kiva(base_url="http://test", api_key="test", model="test")
        router1 = AgentRouter()
        router2 = AgentRouter()

        @router1.agent("a1", "Agent 1")
        def a1() -> str:
            """Agent 1."""
            return "1"

        @router2.agent("a2", "Agent 2")
        def a2() -> str:
            """Agent 2."""
            return "2"

        result = kiva.include_router(router1).include_router(router2)
        assert result is kiva
        assert len(kiva._agents) == 2

    def test_mixed_registration(self):
        """Test mixing direct registration with router inclusion."""
        from kiva import AgentRouter

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        @kiva.agent("direct", "Direct agent")
        def direct_func() -> str:
            """Direct function."""
            return "direct"

        router = AgentRouter(prefix="router")

        @router.agent("routed", "Routed agent")
        def routed_func() -> str:
            """Routed function."""
            return "routed"

        kiva.include_router(router)

        assert len(kiva._agents) == 2
        names = [a.name for a in kiva._agents]
        assert "direct" in names
        assert "router_routed" in names


class TestKivaExceptions:
    """Tests for exception handling."""

    def test_sdk_error_is_base_exception(self):
        """Test that SDKError is the base exception."""
        assert issubclass(ConfigurationError, SDKError)
        assert issubclass(AgentError, SDKError)

    def test_configuration_error_message(self):
        """Test ConfigurationError message."""
        error = ConfigurationError("Test error message")
        assert "Test error message" in str(error)

    def test_agent_error_with_agent_id(self):
        """Test AgentError with agent_id."""
        error = AgentError("Test error", agent_id="test_agent")
        assert error.agent_id == "test_agent"
        assert "test_agent" in str(error)


class TestKivaToolConversion:
    """Tests for tool conversion logic."""

    def test_convert_function_to_tool(self):
        """Test converting a function to a tool."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        def test_func(x: str) -> str:
            """Test function."""
            return x

        tools = kiva._to_tools(test_func)
        assert len(tools) == 1

    def test_convert_class_to_tools(self):
        """Test converting a class to multiple tools."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        class TestClass:
            def method1(self, x: str) -> str:
                """Method 1."""
                return x

            def method2(self, y: int) -> int:
                """Method 2."""
                return y

        tools = kiva._to_tools(TestClass)
        assert len(tools) >= 2

    def test_convert_list_to_tools(self):
        """Test converting a list of functions to tools."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        def func1(x: str) -> str:
            """Func 1."""
            return x

        def func2(y: int) -> int:
            """Func 2."""
            return y

        tools = kiva._to_tools([func1, func2])
        assert len(tools) == 2

    def test_invalid_type_raises_error(self):
        """Test that invalid types raise ValueError."""
        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        with pytest.raises(ValueError):
            kiva._to_tools("invalid")

        with pytest.raises(ValueError):
            kiva._to_tools(123)
