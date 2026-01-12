"""Tests for AgentRouter modular agent organization."""

import pytest

from kiva.router import AgentDefinition, AgentRouter


class TestAgentRouter:
    """Tests for AgentRouter class."""

    def test_create_router_with_defaults(self):
        """Test creating a router with default values."""
        router = AgentRouter()
        assert router.prefix == ""
        assert router.tags == []
        assert router.get_agents() == []

    def test_create_router_with_prefix(self):
        """Test creating a router with a prefix."""
        router = AgentRouter(prefix="weather")
        assert router.prefix == "weather"

    def test_create_router_with_tags(self):
        """Test creating a router with tags."""
        router = AgentRouter(tags=["api", "v1"])
        assert router.tags == ["api", "v1"]

    def test_agent_decorator_function(self):
        """Test registering a function as an agent."""
        router = AgentRouter()

        @router.agent("search", "Searches for information")
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for {query}"

        agents = router.get_agents()
        assert len(agents) == 1
        assert agents[0].name == "search"
        assert agents[0].description == "Searches for information"
        assert agents[0].obj is search

    def test_agent_decorator_class(self):
        """Test registering a class as an agent."""
        router = AgentRouter()

        @router.agent("math", "Math operations")
        class MathTools:
            def add(self, a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

        agents = router.get_agents()
        assert len(agents) == 1
        assert agents[0].name == "math"
        assert agents[0].obj is MathTools

    def test_agent_with_prefix(self):
        """Test that prefix is applied to agent names."""
        router = AgentRouter(prefix="api")

        @router.agent("search", "Search")
        def search(q: str) -> str:
            """Search."""
            return q

        agents = router.get_agents()
        assert agents[0].name == "api_search"

    def test_multiple_agents(self):
        """Test registering multiple agents."""
        router = AgentRouter()

        @router.agent("agent1", "First agent")
        def func1() -> str:
            """First."""
            return "1"

        @router.agent("agent2", "Second agent")
        def func2() -> str:
            """Second."""
            return "2"

        agents = router.get_agents()
        assert len(agents) == 2
        names = [a.name for a in agents]
        assert "agent1" in names
        assert "agent2" in names


class TestAgentRouterNesting:
    """Tests for nested router functionality."""

    def test_include_router_basic(self):
        """Test including a sub-router."""
        main = AgentRouter()
        sub = AgentRouter()

        @sub.agent("search", "Search")
        def search(q: str) -> str:
            """Search."""
            return q

        main.include_router(sub)
        agents = main.get_agents()
        assert len(agents) == 1
        assert agents[0].name == "search"

    def test_include_router_with_prefix(self):
        """Test including a sub-router with additional prefix."""
        main = AgentRouter()
        sub = AgentRouter()

        @sub.agent("search", "Search")
        def search(q: str) -> str:
            """Search."""
            return q

        main.include_router(sub, prefix="v1")
        agents = main.get_agents()
        assert agents[0].name == "v1_search"

    def test_include_router_preserves_sub_prefix(self):
        """Test that sub-router prefix is preserved."""
        main = AgentRouter()
        sub = AgentRouter(prefix="api")

        @sub.agent("search", "Search")
        def search(q: str) -> str:
            """Search."""
            return q

        main.include_router(sub)
        agents = main.get_agents()
        assert agents[0].name == "api_search"

    def test_include_router_combines_prefixes(self):
        """Test that prefixes are combined correctly."""
        main = AgentRouter()
        sub = AgentRouter(prefix="api")

        @sub.agent("search", "Search")
        def search(q: str) -> str:
            """Search."""
            return q

        main.include_router(sub, prefix="v1")
        agents = main.get_agents()
        assert agents[0].name == "v1_api_search"

    def test_nested_routers_multiple_levels(self):
        """Test deeply nested routers."""
        root = AgentRouter()
        level1 = AgentRouter(prefix="l1")
        level2 = AgentRouter(prefix="l2")

        @level2.agent("func", "Function")
        def func() -> str:
            """Func."""
            return "x"

        level1.include_router(level2)
        root.include_router(level1)

        agents = root.get_agents()
        assert len(agents) == 1
        assert agents[0].name == "l1_l2_func"

    def test_include_multiple_routers(self):
        """Test including multiple sub-routers."""
        main = AgentRouter()
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

        main.include_router(weather)
        main.include_router(math)

        agents = main.get_agents()
        assert len(agents) == 2
        names = [a.name for a in agents]
        assert "weather_forecast" in names
        assert "math_add" in names


class TestAgentDefinition:
    """Tests for AgentDefinition dataclass."""

    def test_create_agent_definition(self):
        """Test creating an AgentDefinition."""

        def func() -> str:
            return "x"

        defn = AgentDefinition(name="test", description="Test agent", obj=func)
        assert defn.name == "test"
        assert defn.description == "Test agent"
        assert defn.obj is func


class TestKivaIncludeRouter:
    """Tests for Kiva.include_router integration."""

    def test_kiva_include_router_basic(self):
        """Test including a router in Kiva."""
        from kiva import AgentRouter, Kiva

        kiva = Kiva(base_url="http://test", api_key="test", model="test")
        router = AgentRouter()

        @router.agent("test", "Test agent")
        def test_func() -> str:
            """Test function."""
            return "test"

        kiva.include_router(router)
        assert len(kiva._agents) == 1
        assert kiva._agents[0].name == "test"

    def test_kiva_include_router_with_prefix(self):
        """Test including a router with additional prefix."""
        from kiva import AgentRouter, Kiva

        kiva = Kiva(base_url="http://test", api_key="test", model="test")
        router = AgentRouter(prefix="api")

        @router.agent("search", "Search")
        def search(q: str) -> str:
            """Search."""
            return q

        kiva.include_router(router, prefix="v1")
        assert kiva._agents[0].name == "v1_api_search"

    def test_kiva_include_multiple_routers(self):
        """Test including multiple routers."""
        from kiva import AgentRouter, Kiva

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

    def test_kiva_include_router_method_chaining(self):
        """Test that include_router returns self for chaining."""
        from kiva import AgentRouter, Kiva

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

    def test_kiva_mixed_registration(self):
        """Test mixing direct registration with router inclusion."""
        from kiva import AgentRouter, Kiva

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
