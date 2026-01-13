"""E2E test for Router workflow - simple single-agent tasks.

Tests the router workflow which routes tasks to a single agent.
"""

import pytest

from kiva import Kiva


class TestRouterWorkflowE2E:
    """E2E tests for router workflow with single agent execution."""

    def test_router_simple_weather_query(self, api_config, weather_func):
        """Test simple weather query routed to single agent."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information for cities")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        result = kiva.run("What's the weather in Beijing?", console=False)

        assert result is not None
        assert len(result) > 0
        print(f"\nSimple Weather Result: {result}")

    def test_router_simple_calculation(self, api_config, calculate_func):
        """Test simple calculation routed to calculator agent."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("calculator", "Performs mathematical calculations")
        def calculate(expression: str) -> str:
            """Calculate expression."""
            return calculate_func(expression)

        result = kiva.run("Calculate 15 * 8", console=False)

        assert result is not None
        print(f"\nCalculation Result: {result}")

    def test_router_search_query(self, api_config, search_func):
        """Test search query routed to search agent."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("search", "Searches for information")
        def search(query: str) -> str:
            """Search for information."""
            return search_func(query)

        result = kiva.run("Search for information about Python", console=False)

        assert result is not None
        print(f"\nSearch Result: {result}")

    @pytest.mark.asyncio
    async def test_router_async_execution(self, api_config, weather_func):
        """Test async execution with router workflow."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        result = await kiva.run_async("What's the weather in Tokyo?", console=False)

        assert result is not None
        print(f"\nAsync Router Result: {result}")
