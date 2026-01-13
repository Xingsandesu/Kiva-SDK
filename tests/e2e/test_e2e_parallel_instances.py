"""E2E test for Parallel Agent Instances.

Tests the parallel instance spawning feature for fan_out and map_reduce strategies.
"""

import pytest

from kiva import Kiva


class TestParallelInstancesE2E:
    """E2E tests for parallel agent instance execution."""

    def test_parallel_instances_basic(self, api_config, weather_func):
        """Test basic parallel instance execution."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information for cities")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        result = kiva.run(
            "Get weather for Beijing, Tokyo, and London",
            console=False
        )

        assert result is not None
        print(f"\nParallel Instances Result: {result}")

    def test_parallel_instances_with_multiple_agents(
        self, api_config, weather_func, search_func
    ):
        """Test parallel instances with multiple agent types."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        @kiva.agent("search", "Searches for information")
        def search(query: str) -> str:
            """Search for information."""
            return search_func(query)

        result = kiva.run(
            "Get weather for 3 cities and search for travel tips",
            console=False
        )

        assert result is not None
        print(f"\nMultiple Agents with Instances Result: {result}")

    def test_parallel_instances_calculations(self, api_config, calculate_func):
        """Test parallel instances for calculations."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("calculator", "Performs mathematical calculations")
        def calculate(expression: str) -> str:
            """Calculate expression."""
            return calculate_func(expression)

        result = kiva.run(
            "Calculate these: 10*10, 20*20, 30*30",
            console=False
        )

        assert result is not None
        print(f"\nParallel Calculations Result: {result}")

    @pytest.mark.asyncio
    async def test_parallel_instances_async(self, api_config, weather_func):
        """Test async execution with parallel instances."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        result = await kiva.run_async(
            "Compare weather in Beijing and Tokyo",
            console=False
        )

        assert result is not None
        print(f"\nAsync Parallel Instances Result: {result}")

    def test_parallel_instances_search(self, api_config, search_func):
        """Test parallel instances for search queries."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("search", "Searches for information")
        def search(query: str) -> str:
            """Search for information."""
            return search_func(query)

        result = kiva.run(
            "Search for information about Python, AI, and Machine Learning",
            console=False
        )

        assert result is not None
        print(f"\nParallel Search Result: {result}")
