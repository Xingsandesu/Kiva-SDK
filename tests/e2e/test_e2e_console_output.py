"""E2E test for Console Output (Rich visualization).

Tests the run_with_console() function for rich terminal output.
"""

import pytest

from kiva import run_with_console


class TestConsoleOutputE2E:
    """E2E tests for rich console visualization."""

    @pytest.mark.asyncio
    async def test_console_basic_execution(
        self, api_config, create_weather_agent
    ):
        """Test basic console output execution."""
        agents = [create_weather_agent()]

        result = await run_with_console(
            prompt="What's the weather in Beijing?",
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
        )

        assert result is not None
        print(f"\nConsole Result: {result}")

    @pytest.mark.asyncio
    async def test_console_multi_agent(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Test console output with multiple agents."""
        agents = [create_weather_agent(), create_calculator_agent()]

        result = await run_with_console(
            prompt="Weather in Tokyo and calculate 25 * 4",
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
        )

        assert result is not None
        print(f"\nMulti-agent Console Result: {result}")

    @pytest.mark.asyncio
    async def test_console_refresh_rate(
        self, api_config, create_weather_agent
    ):
        """Test console with custom refresh rate."""
        agents = [create_weather_agent()]

        result = await run_with_console(
            prompt="Weather in London",
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
            refresh_per_second=4,  # Lower refresh rate
        )

        assert result is not None
        print(f"\nCustom Refresh Rate Result: {result}")

    @pytest.mark.asyncio
    async def test_console_parallel_instances(
        self, api_config, create_weather_agent
    ):
        """Test console display of parallel instances."""
        agents = [create_weather_agent()]

        result = await run_with_console(
            prompt="Get weather for Beijing, Tokyo, and London",
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
        )

        assert result is not None
        print(f"\nParallel Instances Console Result: {result}")

    @pytest.mark.asyncio
    async def test_console_three_agents(
        self, api_config, create_weather_agent, create_calculator_agent, create_search_agent
    ):
        """Test console with three different agents."""
        agents = [
            create_weather_agent(),
            create_calculator_agent(),
            create_search_agent(),
        ]

        result = await run_with_console(
            prompt="Weather in Paris, calculate 100/5, search for AI",
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
        )

        assert result is not None
        print(f"\nThree Agents Console Result: {result}")

    @pytest.mark.asyncio
    async def test_console_long_result(
        self, api_config, create_search_agent
    ):
        """Test console handling of long results."""
        agents = [create_search_agent()]

        result = await run_with_console(
            prompt="Search for detailed information about Python programming language",
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
        )

        assert result is not None
        print(f"\nLong Result Console: {result[:200]}...")

    @pytest.mark.asyncio
    async def test_console_special_characters(
        self, api_config, create_weather_agent
    ):
        """Test console handling of special characters."""
        agents = [create_weather_agent()]

        result = await run_with_console(
            prompt="Weather in 北京 🌤️",
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
        )

        # Should handle unicode without crashing
        print(f"\nSpecial Characters Console Result: {result}")
