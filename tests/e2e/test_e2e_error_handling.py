"""E2E test for Error Handling.

Tests error handling, edge cases, and boundary conditions.
"""

import pytest

from kiva import ConfigurationError, Kiva


class TestErrorHandlingE2E:
    """E2E tests for error handling and edge cases."""

    def test_kiva_no_agents_registered(self, api_config):
        """Test Kiva with no agents registered."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        # Running without any agents should raise an error
        with pytest.raises(Exception):
            kiva.run("Test prompt", console=False).result()

        print("\nNo agents error handled correctly")

    def test_empty_prompt_handling(self, api_config, weather_func):
        """Test handling of empty or whitespace prompt."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        # Empty prompt should still work (LLM will handle it)
        result = kiva.run("", console=False).result()
        print(f"\nEmpty prompt result: {result}")

    def test_very_long_prompt(self, api_config, weather_func):
        """Test handling of very long prompts."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        # Create a long prompt
        long_prompt = "What's the weather? " * 100

        result = kiva.run(long_prompt, console=False).result()
        assert result is not None
        print(f"\nLong prompt handled successfully")

    def test_special_characters_in_prompt(self, api_config, weather_func):
        """Test handling of special characters in prompt."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        special_prompt = "Weather in 北京? <script>alert('test')</script> 🌤️"

        result = kiva.run(special_prompt, console=False).result()
        assert result is not None
        print(f"\nSpecial characters handled successfully")

    @pytest.mark.asyncio
    async def test_unicode_agent_names(self, api_config):
        """Test handling of unicode in agent names."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("天气助手", "获取天气信息")
        def weather(city: str) -> str:
            """Get weather."""
            return f"{city}: 晴天"

        result = await kiva.run_async("北京天气如何？", console=False)
        print(f"\nUnicode agent result: {result}")

    def test_agent_with_failing_tool(self, api_config):
        """Test handling of agent execution errors."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("failing", "An agent that might fail")
        def failing_tool(input: str) -> str:
            """A tool that always fails."""
            raise RuntimeError("Tool execution failed!")

        # Should handle gracefully
        try:
            result = kiva.run("Use the failing tool", console=False).result()
            print(f"\nFailing tool result: {result}")
        except Exception as e:
            print(f"\nFailing tool error handled: {e}")

    def test_multiple_agents_partial_failure(
        self, api_config, weather_func
    ):
        """Test handling when some agents fail but others succeed."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        @kiva.agent("unreliable", "An unreliable agent")
        def unreliable_tool(input: str) -> str:
            """An unreliable tool."""
            if "fail" in input.lower():
                raise RuntimeError("Intentional failure")
            return f"Success: {input}"

        result = kiva.run(
            "Get weather in Beijing and process some data",
            console=False
        ).result()

        # Should complete with partial results
        print(f"\nPartial failure result: {result}")

    def test_class_agent_with_no_methods(self, api_config):
        """Test class agent with no public methods."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("empty", "An empty agent")
        class EmptyAgent:
            def _private_method(self) -> str:
                """Private method."""
                return "private"

        # Should handle gracefully (no tools)
        try:
            result = kiva.run("Do something", console=False).result()
            print(f"\nEmpty agent result: {result}")
        except Exception as e:
            print(f"\nEmpty agent error: {e}")

    @pytest.mark.asyncio
    async def test_async_error_handling(self, api_config, weather_func):
        """Test error handling in async context."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return weather_func(city)

        result = await kiva.run_async("Weather in Tokyo", console=False)
        assert result is not None
        print(f"\nAsync error handling result: {result}")
