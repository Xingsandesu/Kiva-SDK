"""E2E test for Error Handling.

Tests error handling, edge cases, and boundary conditions.
"""

import pytest

from kiva import ConfigurationError, run, Kiva


class TestErrorHandlingE2E:
    """E2E tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_agents_raises_error(self, api_config):
        """Test that empty agents list raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            async for _ in run(
                prompt="Test prompt",
                agents=[],
                model_name=api_config["model"],
                api_key=api_config["api_key"],
                base_url=api_config["base_url"],
            ):
                pass
        
        assert "agents" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
        print(f"\nEmpty agents error: {exc_info.value}")

    @pytest.mark.asyncio
    async def test_invalid_agent_raises_error(self, api_config):
        """Test that agent without ainvoke raises ConfigurationError."""
        class InvalidAgent:
            def __init__(self):
                self.name = "invalid"
            
            def invoke(self, data):
                return data
        
        with pytest.raises(ConfigurationError) as exc_info:
            async for _ in run(
                prompt="Test prompt",
                agents=[InvalidAgent()],
                model_name=api_config["model"],
                api_key=api_config["api_key"],
                base_url=api_config["base_url"],
            ):
                pass
        
        assert "ainvoke" in str(exc_info.value) or "create_agent" in str(exc_info.value)
        print(f"\nInvalid agent error: {exc_info.value}")

    @pytest.mark.asyncio
    async def test_agent_execution_error_handling(
        self, api_config, create_model
    ):
        """Test handling of agent execution errors."""
        from langchain.agents import create_agent
        from langchain_core.tools import tool

        @tool
        def failing_tool(input: str) -> str:
            """A tool that always fails."""
            raise RuntimeError("Tool execution failed!")

        agent = create_agent(model=create_model(), tools=[failing_tool])
        agent.name = "failing_agent"
        agent.description = "An agent that fails"

        events = []
        async for event in run(
            prompt="Use the failing tool",
            agents=[agent],
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Should still complete (graceful error handling)
        event_types = [e.type for e in events]
        assert "final_result" in event_types or "error" in event_types
        
        print(f"\nEvent types with failing agent: {event_types}")

    @pytest.mark.asyncio
    async def test_empty_prompt_handling(
        self, api_config, create_weather_agent
    ):
        """Test handling of empty or whitespace prompt."""
        agents = [create_weather_agent()]
        events = []

        # Empty prompt should still work (LLM will handle it)
        async for event in run(
            prompt="",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Should complete without crashing
        event_types = [e.type for e in events]
        print(f"\nEmpty prompt event types: {event_types}")

    @pytest.mark.asyncio
    async def test_very_long_prompt(
        self, api_config, create_weather_agent
    ):
        """Test handling of very long prompts."""
        agents = [create_weather_agent()]
        events = []

        # Create a long prompt
        long_prompt = "What's the weather? " * 100

        async for event in run(
            prompt=long_prompt,
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Should complete
        assert any(e.type == "final_result" for e in events)
        print(f"\nLong prompt handled successfully")

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(
        self, api_config, create_weather_agent
    ):
        """Test handling of special characters in prompt."""
        agents = [create_weather_agent()]
        events = []

        special_prompt = "Weather in 北京? <script>alert('test')</script> 🌤️"

        async for event in run(
            prompt=special_prompt,
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Should complete without issues
        assert any(e.type == "final_result" for e in events)
        print(f"\nSpecial characters handled successfully")

    def test_kiva_no_agents_registered(self, api_config):
        """Test Kiva with no agents registered."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        # Running without any agents should raise an error
        with pytest.raises(Exception):
            kiva.run("Test prompt", console=False)
        
        print("\nNo agents error handled correctly")

    @pytest.mark.asyncio
    async def test_invalid_workflow_override(
        self, api_config, create_weather_agent
    ):
        """Test handling of invalid workflow_override value."""
        agents = [create_weather_agent()]
        events = []

        # Invalid workflow should fall back to default
        async for event in run(
            prompt="Weather in Beijing",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="invalid_workflow",  # Invalid value
        ):
            events.append(event)

        # Should still complete (falls back to router)
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert workflow_event.data["workflow"] in ["router", "supervisor", "parliament"]
        print(f"\nInvalid workflow override handled: {workflow_event.data['workflow']}")

    @pytest.mark.asyncio
    async def test_zero_max_iterations(
        self, api_config, create_weather_agent
    ):
        """Test parliament with zero max_iterations."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="Weather in Tokyo",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=0,  # Edge case
        ):
            events.append(event)

        # Should complete quickly
        assert any(e.type == "final_result" for e in events)
        print(f"\nZero max_iterations handled")

    @pytest.mark.asyncio
    async def test_negative_max_parallel_agents(
        self, api_config, create_weather_agent
    ):
        """Test handling of negative max_parallel_agents."""
        agents = [create_weather_agent()]
        events = []

        # Negative value should be handled gracefully
        async for event in run(
            prompt="Weather in London",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            max_parallel_agents=-1,  # Edge case
        ):
            events.append(event)

        # Should still complete
        print(f"\nNegative max_parallel_agents handled")

    @pytest.mark.asyncio
    async def test_partial_agent_failure(
        self, api_config, create_weather_agent, create_model
    ):
        """Test handling when some agents fail but others succeed."""
        from langchain.agents import create_agent
        from langchain_core.tools import tool

        @tool
        def unreliable_tool(input: str) -> str:
            """An unreliable tool."""
            if "fail" in input.lower():
                raise RuntimeError("Intentional failure")
            return f"Success: {input}"

        unreliable_agent = create_agent(model=create_model(), tools=[unreliable_tool])
        unreliable_agent.name = "unreliable_agent"
        unreliable_agent.description = "An unreliable agent"

        agents = [create_weather_agent(), unreliable_agent]
        events = []

        async for event in run(
            prompt="Get weather in Beijing and process some data",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            events.append(event)

        # Should complete with partial results
        final_event = next((e for e in events if e.type == "final_result"), None)
        if final_event:
            print(f"\nPartial failure result: {final_event.data['result']}")
            print(f"Partial: {final_event.data.get('partial_result', False)}")

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
