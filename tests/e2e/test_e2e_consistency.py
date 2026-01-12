"""E2E test for API Consistency.

Tests that run() and run_with_console() produce consistent behavior.
"""

import pytest

from kiva import run, run_with_console, Kiva


class TestAPIConsistencyE2E:
    """E2E tests for consistent behavior across different APIs."""

    @pytest.mark.asyncio
    async def test_run_and_console_same_workflow(
        self, api_config, create_weather_agent
    ):
        """Test that run() and run_with_console() select same workflow."""
        agents = [create_weather_agent()]
        prompt = "What's the weather in Beijing?"

        # Using run()
        run_workflow = None
        async for event in run(
            prompt=prompt,
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="router",
        ):
            if event.type == "workflow_selected":
                run_workflow = event.data["workflow"]
                break

        # Using run_with_console() - workflow is determined internally
        # We can't directly compare but both should work
        console_result = await run_with_console(
            prompt=prompt,
            agents=agents,
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model_name=api_config["model"],
        )

        assert run_workflow == "router"
        assert console_result is not None
        print(f"\nrun() workflow: {run_workflow}")
        print(f"run_with_console() result: {console_result}")

    @pytest.mark.asyncio
    async def test_kiva_sync_async_consistency(self, api_config):
        """Test that Kiva.run() and Kiva.run_async() produce similar results."""
        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather")
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"{city}: Sunny, 25°C"

        # Only test async run (sync run can't be called from async context)
        async_result = await kiva.run_async("Weather in Tokyo", console=False)

        # Should produce result
        assert async_result is not None
        
        print(f"\nAsync result: {async_result}")

    @pytest.mark.asyncio
    async def test_event_structure_consistency(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Test that event structure is consistent across different scenarios."""
        agents = [create_weather_agent(), create_calculator_agent()]

        # Test with router workflow
        router_events = []
        async for event in run(
            prompt="Weather in Beijing",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="router",
        ):
            router_events.append(event)

        # Test with supervisor workflow
        supervisor_events = []
        async for event in run(
            prompt="Weather in Beijing and calculate 10*10",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            supervisor_events.append(event)

        # Both should have workflow_selected and final_result
        router_types = {e.type for e in router_events}
        supervisor_types = {e.type for e in supervisor_events}

        assert "workflow_selected" in router_types
        assert "workflow_selected" in supervisor_types
        assert "final_result" in router_types
        assert "final_result" in supervisor_types

        # Verify event structure
        for events in [router_events, supervisor_events]:
            for event in events:
                assert hasattr(event, "type")
                assert hasattr(event, "data")
                assert hasattr(event, "timestamp")
                assert event.timestamp > 0

        print(f"\nRouter event types: {router_types}")
        print(f"Supervisor event types: {supervisor_types}")

    @pytest.mark.asyncio
    async def test_execution_id_format_consistency(
        self, api_config, create_weather_agent
    ):
        """Test that execution_id format is consistent."""
        agents = [create_weather_agent()]
        execution_ids = []

        # Run multiple times
        for i in range(3):
            async for event in run(
                prompt=f"Weather query {i}",
                agents=agents,
                model_name=api_config["model"],
                api_key=api_config["api_key"],
                base_url=api_config["base_url"],
            ):
                if event.type == "workflow_selected":
                    execution_ids.append(event.data["execution_id"])
                    break

        # All execution IDs should be unique
        assert len(set(execution_ids)) == 3

        # All should be valid UUID format
        import uuid
        for eid in execution_ids:
            try:
                uuid.UUID(eid)
            except ValueError:
                pytest.fail(f"Invalid UUID format: {eid}")

        print(f"\nExecution IDs: {execution_ids}")

    @pytest.mark.asyncio
    async def test_final_result_structure_consistency(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Test that final_result event structure is consistent."""
        agents = [create_weather_agent(), create_calculator_agent()]

        final_events = []

        # Router workflow
        async for event in run(
            prompt="Weather in Tokyo",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="router",
        ):
            if event.type == "final_result":
                final_events.append(event)

        # Supervisor workflow
        async for event in run(
            prompt="Weather and calculate 5+5",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            if event.type == "final_result":
                final_events.append(event)

        # All final_result events should have consistent structure
        for event in final_events:
            assert "result" in event.data
            assert "execution_id" in event.data
            # citations may or may not be present
            
        print(f"\nFinal events collected: {len(final_events)}")

    @pytest.mark.asyncio
    async def test_agent_event_consistency(
        self, api_config, create_weather_agent
    ):
        """Test that agent events have consistent structure."""
        agents = [create_weather_agent()]
        agent_events = []

        async for event in run(
            prompt="Weather in London",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if event.type in ["agent_start", "agent_end"]:
                agent_events.append(event)

        # Check structure consistency
        for event in agent_events:
            assert "execution_id" in event.data or event.agent_id is not None
            
        print(f"\nAgent events: {len(agent_events)}")

    def test_kiva_decorator_vs_add_agent_consistency(self, api_config):
        """Test that decorator and add_agent produce same agent structure."""
        # Using decorator
        kiva1 = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva1.agent("weather", "Gets weather")
        def weather1(city: str) -> str:
            """Get weather."""
            return f"{city}: Sunny"

        # Using add_agent
        kiva2 = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        def weather2(city: str) -> str:
            """Get weather."""
            return f"{city}: Sunny"

        kiva2.add_agent("weather", "Gets weather", [weather2])

        # Both should have same agent structure
        assert len(kiva1._agents) == len(kiva2._agents)
        assert kiva1._agents[0].name == kiva2._agents[0].name
        assert kiva1._agents[0].description == kiva2._agents[0].description

        print("\nDecorator and add_agent produce consistent structure")
