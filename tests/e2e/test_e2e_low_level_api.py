"""E2E test for Low-Level API (run() function).

Tests the run() async generator for full control over event streaming.
"""

import pytest

from kiva import run, StreamEvent


class TestLowLevelAPIE2E:
    """E2E tests for the low-level run() API."""

    @pytest.mark.asyncio
    async def test_run_returns_async_iterator(
        self, api_config, create_weather_agent
    ):
        """Test that run() returns an async iterator."""
        agents = [create_weather_agent()]
        
        result = run(
            prompt="Weather in Beijing",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        )
        
        # Should be an async iterator
        assert hasattr(result, "__anext__")
        
        # Consume the iterator
        events = []
        async for event in result:
            events.append(event)
        
        assert len(events) > 0
        print(f"\nCollected {len(events)} events")

    @pytest.mark.asyncio
    async def test_run_yields_stream_events(
        self, api_config, create_weather_agent
    ):
        """Test that run() yields StreamEvent objects."""
        agents = [create_weather_agent()]
        
        async for event in run(
            prompt="Weather in Tokyo",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            assert isinstance(event, StreamEvent)
            assert hasattr(event, "type")
            assert hasattr(event, "data")
            assert hasattr(event, "timestamp")
            
        print("\nAll events are StreamEvent instances")

    @pytest.mark.asyncio
    async def test_run_event_types(
        self, api_config, create_weather_agent
    ):
        """Test various event types emitted by run()."""
        agents = [create_weather_agent()]
        event_types = set()
        
        async for event in run(
            prompt="What's the weather in London?",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            event_types.add(event.type)
        
        # Must have these essential events
        assert "workflow_selected" in event_types
        assert "final_result" in event_types
        
        print(f"\nEvent types: {event_types}")

    @pytest.mark.asyncio
    async def test_run_token_streaming(
        self, api_config, create_weather_agent
    ):
        """Test token streaming events."""
        agents = [create_weather_agent()]
        token_events = []
        
        async for event in run(
            prompt="Weather in Paris",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if event.type == "token":
                token_events.append(event)
        
        # Should have some token events (streaming)
        print(f"\nToken events: {len(token_events)}")
        if token_events:
            print(f"Sample token: {token_events[0].data.get('content', '')[:50]}")

    @pytest.mark.asyncio
    async def test_run_workflow_selected_event(
        self, api_config, create_weather_agent
    ):
        """Test workflow_selected event structure."""
        agents = [create_weather_agent()]
        workflow_event = None
        
        async for event in run(
            prompt="Weather in New York",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if event.type == "workflow_selected":
                workflow_event = event
                break
        
        assert workflow_event is not None
        assert "workflow" in workflow_event.data
        assert workflow_event.data["workflow"] in ["router", "supervisor", "parliament"]
        assert "execution_id" in workflow_event.data
        
        print(f"\nWorkflow: {workflow_event.data['workflow']}")
        print(f"Complexity: {workflow_event.data.get('complexity')}")

    @pytest.mark.asyncio
    async def test_run_final_result_event(
        self, api_config, create_weather_agent
    ):
        """Test final_result event structure."""
        agents = [create_weather_agent()]
        final_event = None
        
        async for event in run(
            prompt="Weather in Beijing",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if event.type == "final_result":
                final_event = event
        
        assert final_event is not None
        assert "result" in final_event.data
        assert "execution_id" in final_event.data
        
        print(f"\nFinal Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_run_execution_id_consistency(
        self, api_config, create_weather_agent
    ):
        """Test that execution_id is consistent across events."""
        agents = [create_weather_agent()]
        execution_ids = set()
        
        async for event in run(
            prompt="Weather in Tokyo",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if "execution_id" in event.data:
                execution_ids.add(event.data["execution_id"])
        
        # All events should have the same execution_id
        assert len(execution_ids) == 1, f"Multiple execution IDs found: {execution_ids}"
        print(f"\nConsistent execution_id: {execution_ids.pop()}")

    @pytest.mark.asyncio
    async def test_run_with_multiple_agents(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Test run() with multiple agents."""
        agents = [create_weather_agent(), create_calculator_agent()]
        events = []
        
        async for event in run(
            prompt="Weather in London and calculate 50 * 2",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)
        
        event_types = [e.type for e in events]
        assert "workflow_selected" in event_types
        assert "final_result" in event_types
        
        final_event = next(e for e in events if e.type == "final_result")
        print(f"\nMulti-agent Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_run_event_timestamps(
        self, api_config, create_weather_agent
    ):
        """Test that events have valid timestamps."""
        agents = [create_weather_agent()]
        timestamps = []
        
        async for event in run(
            prompt="Weather in Paris",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            timestamps.append(event.timestamp)
        
        # All timestamps should be positive
        assert all(t > 0 for t in timestamps)
        
        # Timestamps should be roughly in order (allowing for parallel execution)
        print(f"\nTimestamp range: {min(timestamps):.2f} - {max(timestamps):.2f}")

    @pytest.mark.asyncio
    async def test_run_agent_events(
        self, api_config, create_weather_agent
    ):
        """Test agent_start and agent_end events."""
        agents = [create_weather_agent()]
        agent_events = []
        
        async for event in run(
            prompt="Weather in Beijing",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if event.type in ["agent_start", "agent_end"]:
                agent_events.append(event)
        
        print(f"\nAgent events: {len(agent_events)}")
        for e in agent_events:
            print(f"  {e.type}: {e.agent_id or e.data.get('agent_id')}")
