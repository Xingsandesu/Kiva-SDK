"""E2E test for Parliament workflow - iterative conflict resolution.

Tests the parliament workflow which handles complex reasoning with
potential conflicts between agents.
"""

import pytest

from kiva import run


class TestParliamentWorkflowE2E:
    """E2E tests for parliament workflow with conflict resolution."""

    @pytest.mark.asyncio
    async def test_parliament_basic_execution(
        self, api_config, create_weather_agent, create_search_agent
    ):
        """Test basic parliament workflow execution."""
        agents = [create_weather_agent(), create_search_agent()]
        events = []

        async for event in run(
            prompt="Is it a good day for outdoor activities in Beijing? Consider weather and general advice.",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=3,
        ):
            events.append(event)
            print(f"Event: {event.type}")

        event_types = [e.type for e in events]
        
        assert "workflow_selected" in event_types
        assert "final_result" in event_types
        
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert workflow_event.data["workflow"] == "parliament"
        
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        print(f"\nParliament Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_parliament_max_iterations_respected(
        self, api_config, create_weather_agent, create_search_agent
    ):
        """Test that parliament respects max_iterations limit."""
        agents = [create_weather_agent(), create_search_agent()]
        events = []
        iteration_count = 0

        async for event in run(
            prompt="Analyze the best travel destination considering weather and information",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=2,  # Low limit
        ):
            events.append(event)
            # Count parallel_start events as iteration indicators
            if event.type == "parallel_start":
                iteration_count += 1

        # Should not exceed max_iterations
        assert iteration_count <= 2, f"Iterations {iteration_count} exceeded max 2"
        
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        print(f"\nMax Iterations Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_parliament_with_three_agents(
        self, api_config, create_weather_agent, create_calculator_agent, create_search_agent
    ):
        """Test parliament with three agents for complex reasoning."""
        agents = [
            create_weather_agent(),
            create_calculator_agent(),
            create_search_agent(),
        ]
        events = []

        async for event in run(
            prompt="Plan a trip: check weather in Tokyo, calculate budget (1000 * 7 days), and search for travel tips",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=3,
        ):
            events.append(event)

        event_types = [e.type for e in events]
        
        assert "workflow_selected" in event_types
        assert "final_result" in event_types
        
        final_event = next(e for e in events if e.type == "final_result")
        result = final_event.data["result"]
        assert result is not None
        print(f"\nThree Agents Parliament Result: {result}")

    @pytest.mark.asyncio
    async def test_parliament_conflict_detection(
        self, api_config, create_weather_agent, create_search_agent
    ):
        """Test parliament's ability to handle potential conflicts."""
        agents = [create_weather_agent(), create_search_agent()]
        events = []

        async for event in run(
            prompt="Should I go outside today? Get weather info and general outdoor advice.",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=3,
        ):
            events.append(event)

        # Check for parallel execution events (indicating multiple rounds)
        parallel_events = [e for e in events if e.type in ["parallel_start", "parallel_complete"]]
        print(f"\nParallel events: {len(parallel_events)}")
        
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        print(f"Conflict Resolution Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_parliament_single_iteration_no_conflict(
        self, api_config, create_weather_agent
    ):
        """Test parliament with single agent (no conflict possible)."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="What's the weather in Beijing?",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=5,
        ):
            events.append(event)

        # With single agent, should complete quickly without multiple iterations
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        print(f"\nSingle Agent Parliament Result: {final_event.data['result']}")
