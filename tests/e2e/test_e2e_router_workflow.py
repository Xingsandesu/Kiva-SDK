"""E2E test for Router workflow - simple single-agent tasks.

Tests the router workflow which routes tasks to a single agent.
"""

import pytest

from kiva import run


class TestRouterWorkflowE2E:
    """E2E tests for router workflow with single agent execution."""

    @pytest.mark.asyncio
    async def test_router_simple_weather_query(
        self, api_config, create_weather_agent
    ):
        """Test simple weather query routed to single agent."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="What's the weather in Beijing?",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)
            print(f"Event: {event.type} - {event.data}")

        # Verify event sequence
        event_types = [e.type for e in events]
        
        # Must have workflow_selected
        assert "workflow_selected" in event_types, "Should have workflow_selected event"
        
        # Must have final_result
        assert "final_result" in event_types, "Should have final_result event"
        
        # Get workflow info
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert workflow_event.data["workflow"] in ["router", "supervisor", "parliament"]
        
        # Get final result
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        assert len(final_event.data["result"]) > 0
        
        print(f"\nFinal Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_router_simple_calculation(
        self, api_config, create_calculator_agent
    ):
        """Test simple calculation routed to calculator agent."""
        agents = [create_calculator_agent()]
        events = []

        async for event in run(
            prompt="Calculate 15 * 8",
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
        result = final_event.data["result"]
        assert result is not None
        # The result should contain 120 (15 * 8)
        print(f"\nCalculation Result: {result}")

    @pytest.mark.asyncio
    async def test_router_with_workflow_override(
        self, api_config, create_weather_agent
    ):
        """Test forcing router workflow via override."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="What's the weather in Tokyo?",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="router",  # Force router workflow
        ):
            events.append(event)

        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert workflow_event.data["workflow"] == "router"
        
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        print(f"\nRouter Override Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_router_search_query(
        self, api_config, create_search_agent
    ):
        """Test search query routed to search agent."""
        agents = [create_search_agent()]
        events = []

        async for event in run(
            prompt="Search for information about Python",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        final_event = next(e for e in events if e.type == "final_result")
        result = final_event.data["result"]
        assert result is not None
        print(f"\nSearch Result: {result}")

    @pytest.mark.asyncio
    async def test_router_event_sequence_order(
        self, api_config, create_weather_agent
    ):
        """Test that events are emitted in correct order."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="What's the weather in London?",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="router",
        ):
            events.append(event)

        event_types = [e.type for e in events]
        
        # workflow_selected should come before final_result
        workflow_idx = event_types.index("workflow_selected")
        final_idx = event_types.index("final_result")
        assert workflow_idx < final_idx, "workflow_selected should come before final_result"
        
        # final_result should be the last meaningful event
        assert event_types[-1] == "final_result" or event_types.index("final_result") == len(event_types) - 1
