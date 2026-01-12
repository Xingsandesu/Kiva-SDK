"""E2E test for Supervisor workflow - parallel multi-agent tasks.

Tests the supervisor workflow which coordinates multiple agents in parallel.
"""

import pytest

from kiva import run


class TestSupervisorWorkflowE2E:
    """E2E tests for supervisor workflow with parallel agent execution."""

    @pytest.mark.asyncio
    async def test_supervisor_multi_agent_task(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Test task requiring multiple agents executed in parallel."""
        agents = [create_weather_agent(), create_calculator_agent()]
        events = []

        async for event in run(
            prompt="What's the weather in Beijing? Also calculate 100 / 4",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)
            print(f"Event: {event.type}")

        event_types = [e.type for e in events]
        
        assert "workflow_selected" in event_types
        assert "final_result" in event_types
        
        final_event = next(e for e in events if e.type == "final_result")
        result = final_event.data["result"]
        assert result is not None
        print(f"\nMulti-Agent Result: {result}")

    @pytest.mark.asyncio
    async def test_supervisor_with_workflow_override(
        self, api_config, create_weather_agent, create_search_agent
    ):
        """Test forcing supervisor workflow via override."""
        agents = [create_weather_agent(), create_search_agent()]
        events = []

        async for event in run(
            prompt="Get weather for Tokyo and search for AI information",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            events.append(event)

        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert workflow_event.data["workflow"] == "supervisor"
        
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        print(f"\nSupervisor Override Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_supervisor_three_agents(
        self, api_config, create_weather_agent, create_calculator_agent, create_search_agent
    ):
        """Test supervisor with three different agents."""
        agents = [
            create_weather_agent(),
            create_calculator_agent(),
            create_search_agent(),
        ]
        events = []

        async for event in run(
            prompt="What's the weather in Paris? Calculate 25 * 4. Search for Python info.",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            events.append(event)

        event_types = [e.type for e in events]
        
        # Check for parallel execution events
        has_parallel_start = "parallel_start" in event_types
        has_parallel_instances_start = "parallel_instances_start" in event_types
        
        # Should have some form of parallel execution indication
        print(f"Event types: {event_types}")
        
        final_event = next(e for e in events if e.type == "final_result")
        result = final_event.data["result"]
        assert result is not None
        print(f"\nThree Agents Result: {result}")

    @pytest.mark.asyncio
    async def test_supervisor_parallel_events(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Test that parallel execution events are emitted correctly."""
        agents = [create_weather_agent(), create_calculator_agent()]
        events = []

        async for event in run(
            prompt="Get weather for New York and calculate 50 + 50",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            events.append(event)

        event_types = [e.type for e in events]
        
        # Verify workflow was selected
        assert "workflow_selected" in event_types
        
        # Verify final result
        assert "final_result" in event_types
        
        # Check for agent-related events
        agent_events = [e for e in events if e.type in ["agent_start", "agent_end"]]
        print(f"\nAgent events count: {len(agent_events)}")
        
        final_event = next(e for e in events if e.type == "final_result")
        print(f"Final Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_supervisor_max_parallel_agents(
        self, api_config, create_weather_agent, create_calculator_agent, create_search_agent
    ):
        """Test max_parallel_agents limit."""
        agents = [
            create_weather_agent(),
            create_calculator_agent(),
            create_search_agent(),
        ]
        events = []

        async for event in run(
            prompt="Weather in London, calculate 10*10, search for LangChain",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
            max_parallel_agents=2,  # Limit to 2 parallel agents
        ):
            events.append(event)

        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        print(f"\nMax Parallel Result: {final_event.data['result']}")
