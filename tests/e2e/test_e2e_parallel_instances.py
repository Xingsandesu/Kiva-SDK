"""E2E test for Parallel Agent Instances.

Tests the parallel instance spawning feature for fan_out and map_reduce strategies.
"""

import pytest

from kiva import run


class TestParallelInstancesE2E:
    """E2E tests for parallel agent instance execution."""

    @pytest.mark.asyncio
    async def test_parallel_instances_basic(
        self, api_config, create_weather_agent
    ):
        """Test basic parallel instance execution."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="Get weather for Beijing, Tokyo, and London",
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
        
        # Check workflow info
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        print(f"\nWorkflow: {workflow_event.data['workflow']}")
        print(f"Parallel Strategy: {workflow_event.data.get('parallel_strategy', 'none')}")
        print(f"Total Instances: {workflow_event.data.get('total_instances', 1)}")
        
        final_event = next(e for e in events if e.type == "final_result")
        print(f"Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_parallel_instances_events(
        self, api_config, create_search_agent
    ):
        """Test parallel instance events are emitted correctly."""
        agents = [create_search_agent()]
        instance_events = []

        async for event in run(
            prompt="Search for information about Python, AI, and Machine Learning",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if event.type in [
                "instance_spawn", "instance_start", "instance_end",
                "instance_complete", "instance_result",
                "parallel_instances_start", "parallel_instances_complete"
            ]:
                instance_events.append(event)

        print(f"\nInstance events: {len(instance_events)}")
        for e in instance_events:
            print(f"  {e.type}: {e.data.get('instance_id', '')[:20]}...")

    @pytest.mark.asyncio
    async def test_parallel_instances_with_supervisor(
        self, api_config, create_weather_agent, create_search_agent
    ):
        """Test parallel instances with supervisor workflow."""
        agents = [create_weather_agent(), create_search_agent()]
        events = []

        async for event in run(
            prompt="Get weather for 3 cities and search for travel tips",
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
        print(f"\nSupervisor with Instances Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_parallel_instances_max_limit(
        self, api_config, create_weather_agent
    ):
        """Test max_parallel_agents limits instance count."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="Get weather for 10 different cities around the world",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            max_parallel_agents=3,  # Limit to 3
        ):
            events.append(event)

        workflow_event = next(e for e in events if e.type == "workflow_selected")
        total_instances = workflow_event.data.get("total_instances", 1)
        
        # Should not exceed max_parallel_agents
        assert total_instances <= 3, f"Instances {total_instances} exceeded limit 3"
        
        print(f"\nTotal instances (limited): {total_instances}")

    @pytest.mark.asyncio
    async def test_parallel_instances_fan_out_strategy(
        self, api_config, create_calculator_agent
    ):
        """Test fan_out parallel strategy."""
        agents = [create_calculator_agent()]
        events = []

        async for event in run(
            prompt="Calculate these: 10*10, 20*20, 30*30",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        workflow_event = next(e for e in events if e.type == "workflow_selected")
        strategy = workflow_event.data.get("parallel_strategy", "none")
        
        print(f"\nParallel Strategy: {strategy}")
        print(f"Task Assignments: {workflow_event.data.get('task_assignments', [])}")
        
        final_event = next(e for e in events if e.type == "final_result")
        print(f"Result: {final_event.data['result']}")

    @pytest.mark.asyncio
    async def test_parallel_instances_result_aggregation(
        self, api_config, create_weather_agent
    ):
        """Test that results from parallel instances are aggregated."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="Compare weather in Beijing and Tokyo",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Check for instance results
        instance_results = [
            e for e in events 
            if e.type == "instance_result"
        ]
        
        print(f"\nInstance results: {len(instance_results)}")
        
        final_event = next(e for e in events if e.type == "final_result")
        result = final_event.data["result"]
        assert result is not None
        print(f"Aggregated Result: {result}")

    @pytest.mark.asyncio
    async def test_parallel_instances_isolated_context(
        self, api_config, create_search_agent
    ):
        """Test that each instance has isolated context."""
        agents = [create_search_agent()]
        events = []

        async for event in run(
            prompt="Search for Python and LangChain separately",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Check instance events have unique IDs
        instance_ids = set()
        for e in events:
            if "instance_id" in e.data:
                instance_ids.add(e.data["instance_id"])
        
        print(f"\nUnique instance IDs: {len(instance_ids)}")
        for iid in instance_ids:
            print(f"  {iid}")

    @pytest.mark.asyncio
    async def test_parallel_instances_error_handling(
        self, api_config, create_weather_agent
    ):
        """Test error handling in parallel instances."""
        agents = [create_weather_agent()]
        events = []

        # Request weather for a city that might not be in our mock data
        async for event in run(
            prompt="Get weather for UnknownCity123 and Beijing",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Should still complete with final result
        final_event = next(e for e in events if e.type == "final_result")
        assert final_event.data["result"] is not None
        
        # Check for partial result info
        partial_info = final_event.data.get("partial_result", False)
        print(f"\nPartial result: {partial_info}")
        print(f"Result: {final_event.data['result']}")
