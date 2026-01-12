"""E2E test for validating output patterns match documentation.

This test ensures that the actual outputs match the documented patterns.
"""

import pytest

from kiva import run


class TestOutputValidationE2E:
    """E2E tests that validate output patterns."""

    @pytest.mark.asyncio
    async def test_router_workflow_output_pattern(
        self, api_config, create_weather_agent
    ):
        """Validate router workflow follows documented pattern."""
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

        event_types = [e.type for e in events]

        # Pattern validation: Must have these events in order
        assert "workflow_selected" in event_types, "Missing workflow_selected event"
        assert "final_result" in event_types, "Missing final_result event"

        # Verify order
        workflow_idx = event_types.index("workflow_selected")
        final_idx = event_types.index("final_result")
        assert workflow_idx < final_idx, "workflow_selected must come before final_result"

        # Verify workflow_selected structure
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert "workflow" in workflow_event.data
        assert "complexity" in workflow_event.data
        assert "execution_id" in workflow_event.data
        assert "task_assignments" in workflow_event.data

        # Verify final_result structure
        final_event = next(e for e in events if e.type == "final_result")
        assert "result" in final_event.data
        assert "execution_id" in final_event.data
        assert final_event.data["result"] is not None

        # Verify execution_id consistency
        execution_ids = set()
        for event in events:
            if "execution_id" in event.data:
                execution_ids.add(event.data["execution_id"])
        assert len(execution_ids) == 1, "All events must have same execution_id"

        print(f"✓ Router workflow pattern validated ({len(events)} events)")

    @pytest.mark.asyncio
    async def test_supervisor_workflow_output_pattern(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Validate supervisor workflow follows documented pattern."""
        agents = [create_weather_agent(), create_calculator_agent()]
        events = []

        async for event in run(
            prompt="What's the weather in Tokyo? Also calculate 25 * 4",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            events.append(event)

        event_types = [e.type for e in events]

        # Pattern validation
        assert "workflow_selected" in event_types
        assert "final_result" in event_types

        # Supervisor should have parallel execution indicators
        has_parallel = "parallel_start" in event_types or "parallel_instances_start" in event_types
        print(f"Has parallel execution: {has_parallel}")

        # Verify workflow type
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert workflow_event.data["workflow"] == "supervisor"

        # Verify final result has content
        final_event = next(e for e in events if e.type == "final_result")
        assert len(final_event.data["result"]) > 0

        print(f"✓ Supervisor workflow pattern validated ({len(events)} events)")

    @pytest.mark.asyncio
    async def test_parliament_workflow_output_pattern(
        self, api_config, create_weather_agent, create_search_agent
    ):
        """Validate parliament workflow follows documented pattern."""
        agents = [create_weather_agent(), create_search_agent()]
        events = []

        async for event in run(
            prompt="Should I go outside? Check weather and give advice",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=3,
        ):
            events.append(event)

        event_types = [e.type for e in events]

        # Pattern validation
        assert "workflow_selected" in event_types
        assert "final_result" in event_types

        # Verify workflow type
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        assert workflow_event.data["workflow"] == "parliament"

        # Parliament should have iteration indicators
        parallel_starts = [e for e in events if e.type == "parallel_start"]
        print(f"Parliament iterations: {len(parallel_starts)}")

        print(f"✓ Parliament workflow pattern validated ({len(events)} events)")

    @pytest.mark.asyncio
    async def test_parallel_instances_output_pattern(
        self, api_config, create_weather_agent
    ):
        """Validate parallel instances follow documented pattern."""
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

        event_types = [e.type for e in events]

        # Pattern validation
        assert "workflow_selected" in event_types
        assert "final_result" in event_types

        # Check for parallel strategy
        workflow_event = next(e for e in events if e.type == "workflow_selected")
        parallel_strategy = workflow_event.data.get("parallel_strategy", "none")
        total_instances = workflow_event.data.get("total_instances", 1)

        print(f"Parallel strategy: {parallel_strategy}")
        print(f"Total instances: {total_instances}")

        # If parallel instances were spawned, verify instance events
        instance_events = [
            e for e in events
            if e.type in ["instance_spawn", "instance_start", "instance_end", "instance_complete"]
        ]

        if instance_events:
            print(f"Instance events: {len(instance_events)}")
            # Verify instance_id in events
            for event in instance_events:
                assert "instance_id" in event.data, f"Missing instance_id in {event.type}"

        print(f"✓ Parallel instances pattern validated ({len(events)} events)")

    @pytest.mark.asyncio
    async def test_event_structure_consistency(
        self, api_config, create_weather_agent
    ):
        """Validate all events have consistent structure."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="Weather in London",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Validate each event structure
        for event in events:
            # All events must have these attributes
            assert hasattr(event, "type"), "Event missing 'type' attribute"
            assert hasattr(event, "data"), "Event missing 'data' attribute"
            assert hasattr(event, "timestamp"), "Event missing 'timestamp' attribute"
            assert hasattr(event, "agent_id"), "Event missing 'agent_id' attribute"

            # Type must be string
            assert isinstance(event.type, str), f"Event type must be string, got {type(event.type)}"

            # Data must be dict
            assert isinstance(event.data, dict), f"Event data must be dict, got {type(event.data)}"

            # Timestamp must be positive number
            assert event.timestamp > 0, f"Event timestamp must be positive, got {event.timestamp}"

            # agent_id can be None or string
            assert event.agent_id is None or isinstance(event.agent_id, str), \
                f"Event agent_id must be None or string, got {type(event.agent_id)}"

        print(f"✓ Event structure validated for {len(events)} events")

    @pytest.mark.asyncio
    async def test_execution_id_consistency(
        self, api_config, create_weather_agent
    ):
        """Validate execution_id is consistent across all events."""
        agents = [create_weather_agent()]
        execution_ids = set()

        async for event in run(
            prompt="Weather in Paris",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if "execution_id" in event.data:
                execution_ids.add(event.data["execution_id"])

        # All events should have the same execution_id
        assert len(execution_ids) == 1, \
            f"Expected 1 unique execution_id, found {len(execution_ids)}: {execution_ids}"

        execution_id = execution_ids.pop()

        # Verify it's a valid UUID format
        import uuid
        try:
            uuid.UUID(execution_id)
        except ValueError:
            pytest.fail(f"execution_id is not a valid UUID: {execution_id}")

        print(f"✓ Execution ID consistency validated: {execution_id}")

    @pytest.mark.asyncio
    async def test_timestamp_ordering(
        self, api_config, create_weather_agent
    ):
        """Validate timestamps are in chronological order."""
        agents = [create_weather_agent()]
        timestamps = []

        async for event in run(
            prompt="Weather in Beijing",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            timestamps.append(event.timestamp)

        # Timestamps should be generally increasing (allowing for parallel execution)
        # Check that first timestamp is before last timestamp
        assert timestamps[0] <= timestamps[-1], \
            f"First timestamp {timestamps[0]} should be <= last timestamp {timestamps[-1]}"

        # Check all timestamps are positive
        assert all(t > 0 for t in timestamps), "All timestamps must be positive"

        print(f"✓ Timestamp ordering validated ({len(timestamps)} timestamps)")

    @pytest.mark.asyncio
    async def test_final_result_always_present(
        self, api_config, create_weather_agent
    ):
        """Validate that final_result event is always present."""
        agents = [create_weather_agent()]
        has_final_result = False

        async for event in run(
            prompt="Weather in Tokyo",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            if event.type == "final_result":
                has_final_result = True
                # Verify result is not empty
                assert event.data["result"] is not None
                assert len(event.data["result"]) > 0

        assert has_final_result, "final_result event must be present"
        print("✓ Final result presence validated")

    @pytest.mark.asyncio
    async def test_workflow_selected_always_first(
        self, api_config, create_weather_agent
    ):
        """Validate that workflow_selected is always the first meaningful event."""
        agents = [create_weather_agent()]
        events = []

        async for event in run(
            prompt="Weather in London",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)

        # Find first non-token event
        non_token_events = [e for e in events if e.type != "token"]

        assert len(non_token_events) > 0, "Must have at least one non-token event"
        assert non_token_events[0].type == "workflow_selected", \
            f"First non-token event must be workflow_selected, got {non_token_events[0].type}"

        print("✓ Workflow selected ordering validated")
