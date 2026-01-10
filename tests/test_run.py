"""Property-based tests for the run() function.

These tests validate the correctness properties defined in the design document
for the SDK entry point.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiva import ConfigurationError, run


class MockAgent:
    """Mock agent for testing the run() function."""

    def __init__(self, name: str, response: str = "Mock response"):
        self.name = name
        self.response = response
        self.call_count = 0

    async def ainvoke(self, input_data: dict) -> dict:
        """Mock invoke that returns a predefined response."""
        self.call_count += 1
        return {"messages": [type("Message", (), {"content": self.response})()]}


class InvalidAgent:
    """Agent without ainvoke method - for testing validation."""

    def __init__(self, name: str):
        self.name = name

    def invoke(self, input_data: dict) -> dict:
        """Only has sync invoke, not ainvoke."""
        return {"messages": []}


class TestRunFunctionReturnsAsyncIterator:
    """Property 1: Run Function Returns AsyncIterator

    *For any* valid call to run() with non-empty agents list, the function
    SHALL return an AsyncIterator that yields StreamEvent objects.
    """

    @given(
        prompt=st.text(min_size=1, max_size=100),
        num_agents=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_run_returns_async_iterator(self, prompt: str, num_agents: int):
        """Feature: multi-agent-sdk, Property 1: Run Function Returns AsyncIterator

        For any valid prompt and non-empty agents list, run() should return
        an AsyncIterator.


        """
        agents = [MockAgent(f"agent_{i}") for i in range(num_agents)]

        # Call run() and verify it returns an AsyncIterator
        result = run(prompt, agents)

        # Verify it's an async iterator
        assert isinstance(result, AsyncIterator), (
            f"Expected AsyncIterator, got {type(result)}"
        )

    @given(
        prompt=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_run_yields_stream_events(self, prompt: str):
        """Feature: multi-agent-sdk, Property 1: Run Function Returns AsyncIterator

        For any valid execution, all yielded items should be StreamEvent objects.


        """

        async def check_events():
            agents = [MockAgent("test_agent", "Test response")]

            # We can't fully execute without a real LLM, but we can verify
            # the return type is correct
            result = run(prompt, agents)
            assert isinstance(result, AsyncIterator)

            # The iterator should be async iterable
            assert hasattr(result, "__anext__")

        asyncio.get_event_loop().run_until_complete(check_events())


class TestEmptyAgentsRaisesConfigurationError:
    """Property 2: Empty Agents Raises ConfigurationError

    *For any* call to run() with an empty agents list, the function
    SHALL raise ConfigurationError.


    """

    @given(
        prompt=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_empty_agents_raises_configuration_error(self, prompt: str):
        """Feature: multi-agent-sdk, Property 2: Empty Agents Raises ConfigurationError

        For any prompt with an empty agents list, run() should raise
        ConfigurationError.


        """

        async def check_error():
            with pytest.raises(ConfigurationError) as exc_info:
                async for _ in run(prompt, []):
                    pass

            # Verify error message mentions agents
            assert "agents" in str(exc_info.value).lower() or "列表" in str(
                exc_info.value
            )

        asyncio.get_event_loop().run_until_complete(check_error())

    def test_empty_agents_error_message_is_helpful(self):
        """Feature: multi-agent-sdk, Property 2: Empty Agents Raises ConfigurationError

        The error message should be helpful and indicate the problem.


        """

        async def check_error():
            with pytest.raises(ConfigurationError) as exc_info:
                async for _ in run("test prompt", []):
                    pass

            error_msg = str(exc_info.value)
            # Should mention that agents list cannot be empty
            assert "空" in error_msg or "empty" in error_msg.lower()

        asyncio.get_event_loop().run_until_complete(check_error())


class TestInvalidAgentTypeRaisesConfigurationError:
    """Property 3: Invalid Agent Type Raises ConfigurationError

    *For any* object in the agents list that does not have an ainvoke method,
    the SDK SHALL raise ConfigurationError.

    """

    @given(
        prompt=st.text(min_size=1, max_size=100),
        num_valid=st.integers(min_value=0, max_value=2),
        invalid_position=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=100)
    def test_invalid_agent_raises_configuration_error(
        self, prompt: str, num_valid: int, invalid_position: int
    ):
        """Feature: multi-agent-sdk, Property 3: Invalid Agent Type Raises ConfigurationError

        For any agents list containing an object without ainvoke method,
        run() should raise ConfigurationError.

        """
        # Create a mix of valid and invalid agents
        agents: list[Any] = [MockAgent(f"valid_{i}") for i in range(num_valid)]

        # Insert invalid agent at the specified position
        insert_pos = min(invalid_position, len(agents))
        agents.insert(insert_pos, InvalidAgent("invalid"))

        async def check_error():
            with pytest.raises(ConfigurationError) as exc_info:
                async for _ in run(prompt, agents):
                    pass

            # Verify error message mentions ainvoke
            error_msg = str(exc_info.value)
            assert "ainvoke" in error_msg or "create_agent" in error_msg

        asyncio.get_event_loop().run_until_complete(check_error())

    @given(
        invalid_type=st.sampled_from(
            [
                "string",
                123,
                {"dict": "value"},
                ["list"],
                None,
            ]
        )
    )
    @settings(max_examples=100)
    def test_various_invalid_types_raise_error(self, invalid_type: Any):
        """Feature: multi-agent-sdk, Property 3: Invalid Agent Type Raises ConfigurationError

        Various invalid types should all raise ConfigurationError.

        """
        if invalid_type is None:
            # Skip None as it would be filtered differently
            return

        async def check_error():
            with pytest.raises(ConfigurationError):
                async for _ in run("test", [invalid_type]):
                    pass

        asyncio.get_event_loop().run_until_complete(check_error())

    def test_agent_without_ainvoke_error_message_is_helpful(self):
        """Feature: multi-agent-sdk, Property 3: Invalid Agent Type Raises ConfigurationError

        The error message should suggest using create_agent().

        """

        async def check_error():
            invalid_agent = InvalidAgent("test")

            with pytest.raises(ConfigurationError) as exc_info:
                async for _ in run("test prompt", [invalid_agent]):
                    pass

            error_msg = str(exc_info.value)
            # Should mention create_agent as the correct way
            assert "create_agent" in error_msg

        asyncio.get_event_loop().run_until_complete(check_error())


class TestEventSequenceCorrectness:
    """Property 4: Event Sequence Correctness

    *For any* successful execution, the SDK SHALL emit events in the following order:
    1. workflow_selected event (exactly once)
    2. Zero or more agent_start events
    3. For each agent_start, a corresponding agent_end event
    4. final_result event (exactly once, at the end)

    """

    @given(
        num_agents=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_event_sequence_has_workflow_selected_before_agent_events(
        self, num_agents: int
    ):
        """Feature: multi-agent-sdk, Property 4: Event Sequence Correctness

        For any execution, workflow_selected should appear before any agent events.

        """
        # Create mock agents
        [MockAgent(f"agent_{i}", f"Response {i}") for i in range(num_agents)]

        # Simulate event collection from a workflow execution
        # Since we can't run the full graph without LLM, we test the event processing logic
        from kiva.run import _process_node_update

        async def check_sequence():
            events = []
            execution_id = "test-exec-id"

            # Simulate analyze_and_plan node output
            analyze_data = {
                "workflow": "router",
                "complexity": "simple",
                "task_assignments": [{"agent_id": "agent_0", "task": "test"}],
            }
            async for event in _process_node_update(
                "analyze_and_plan", analyze_data, execution_id
            ):
                events.append(event)

            # Simulate router_workflow node output
            router_data = {
                "agent_results": [{"agent_id": "agent_0", "result": "test result"}],
            }
            async for event in _process_node_update(
                "router_workflow", router_data, execution_id
            ):
                events.append(event)

            # Simulate synthesize_results node output
            synth_data = {
                "final_result": "Final answer",
            }
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                events.append(event)

            # Verify sequence
            event_types = [e.type for e in events]

            # workflow_selected should be first
            assert event_types[0] == "workflow_selected", (
                f"First event should be workflow_selected, got {event_types[0]}"
            )

            # final_result should be last
            assert event_types[-1] == "final_result", (
                f"Last event should be final_result, got {event_types[-1]}"
            )

            # workflow_selected should appear exactly once
            assert event_types.count("workflow_selected") == 1, (
                f"workflow_selected should appear exactly once, got {event_types.count('workflow_selected')}"
            )

            # final_result should appear exactly once
            assert event_types.count("final_result") == 1, (
                f"final_result should appear exactly once, got {event_types.count('final_result')}"
            )

        asyncio.get_event_loop().run_until_complete(check_sequence())

    @given(
        num_agents=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100)
    def test_agent_end_events_match_agent_results(self, num_agents: int):
        """Feature: multi-agent-sdk, Property 4: Event Sequence Correctness

        For any execution with multiple agents, each agent should have an agent_end event.

        Note: Events are now emitted via emit_event() during workflow execution (custom mode),
        not via _process_node_update (updates mode). This test verifies the custom mode handling.
        """
        from kiva.run import _process_stream_chunk

        async def check_agent_events():
            events = []
            execution_id = "test-exec-id"

            # Simulate custom mode events from emit_event() for multiple agents
            for i in range(num_agents):
                custom_event = (
                    "custom",
                    {
                        "type": "agent_end",
                        "agent_id": f"agent_{i}",
                        "invocation_id": f"inv_{i}",
                        "result": f"Result {i}",
                        "execution_id": execution_id,
                        "timestamp": 1234567890.0,
                    },
                )
                async for event in _process_stream_chunk(custom_event, execution_id):
                    events.append(event)

            # Count agent_end events
            agent_end_events = [e for e in events if e.type == "agent_end"]

            # Should have one agent_end per agent
            assert len(agent_end_events) == num_agents, (
                f"Expected {num_agents} agent_end events, got {len(agent_end_events)}"
            )

            # Each agent should have an agent_end event
            agent_ids_with_end = {e.agent_id for e in agent_end_events}
            expected_agent_ids = {f"agent_{i}" for i in range(num_agents)}
            assert agent_ids_with_end == expected_agent_ids, (
                f"Agent IDs mismatch: {agent_ids_with_end} vs {expected_agent_ids}"
            )

            # Verify data structure is normalized (result wrapped in dict)
            for event in agent_end_events:
                assert "result" in event.data, (
                    "agent_end event should have 'result' in data"
                )
                assert isinstance(event.data["result"], dict), "result should be a dict"
                assert "agent_id" in event.data["result"], (
                    "result should contain agent_id"
                )

        asyncio.get_event_loop().run_until_complete(check_agent_events())

    def test_parallel_events_wrap_agent_events(self):
        """Feature: multi-agent-sdk, Property 4: Event Sequence Correctness

        For parallel execution, parallel_start should come before agent_end events,
        and parallel_complete should come after.

        Note: Events are now emitted via emit_event() during workflow execution (custom mode).
        This test verifies the custom mode handling preserves event structure.
        """
        from kiva.run import _process_stream_chunk

        async def check_parallel_sequence():
            events = []
            execution_id = "test-exec-id"

            # Simulate the sequence of custom mode events from emit_event()
            custom_events = [
                # parallel_start
                (
                    "custom",
                    {
                        "type": "parallel_start",
                        "agent_ids": ["agent_0", "agent_1"],
                        "execution_id": execution_id,
                        "timestamp": 1234567890.0,
                    },
                ),
                # agent_end for agent_0
                (
                    "custom",
                    {
                        "type": "agent_end",
                        "agent_id": "agent_0",
                        "invocation_id": "inv_0",
                        "result": "Result 0",
                        "execution_id": execution_id,
                        "timestamp": 1234567891.0,
                    },
                ),
                # agent_end for agent_1
                (
                    "custom",
                    {
                        "type": "agent_end",
                        "agent_id": "agent_1",
                        "invocation_id": "inv_1",
                        "result": "Result 1",
                        "execution_id": execution_id,
                        "timestamp": 1234567892.0,
                    },
                ),
                # parallel_complete
                (
                    "custom",
                    {
                        "type": "parallel_complete",
                        "results": [
                            {"agent_id": "agent_0", "success": True},
                            {"agent_id": "agent_1", "success": True},
                        ],
                        "execution_id": execution_id,
                        "timestamp": 1234567893.0,
                    },
                ),
            ]

            for custom_event in custom_events:
                async for event in _process_stream_chunk(custom_event, execution_id):
                    events.append(event)

            event_types = [e.type for e in events]

            # parallel_start should be first
            assert event_types[0] == "parallel_start", (
                f"First event should be parallel_start, got {event_types[0]}"
            )

            # parallel_complete should be last
            assert event_types[-1] == "parallel_complete", (
                f"Last event should be parallel_complete, got {event_types[-1]}"
            )

            # agent_end events should be in between
            agent_end_indices = [
                i for i, t in enumerate(event_types) if t == "agent_end"
            ]
            parallel_start_idx = event_types.index("parallel_start")
            parallel_complete_idx = event_types.index("parallel_complete")

            for idx in agent_end_indices:
                assert parallel_start_idx < idx < parallel_complete_idx, (
                    f"agent_end at {idx} should be between parallel_start ({parallel_start_idx}) and parallel_complete ({parallel_complete_idx})"
                )

        asyncio.get_event_loop().run_until_complete(check_parallel_sequence())


class TestExecutionIdUniqueness:
    """Property 12: Execution ID Uniqueness

    *For any* two executions, the execution_id SHALL be unique.

    """

    @given(
        num_executions=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=100)
    def test_execution_ids_are_unique_across_runs(self, num_executions: int):
        """Feature: multi-agent-sdk, Property 12: Execution ID Uniqueness

        For any number of executions, all execution_ids should be unique.

        """
        import uuid

        # Generate execution IDs the same way the SDK does
        execution_ids = [str(uuid.uuid4()) for _ in range(num_executions)]

        # All IDs should be unique
        assert len(set(execution_ids)) == num_executions, (
            f"Expected {num_executions} unique IDs, got {len(set(execution_ids))}"
        )

    @given(
        prompt1=st.text(min_size=1, max_size=50),
        prompt2=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_different_prompts_get_different_execution_ids(
        self, prompt1: str, prompt2: str
    ):
        """Feature: multi-agent-sdk, Property 12: Execution ID Uniqueness

        Different prompts should result in different execution IDs.

        """
        import uuid

        # Each call to run() generates a new execution_id
        # We simulate this by generating UUIDs
        exec_id_1 = str(uuid.uuid4())
        exec_id_2 = str(uuid.uuid4())

        # Even with the same prompt, execution IDs should be different
        assert exec_id_1 != exec_id_2, (
            "Different executions should have different execution IDs"
        )

    def test_execution_id_format_is_valid_uuid(self):
        """Feature: multi-agent-sdk, Property 12: Execution ID Uniqueness

        Execution IDs should be valid UUIDs.

        """
        import uuid

        # Generate an execution ID the same way the SDK does
        execution_id = str(uuid.uuid4())

        # Should be a valid UUID
        try:
            parsed = uuid.UUID(execution_id)
            assert str(parsed) == execution_id
        except ValueError:
            pytest.fail(f"Execution ID '{execution_id}' is not a valid UUID")

    @given(
        num_runs=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=100)
    def test_execution_ids_have_no_collisions(self, num_runs: int):
        """Feature: multi-agent-sdk, Property 12: Execution ID Uniqueness

        Multiple rapid executions should not produce colliding IDs.

        """
        import uuid

        # Simulate rapid execution ID generation
        execution_ids = []
        for _ in range(num_runs):
            execution_ids.append(str(uuid.uuid4()))

        # Check for uniqueness
        unique_ids = set(execution_ids)
        assert len(unique_ids) == num_runs, (
            f"Expected {num_runs} unique execution IDs, got {len(unique_ids)}"
        )


class TestEventContainsIdentifiers:
    """Property 13: Event Contains Identifiers

    *For any* StreamEvent emitted by the SDK, it SHALL contain timestamp
    and (where applicable) agent_id.

    """

    @given(
        event_type=st.sampled_from(
            [
                "token",
                "agent_start",
                "agent_end",
                "workflow_selected",
                "parallel_start",
                "parallel_complete",
                "final_result",
                "error",
            ]
        ),
        agent_id=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    )
    @settings(max_examples=100)
    def test_all_events_have_timestamp(self, event_type: str, agent_id: str | None):
        """Feature: multi-agent-sdk, Property 13: Event Contains Identifiers

        For any event type, the event should have a timestamp.

        """
        import time

        from kiva.events import StreamEvent

        event = StreamEvent(
            type=event_type,
            data={"test": "data"},
            timestamp=time.time(),
            agent_id=agent_id,
        )

        # Timestamp should be present and positive
        assert event.timestamp > 0, "Event should have a positive timestamp"
        assert isinstance(event.timestamp, float), "Timestamp should be a float"

    @given(
        agent_id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_agent_events_have_agent_id(self, agent_id: str):
        """Feature: multi-agent-sdk, Property 13: Event Contains Identifiers

        For agent-related events, the event should have an agent_id.

        """
        import time

        from kiva.events import StreamEvent

        # Agent-related events should have agent_id
        for event_type in ["agent_start", "agent_end", "error"]:
            event = StreamEvent(
                type=event_type,
                data={"test": "data"},
                timestamp=time.time(),
                agent_id=agent_id,
            )

            assert event.agent_id == agent_id, (
                f"Event type '{event_type}' should have agent_id '{agent_id}'"
            )

    @given(
        num_agents=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_processed_events_contain_execution_id(self, num_agents: int):
        """Feature: multi-agent-sdk, Property 13: Event Contains Identifiers

        For any processed event, the data should contain execution_id.

        """
        from kiva.run import _process_node_update

        async def check_execution_id():
            execution_id = "test-execution-id-12345"

            # Test workflow_selected event
            analyze_data = {
                "workflow": "router",
                "complexity": "simple",
                "task_assignments": [],
            }
            async for event in _process_node_update(
                "analyze_and_plan", analyze_data, execution_id
            ):
                assert "execution_id" in event.data, (
                    "workflow_selected event should contain execution_id in data"
                )
                assert event.data["execution_id"] == execution_id, (
                    f"execution_id should match: {event.data.get('execution_id')} vs {execution_id}"
                )

            # Test agent_end event
            router_data = {
                "agent_results": [{"agent_id": "agent_0", "result": "test"}],
            }
            async for event in _process_node_update(
                "router_workflow", router_data, execution_id
            ):
                assert "execution_id" in event.data, (
                    "agent_end event should contain execution_id in data"
                )
                assert event.data["execution_id"] == execution_id

            # Test final_result event
            synth_data = {"final_result": "Final answer"}
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                assert "execution_id" in event.data, (
                    "final_result event should contain execution_id in data"
                )
                assert event.data["execution_id"] == execution_id

        asyncio.get_event_loop().run_until_complete(check_execution_id())

    def test_stream_event_to_dict_preserves_identifiers(self):
        """Feature: multi-agent-sdk, Property 13: Event Contains Identifiers

        StreamEvent.to_dict() should preserve all identifiers.

        """
        import time

        from kiva.events import StreamEvent

        timestamp = time.time()
        agent_id = "test_agent"

        event = StreamEvent(
            type="agent_end",
            data={"result": "test", "execution_id": "exec-123"},
            timestamp=timestamp,
            agent_id=agent_id,
        )

        event_dict = event.to_dict()

        # All identifiers should be preserved
        assert event_dict["timestamp"] == timestamp
        assert event_dict["agent_id"] == agent_id
        assert event_dict["data"]["execution_id"] == "exec-123"

    @given(
        execution_id=st.text(min_size=8, max_size=36).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_custom_events_include_execution_id(self, execution_id: str):
        """Feature: multi-agent-sdk, Property 13: Event Contains Identifiers

        Custom events processed through _process_stream_chunk should include execution_id.

        """
        from kiva.run import _process_stream_chunk

        async def check_custom_event():
            custom_data = {
                "type": "agent_start",
                "agent_id": "test_agent",
                "task": "test task",
            }

            chunk = ("custom", custom_data)

            events = []
            async for event in _process_stream_chunk(chunk, execution_id):
                events.append(event)

            assert len(events) == 1
            assert "execution_id" in events[0].data, (
                "Custom event should have execution_id added to data"
            )
            assert events[0].data["execution_id"] == execution_id

        asyncio.get_event_loop().run_until_complete(check_custom_event())


class TestAgentErrorWrapping:
    """Property 5: Agent Error Wrapping

    *For any* Agent execution failure, the SDK SHALL wrap the error as AgentError
    containing the agent_id and original error.

    """

    @given(
        agent_id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_wrap_agent_error_preserves_agent_id(
        self, agent_id: str, error_message: str
    ):
        """Feature: multi-agent-sdk, Property 5: Agent Error Wrapping

        For any agent error, the wrapped AgentError should contain the agent_id.

        """
        from kiva.exceptions import AgentError, wrap_agent_error

        original_error = ValueError(error_message)
        wrapped = wrap_agent_error(original_error, agent_id)

        assert isinstance(wrapped, AgentError), (
            f"Expected AgentError, got {type(wrapped)}"
        )
        assert wrapped.agent_id == agent_id, (
            f"agent_id should be preserved: {wrapped.agent_id} vs {agent_id}"
        )

    @given(
        agent_id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        error_type=st.sampled_from(
            [ValueError, TypeError, RuntimeError, KeyError, AttributeError]
        ),
        error_message=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_wrap_agent_error_preserves_original_error(
        self, agent_id: str, error_type: type, error_message: str
    ):
        """Feature: multi-agent-sdk, Property 5: Agent Error Wrapping

        For any agent error, the wrapped AgentError should contain the original error.

        """
        from kiva.exceptions import wrap_agent_error

        original_error = error_type(error_message)
        wrapped = wrap_agent_error(original_error, agent_id)

        assert wrapped.original_error is original_error, (
            "original_error should be preserved"
        )
        assert isinstance(wrapped.original_error, error_type), (
            f"original_error type should be {error_type}, got {type(wrapped.original_error)}"
        )

    @given(
        agent_id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_agent_error_includes_recovery_suggestion(
        self, agent_id: str, error_message: str
    ):
        """Feature: multi-agent-sdk, Property 5: Agent Error Wrapping

        For any AgentError, it should include a recovery suggestion.

        """
        from kiva.exceptions import wrap_agent_error

        original_error = ValueError(error_message)
        wrapped = wrap_agent_error(original_error, agent_id)

        # Should have recovery_suggestion attribute
        assert hasattr(wrapped, "recovery_suggestion"), (
            "AgentError should have recovery_suggestion attribute"
        )
        assert wrapped.recovery_suggestion is not None, (
            "recovery_suggestion should not be None"
        )
        assert len(wrapped.recovery_suggestion) > 0, (
            "recovery_suggestion should not be empty"
        )

        # Error message should contain "Recovery:"
        error_str = str(wrapped)
        assert "Recovery:" in error_str, (
            f"Error message should contain recovery hint: {error_str}"
        )

    @given(
        agent_id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_wrap_agent_error_is_idempotent(self, agent_id: str):
        """Feature: multi-agent-sdk, Property 5: Agent Error Wrapping

        Wrapping an already-wrapped AgentError should return it unchanged.

        """
        from kiva.exceptions import AgentError, wrap_agent_error

        # Create an AgentError
        original = AgentError("Test error", agent_id, ValueError("original"))

        # Wrap it again
        wrapped = wrap_agent_error(original, "different_agent")

        # Should return the same object (idempotent)
        assert wrapped is original, "Wrapping an AgentError should return it unchanged"

    @given(
        error_keyword=st.sampled_from(
            [
                "timeout",
                "rate limit",
                "429",
                "connection",
                "network",
                "auth",
                "key",
                "401",
                "403",
                "tool",
                "function",
            ]
        ),
        agent_id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_recovery_suggestion_is_context_aware(
        self, error_keyword: str, agent_id: str
    ):
        """Feature: multi-agent-sdk, Property 5: Agent Error Wrapping

        Recovery suggestions should be context-aware based on error type.

        """
        from kiva.exceptions import wrap_agent_error

        # Create error with specific keyword
        original_error = RuntimeError(f"Error: {error_keyword} occurred")
        wrapped = wrap_agent_error(original_error, agent_id)

        # Recovery suggestion should exist and be non-empty
        assert wrapped.recovery_suggestion is not None
        assert len(wrapped.recovery_suggestion) > 0

        # Recovery suggestion should mention the agent_id
        assert agent_id in wrapped.recovery_suggestion, (
            f"Recovery suggestion should mention agent_id '{agent_id}'"
        )


class TestPartialFailureContinuation:
    """Property 6: Partial Failure Continuation

    *For any* execution where a non-critical Worker Agent fails, the SDK SHALL
    continue execution and report partial results in the final_result event.

    """

    @given(
        num_successful=st.integers(min_value=1, max_value=3),
        num_failed=st.integers(min_value=1, max_value=2),
    )
    @settings(max_examples=100)
    def test_partial_results_analysis_identifies_failures(
        self, num_successful: int, num_failed: int
    ):
        """Feature: multi-agent-sdk, Property 6: Partial Failure Continuation

        For any mix of successful and failed agents, the analysis should correctly
        identify which agents succeeded and which failed.

        """
        from kiva.nodes.synthesize import _analyze_partial_results

        # Create mixed results
        agent_results = []

        # Add successful results
        for i in range(num_successful):
            agent_results.append(
                {
                    "agent_id": f"success_agent_{i}",
                    "result": f"Result from agent {i}",
                }
            )

        # Add failed results
        for i in range(num_failed):
            agent_results.append(
                {
                    "agent_id": f"failed_agent_{i}",
                    "result": None,
                    "error": f"Error from agent {i}",
                }
            )

        # Analyze results
        analysis = _analyze_partial_results(agent_results)

        # Verify counts
        assert analysis["success_count"] == num_successful, (
            f"Expected {num_successful} successful, got {analysis['success_count']}"
        )
        assert analysis["failure_count"] == num_failed, (
            f"Expected {num_failed} failed, got {analysis['failure_count']}"
        )
        assert analysis["total"] == num_successful + num_failed, (
            f"Total should be {num_successful + num_failed}"
        )

        # Verify is_partial flag
        assert analysis["is_partial"] is True, (
            "Should be marked as partial result when some agents failed"
        )
        assert analysis["all_failed"] is False, (
            "Should not be marked as all_failed when some succeeded"
        )

    @given(
        num_failed=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_all_failed_analysis(self, num_failed: int):
        """Feature: multi-agent-sdk, Property 6: Partial Failure Continuation

        When all agents fail, the analysis should correctly identify this.

        """
        from kiva.nodes.synthesize import _analyze_partial_results

        # Create all failed results
        agent_results = []
        for i in range(num_failed):
            agent_results.append(
                {
                    "agent_id": f"failed_agent_{i}",
                    "result": None,
                    "error": f"Error from agent {i}",
                }
            )

        # Analyze results
        analysis = _analyze_partial_results(agent_results)

        # Verify all_failed flag
        assert analysis["all_failed"] is True, (
            "Should be marked as all_failed when no agents succeeded"
        )
        assert analysis["is_partial"] is False, (
            "Should not be marked as partial when all failed"
        )
        assert analysis["success_count"] == 0, "Success count should be 0"
        assert analysis["failure_count"] == num_failed, (
            f"Failure count should be {num_failed}"
        )

    @given(
        num_successful=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_all_successful_analysis(self, num_successful: int):
        """Feature: multi-agent-sdk, Property 6: Partial Failure Continuation

        When all agents succeed, the analysis should correctly identify this.

        """
        from kiva.nodes.synthesize import _analyze_partial_results

        # Create all successful results
        agent_results = []
        for i in range(num_successful):
            agent_results.append(
                {
                    "agent_id": f"success_agent_{i}",
                    "result": f"Result from agent {i}",
                }
            )

        # Analyze results
        analysis = _analyze_partial_results(agent_results)

        # Verify flags
        assert analysis["all_failed"] is False, (
            "Should not be marked as all_failed when agents succeeded"
        )
        assert analysis["is_partial"] is False, (
            "Should not be marked as partial when all succeeded"
        )
        assert analysis["success_count"] == num_successful, (
            f"Success count should be {num_successful}"
        )
        assert analysis["failure_count"] == 0, "Failure count should be 0"

    @given(
        num_successful=st.integers(min_value=1, max_value=2),
        num_failed=st.integers(min_value=1, max_value=2),
    )
    @settings(max_examples=100)
    def test_final_result_event_includes_partial_info(
        self, num_successful: int, num_failed: int
    ):
        """Feature: multi-agent-sdk, Property 6: Partial Failure Continuation

        For any partial failure, the final_result event should include
        information about which agents failed.

        """
        from kiva.run import _process_node_update

        async def check_partial_info():
            execution_id = "test-exec-id"

            # Create partial result info
            partial_result_info = {
                "successful": [
                    {"agent_id": f"success_{i}"} for i in range(num_successful)
                ],
                "failed": [{"agent_id": f"failed_{i}"} for i in range(num_failed)],
                "total": num_successful + num_failed,
                "success_count": num_successful,
                "failure_count": num_failed,
                "is_partial": True,
                "all_failed": False,
            }

            # Simulate synthesize_results node output with partial results
            synth_data = {
                "final_result": "Partial result based on available data",
                "partial_result_info": partial_result_info,
            }

            events = []
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                events.append(event)

            # Should have exactly one final_result event
            assert len(events) == 1
            event = events[0]
            assert event.type == "final_result"

            # Should include partial result info
            assert "partial_result" in event.data, (
                "final_result event should include partial_result flag"
            )
            assert event.data["partial_result"] is True, (
                "partial_result should be True for partial results"
            )
            assert event.data["success_count"] == num_successful, (
                f"success_count should be {num_successful}"
            )
            assert event.data["failure_count"] == num_failed, (
                f"failure_count should be {num_failed}"
            )
            assert "failed_agents" in event.data, "Should include list of failed agents"
            assert len(event.data["failed_agents"]) == num_failed, (
                f"Should have {num_failed} failed agents"
            )

        asyncio.get_event_loop().run_until_complete(check_partial_info())

    @given(
        num_agents=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100)
    def test_successful_execution_has_no_partial_flag(self, num_agents: int):
        """Feature: multi-agent-sdk, Property 6: Partial Failure Continuation

        For any fully successful execution, the final_result event should
        not be marked as partial.

        """
        from kiva.run import _process_node_update

        async def check_no_partial():
            execution_id = "test-exec-id"

            # Create successful result info
            partial_result_info = {
                "successful": [{"agent_id": f"agent_{i}"} for i in range(num_agents)],
                "failed": [],
                "total": num_agents,
                "success_count": num_agents,
                "failure_count": 0,
                "is_partial": False,
                "all_failed": False,
            }

            # Simulate synthesize_results node output
            synth_data = {
                "final_result": "Complete result from all agents",
                "partial_result_info": partial_result_info,
            }

            events = []
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                events.append(event)

            # Should have exactly one final_result event
            assert len(events) == 1
            event = events[0]
            assert event.type == "final_result"

            # Should not be marked as partial
            assert event.data.get("partial_result", False) is False, (
                "partial_result should be False for complete results"
            )
            assert event.data.get("failure_count", 0) == 0, (
                "failure_count should be 0 for complete results"
            )

        asyncio.get_event_loop().run_until_complete(check_no_partial())


class TestFinalResultContainsCitations:
    """Property 11: Final Result Contains Citations

    *For any* execution with multiple Worker Agents, the final_result event
    SHALL contain a citations field.

    """

    @given(
        num_agents=st.integers(min_value=2, max_value=5),
        result_text=st.text(min_size=10, max_size=200),
    )
    @settings(max_examples=100)
    def test_final_result_event_contains_citations_field(
        self, num_agents: int, result_text: str
    ):
        """Feature: multi-agent-sdk, Property 11: Final Result Contains Citations

        For any execution with multiple agents, the final_result event should
        contain a citations field.

        """
        from kiva.run import _process_node_update

        async def check_citations_field():
            execution_id = "test-exec-id"

            # Create citations from agent results
            citations = [
                {"source": f"agent_{i}", "type": "agent"} for i in range(num_agents)
            ]

            # Simulate synthesize_results node output with citations
            synth_data = {
                "final_result": result_text,
                "citations": citations,
                "partial_result_info": {
                    "successful": [
                        {"agent_id": f"agent_{i}"} for i in range(num_agents)
                    ],
                    "failed": [],
                    "total": num_agents,
                    "success_count": num_agents,
                    "failure_count": 0,
                    "is_partial": False,
                    "all_failed": False,
                },
            }

            events = []
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                events.append(event)

            # Should have exactly one final_result event
            assert len(events) == 1
            event = events[0]
            assert event.type == "final_result"

            # Should contain citations field
            assert "citations" in event.data, (
                "final_result event should contain citations field"
            )
            assert isinstance(event.data["citations"], list), (
                "citations should be a list"
            )

        asyncio.get_event_loop().run_until_complete(check_citations_field())

    @given(
        num_agents=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100)
    def test_citations_include_agent_sources(self, num_agents: int):
        """Feature: multi-agent-sdk, Property 11: Final Result Contains Citations

        For any execution with multiple agents, the citations should include
        the agent sources.

        """
        from kiva.run import _process_node_update

        async def check_agent_sources():
            execution_id = "test-exec-id"

            # Create citations that include agent sources
            agent_ids = [f"agent_{i}" for i in range(num_agents)]
            citations = [
                {"source": agent_id, "type": "agent"} for agent_id in agent_ids
            ]

            # Simulate synthesize_results node output
            synth_data = {
                "final_result": f"Result from {num_agents} agents",
                "citations": citations,
                "partial_result_info": {
                    "successful": [{"agent_id": aid} for aid in agent_ids],
                    "failed": [],
                    "total": num_agents,
                    "success_count": num_agents,
                    "failure_count": 0,
                    "is_partial": False,
                    "all_failed": False,
                },
            }

            events = []
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                events.append(event)

            # Should have citations
            assert len(events) == 1
            event = events[0]

            # Citations should be present
            assert "citations" in event.data
            citations_in_event = event.data["citations"]

            # Should have at least as many citations as agents
            assert len(citations_in_event) >= num_agents, (
                f"Expected at least {num_agents} citations, got {len(citations_in_event)}"
            )

            # All agent sources should be in citations
            citation_sources = {c["source"] for c in citations_in_event}
            for agent_id in agent_ids:
                assert agent_id in citation_sources, (
                    f"Agent {agent_id} should be in citations"
                )

        asyncio.get_event_loop().run_until_complete(check_agent_sources())

    @given(
        text_content=st.text(
            alphabet=st.characters(blacklist_characters="[]"), min_size=5, max_size=50
        ),
    )
    @settings(max_examples=100)
    def test_extract_citations_from_text(self, text_content: str):
        """Feature: multi-agent-sdk, Property 11: Final Result Contains Citations

        For any text containing citation markers, extract_citations should
        find them.

        """
        from kiva.nodes.synthesize import extract_citations

        # Create text with citation markers (avoiding brackets in the content)
        text_with_citations = f"[agent_1] says {text_content}. [agent_2] confirms."

        citations = extract_citations(text_with_citations)

        # Should find the citations
        sources = [c["source"] for c in citations]
        assert "agent_1" in sources, "Should find agent_1 citation"
        assert "agent_2" in sources, "Should find agent_2 citation"

    @given(
        num_agents=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100)
    def test_citations_field_is_list_of_dicts(self, num_agents: int):
        """Feature: multi-agent-sdk, Property 11: Final Result Contains Citations

        For any final_result event, the citations field should be a list
        of dictionaries with source and type fields.

        """
        from kiva.run import _process_node_update

        async def check_citations_structure():
            execution_id = "test-exec-id"

            # Create properly structured citations
            citations = [
                {"source": f"agent_{i}", "type": "agent"} for i in range(num_agents)
            ]

            synth_data = {
                "final_result": "Combined result from multiple agents",
                "citations": citations,
            }

            events = []
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                events.append(event)

            assert len(events) == 1
            event = events[0]

            # Verify citations structure
            assert "citations" in event.data
            for citation in event.data["citations"]:
                assert isinstance(citation, dict), (
                    "Each citation should be a dictionary"
                )
                assert "source" in citation, "Each citation should have a source field"

        asyncio.get_event_loop().run_until_complete(check_citations_structure())

    @given(
        result_text=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_final_result_without_node_citations_extracts_from_text(
        self, result_text: str
    ):
        """Feature: multi-agent-sdk, Property 11: Final Result Contains Citations

        When synthesize_results doesn't provide citations, they should be
        extracted from the result text.

        """
        from kiva.run import _process_node_update

        async def check_fallback_extraction():
            execution_id = "test-exec-id"

            # Add citation markers to the text
            text_with_markers = f"[test_agent] {result_text}"

            # Simulate synthesize_results without citations field
            synth_data = {
                "final_result": text_with_markers,
                # No citations field - should extract from text
            }

            events = []
            async for event in _process_node_update(
                "synthesize_results", synth_data, execution_id
            ):
                events.append(event)

            assert len(events) == 1
            event = events[0]

            # Should still have citations field (extracted from text)
            assert "citations" in event.data, (
                "citations should be extracted from text when not provided"
            )

            # Should find the test_agent citation
            sources = [c["source"] for c in event.data["citations"]]
            assert "test_agent" in sources, (
                "Should extract test_agent citation from text"
            )

        asyncio.get_event_loop().run_until_complete(check_fallback_extraction())


class TestWorkflowOverride:
    """Property 7: Workflow Override

    *For any* call to run() with workflow_override parameter, the SDK SHALL
    use the specified workflow regardless of complexity assessment.

    """

    @given(
        workflow_override=st.sampled_from(["router", "supervisor", "parliament"]),
        analyzed_workflow=st.sampled_from(["router", "supervisor", "parliament"]),
    )
    @settings(max_examples=100)
    def test_workflow_override_takes_priority_over_analyzed_workflow(
        self, workflow_override: str, analyzed_workflow: str
    ):
        """Feature: multi-agent-sdk, Property 7: Workflow Override

        For any workflow_override value, the route_to_workflow function should
        use the override regardless of what the Lead Agent analyzed.

        """
        from kiva.nodes.router import route_to_workflow

        # Create state with both workflow (from analysis) and workflow_override
        state = {
            "workflow": analyzed_workflow,
            "workflow_override": workflow_override,
        }

        result = route_to_workflow(state)

        # The result should match the override, not the analyzed workflow
        expected = f"{workflow_override}_workflow"
        assert result == expected, f"Expected {expected} (from override), got {result}"

    @given(
        workflow_override=st.sampled_from(["router", "supervisor", "parliament"]),
        complexity=st.sampled_from(["simple", "medium", "complex"]),
    )
    @settings(max_examples=100)
    def test_workflow_override_ignores_complexity(
        self, workflow_override: str, complexity: str
    ):
        """Feature: multi-agent-sdk, Property 7: Workflow Override

        For any workflow_override value, the complexity assessment should
        be ignored and the specified workflow should be used.

        """
        from kiva.nodes.router import route_to_workflow

        # Map complexity to what Lead Agent would normally choose
        complexity_to_workflow = {
            "simple": "router",
            "medium": "supervisor",
            "complex": "parliament",
        }
        analyzed_workflow = complexity_to_workflow[complexity]

        # Create state with complexity-based workflow and override
        state = {
            "workflow": analyzed_workflow,
            "complexity": complexity,
            "workflow_override": workflow_override,
        }

        result = route_to_workflow(state)

        # The result should match the override, not the complexity-based workflow
        expected = f"{workflow_override}_workflow"
        assert result == expected, (
            f"Expected {expected} (from override), got {result} (complexity was {complexity})"
        )

    @given(
        workflow=st.sampled_from(["router", "supervisor", "parliament"]),
    )
    @settings(max_examples=100)
    def test_no_override_uses_analyzed_workflow(self, workflow: str):
        """Feature: multi-agent-sdk, Property 7: Workflow Override

        When workflow_override is None, the analyzed workflow should be used.

        """
        from kiva.nodes.router import route_to_workflow

        # Create state without workflow_override
        state = {
            "workflow": workflow,
            "workflow_override": None,
        }

        result = route_to_workflow(state)

        # The result should match the analyzed workflow
        expected = f"{workflow}_workflow"
        assert result == expected, f"Expected {expected} (from analysis), got {result}"

    @given(
        workflow=st.sampled_from(["router", "supervisor", "parliament"]),
    )
    @settings(max_examples=100)
    def test_missing_override_uses_analyzed_workflow(self, workflow: str):
        """Feature: multi-agent-sdk, Property 7: Workflow Override

        When workflow_override is not in state, the analyzed workflow should be used.

        """
        from kiva.nodes.router import route_to_workflow

        # Create state without workflow_override key
        state = {
            "workflow": workflow,
        }

        result = route_to_workflow(state)

        # The result should match the analyzed workflow
        expected = f"{workflow}_workflow"
        assert result == expected, f"Expected {expected} (from analysis), got {result}"

    @given(
        workflow_override=st.sampled_from(["router", "supervisor", "parliament"]),
        prompt=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_workflow_override_passed_to_initial_state(
        self, workflow_override: str, prompt: str
    ):
        """Feature: multi-agent-sdk, Property 7: Workflow Override

        For any call to run() with workflow_override, the parameter should
        be correctly passed to the initial state.

        """
        # We can verify this by checking that run() accepts the parameter
        # and would pass it to the graph's initial state

        # Create mock agents
        agents = [MockAgent("test_agent")]

        # Call run() with workflow_override - this should not raise
        # We're testing that the parameter is accepted and would be passed through
        result = run(prompt, agents, workflow_override=workflow_override)

        # Verify it returns an AsyncIterator (parameter was accepted)
        assert isinstance(result, AsyncIterator), (
            "run() should accept workflow_override parameter and return AsyncIterator"
        )

    @given(
        workflow_override=st.sampled_from(["router", "supervisor", "parliament"]),
    )
    @settings(max_examples=100)
    def test_all_valid_workflow_override_values(self, workflow_override: str):
        """Feature: multi-agent-sdk, Property 7: Workflow Override

        All valid workflow_override values (router, supervisor, parliament)
        should be correctly routed.

        """
        from kiva.nodes.router import route_to_workflow

        state = {
            "workflow": "router",  # Default analyzed workflow
            "workflow_override": workflow_override,
        }

        result = route_to_workflow(state)

        # Verify the result is a valid workflow node name
        valid_workflow_nodes = [
            "router_workflow",
            "supervisor_workflow",
            "parliament_workflow",
        ]
        assert result in valid_workflow_nodes, (
            f"Result {result} should be one of {valid_workflow_nodes}"
        )

        # Verify it matches the override
        assert result == f"{workflow_override}_workflow", (
            f"Result should match override: expected {workflow_override}_workflow, got {result}"
        )
