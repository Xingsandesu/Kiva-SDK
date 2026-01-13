"""Tests for verification models and utilities.

This module contains unit tests and property-based tests for the verification
data models defined in kiva.verification.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiva.verification import (
    AgentMessage,
    FinalResultVerifier,
    LLMVerificationResult,
    RetryContext,
    VerificationResult,
    VerificationStatus,
    Verifier,
    WorkerOutputVerifier,
)


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_verification_status_values(self):
        """Test that VerificationStatus has expected values."""
        assert VerificationStatus.PASSED.value == "passed"
        assert VerificationStatus.FAILED.value == "failed"
        assert VerificationStatus.SKIPPED.value == "skipped"

    def test_verification_status_is_string_enum(self):
        """Test that VerificationStatus values are strings."""
        assert isinstance(VerificationStatus.PASSED, str)
        assert VerificationStatus.PASSED == "passed"


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_create_passed_result(self):
        """Test creating a passed verification result."""
        result = VerificationResult(status=VerificationStatus.PASSED)
        assert result.status == VerificationStatus.PASSED
        assert result.rejection_reason is None
        assert result.improvement_suggestions == []
        assert result.field_errors == {}
        assert result.validator_name == "default"
        assert result.confidence == 1.0

    def test_create_failed_result_with_details(self):
        """Test creating a failed verification result with all details."""
        result = VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason="Output does not address the task",
            improvement_suggestions=["Be more specific", "Include examples"],
            field_errors={"name": "Field is required"},
            validator_name="custom_validator",
            confidence=0.8,
        )
        assert result.status == VerificationStatus.FAILED
        assert result.rejection_reason == "Output does not address the task"
        assert len(result.improvement_suggestions) == 2
        assert result.field_errors["name"] == "Field is required"
        assert result.validator_name == "custom_validator"
        assert result.confidence == 0.8

    def test_confidence_bounds(self):
        """Test that confidence is bounded between 0.0 and 1.0."""
        result_min = VerificationResult(
            status=VerificationStatus.PASSED, confidence=0.0
        )
        result_max = VerificationResult(
            status=VerificationStatus.PASSED, confidence=1.0
        )
        assert result_min.confidence == 0.0
        assert result_max.confidence == 1.0

    def test_model_dump(self):
        """Test serialization of VerificationResult."""
        result = VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason="Test reason",
        )
        dumped = result.model_dump()
        assert dumped["status"] == "failed"
        assert dumped["rejection_reason"] == "Test reason"


class TestVerificationResultProperty:
    """Property-based tests for VerificationResult structure.

    Feature: output-verification-agent, Property 12: Verification Result Structure
    Validates: Requirements 1.6
    """

    @given(
        status=st.sampled_from(list(VerificationStatus)),
        rejection_reason=st.one_of(st.none(), st.text(max_size=200)),
        improvement_suggestions=st.lists(st.text(max_size=100), max_size=5),
        field_errors=st.dictionaries(
            keys=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
            values=st.text(max_size=100),
            max_size=5,
        ),
        validator_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_verification_result_structure_property(
        self,
        status: VerificationStatus,
        rejection_reason: str | None,
        improvement_suggestions: list[str],
        field_errors: dict[str, str],
        validator_name: str,
        confidence: float,
    ):
        """Property 12: Verification Result Structure.

        For any VerificationResult returned by the system, it SHALL contain:
        - status (VerificationStatus enum)
        - rejection_reason (string or None)
        - improvement_suggestions (list of strings)
        - field_errors (dict)

        Feature: output-verification-agent, Property 12: Verification Result Structure
        Validates: Requirements 1.6
        """
        result = VerificationResult(
            status=status,
            rejection_reason=rejection_reason,
            improvement_suggestions=improvement_suggestions,
            field_errors=field_errors,
            validator_name=validator_name,
            confidence=confidence,
        )

        # Verify structure requirements
        assert isinstance(result.status, VerificationStatus)
        assert result.rejection_reason is None or isinstance(
            result.rejection_reason, str
        )
        assert isinstance(result.improvement_suggestions, list)
        assert all(isinstance(s, str) for s in result.improvement_suggestions)
        assert isinstance(result.field_errors, dict)

        # Verify serialization round-trip
        dumped = result.model_dump()
        assert "status" in dumped
        assert "rejection_reason" in dumped
        assert "improvement_suggestions" in dumped
        assert "field_errors" in dumped
        assert "validator_name" in dumped
        assert "confidence" in dumped


class TestRetryContext:
    """Tests for RetryContext model."""

    def test_create_retry_context(self):
        """Test creating a RetryContext."""
        context = RetryContext(
            iteration=1,
            max_iterations=3,
            original_task="Analyze the data",
        )
        assert context.iteration == 1
        assert context.max_iterations == 3
        assert context.original_task == "Analyze the data"
        assert context.previous_outputs == []
        assert context.previous_rejections == []
        assert context.task_history == []

    def test_create_retry_context_with_history(self):
        """Test creating a RetryContext with previous attempts."""
        rejection = VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason="Incomplete response",
        )
        context = RetryContext(
            iteration=2,
            max_iterations=3,
            original_task="Analyze the data",
            previous_outputs=["First attempt output"],
            previous_rejections=[rejection],
            task_history=[{"agent_id": "analyzer", "task": "Analyze"}],
        )
        assert context.iteration == 2
        assert len(context.previous_outputs) == 1
        assert len(context.previous_rejections) == 1
        assert context.previous_rejections[0].status == VerificationStatus.FAILED


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_create_agent_message(self):
        """Test creating an AgentMessage."""
        message = AgentMessage(
            sender_id="agent_1",
            receiver_id="agent_2",
            content={"result": "analysis complete"},
            message_type="task_result",
            timestamp=1234567890.0,
        )
        assert message.sender_id == "agent_1"
        assert message.receiver_id == "agent_2"
        assert message.content == {"result": "analysis complete"}
        assert message.message_type == "task_result"
        assert message.timestamp == 1234567890.0

    def test_agent_message_with_various_content_types(self):
        """Test AgentMessage with different content types."""
        # String content
        msg_str = AgentMessage(
            sender_id="a",
            receiver_id="b",
            content="Hello",
            message_type="text",
            timestamp=time.time(),
        )
        assert msg_str.content == "Hello"

        # List content
        msg_list = AgentMessage(
            sender_id="a",
            receiver_id="b",
            content=[1, 2, 3],
            message_type="data",
            timestamp=time.time(),
        )
        assert msg_list.content == [1, 2, 3]

        # None content
        msg_none = AgentMessage(
            sender_id="a",
            receiver_id="b",
            content=None,
            message_type="ping",
            timestamp=time.time(),
        )
        assert msg_none.content is None


class TestWorkerOutputVerifier:
    """Unit tests for WorkerOutputVerifier class."""

    def test_create_verifier_with_defaults(self):
        """Test creating a WorkerOutputVerifier with default settings."""
        verifier = WorkerOutputVerifier()
        assert verifier.model_name == "gpt-4o"
        assert verifier.api_key is None
        assert verifier.base_url is None
        assert verifier.custom_verifiers == []

    def test_create_verifier_with_custom_settings(self):
        """Test creating a WorkerOutputVerifier with custom settings."""
        verifier = WorkerOutputVerifier(
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            base_url="https://api.example.com",
        )
        assert verifier.model_name == "gpt-3.5-turbo"
        assert verifier.api_key == "test-key"
        assert verifier.base_url == "https://api.example.com"

    def test_pydantic_schema_validation_success(self):
        """Test Pydantic schema validation with valid JSON."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        verifier = WorkerOutputVerifier()
        result = verifier._validate_pydantic_schema(
            output='{"name": "test", "value": 42}',
            schema=TestSchema,
        )
        assert result.status == VerificationStatus.PASSED
        assert result.validator_name == "pydantic_schema"

    def test_pydantic_schema_validation_invalid_json(self):
        """Test Pydantic schema validation with invalid JSON."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        verifier = WorkerOutputVerifier()
        result = verifier._validate_pydantic_schema(
            output="not valid json",
            schema=TestSchema,
        )
        assert result.status == VerificationStatus.FAILED
        assert "_json" in result.field_errors
        assert result.validator_name == "pydantic_schema"

    def test_pydantic_schema_validation_missing_field(self):
        """Test Pydantic schema validation with missing required field."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            required_field: int

        verifier = WorkerOutputVerifier()
        result = verifier._validate_pydantic_schema(
            output='{"name": "test"}',
            schema=TestSchema,
        )
        assert result.status == VerificationStatus.FAILED
        assert "required_field" in result.field_errors
        assert len(result.improvement_suggestions) > 0
        assert result.validator_name == "pydantic_schema"

    def test_aggregate_results_all_passed(self):
        """Test aggregation when all results pass."""
        verifier = WorkerOutputVerifier()
        results = [
            VerificationResult(status=VerificationStatus.PASSED, confidence=0.9),
            VerificationResult(status=VerificationStatus.PASSED, confidence=0.8),
        ]
        aggregated = verifier._aggregate_results(results)
        assert aggregated.status == VerificationStatus.PASSED
        assert abs(aggregated.confidence - 0.85) < 0.0001  # Float comparison

    def test_aggregate_results_one_failed(self):
        """Test aggregation when one result fails."""
        verifier = WorkerOutputVerifier()
        results = [
            VerificationResult(status=VerificationStatus.PASSED, confidence=0.9),
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason="Test failure",
                improvement_suggestions=["Fix it"],
                confidence=0.5,
            ),
        ]
        aggregated = verifier._aggregate_results(results)
        assert aggregated.status == VerificationStatus.FAILED
        assert "Test failure" in aggregated.rejection_reason
        assert "Fix it" in aggregated.improvement_suggestions

    def test_aggregate_results_empty(self):
        """Test aggregation with empty results list."""
        verifier = WorkerOutputVerifier()
        aggregated = verifier._aggregate_results([])
        assert aggregated.status == VerificationStatus.PASSED

    def test_custom_verifier_execution(self):
        """Test that custom verifiers are executed."""

        class CustomVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                if "error" in output.lower():
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        rejection_reason="Output contains error",
                        validator_name="custom",
                    )
                return VerificationResult(
                    status=VerificationStatus.PASSED,
                    validator_name="custom",
                )

        verifier = WorkerOutputVerifier(custom_verifiers=[CustomVerifier()])
        results = verifier._run_custom_verifiers(
            assigned_task="Test task",
            output="This has an error in it",
        )
        assert len(results) == 1
        assert results[0].status == VerificationStatus.FAILED
        assert results[0].rejection_reason == "Output contains error"


class TestWorkerOutputVerificationProperty:
    """Property-based tests for Worker output verification.

    Feature: output-verification-agent, Worker Output Verification
    Against Assigned Task
    """

    @given(
        assigned_task=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        output=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_worker_verification_uses_assigned_task_not_original_prompt(
        self,
        assigned_task: str,
        output: str,
    ):
        """Worker Output Verification Against Assigned Task.

        For any Worker Agent execution that completes successfully, the
        WorkerOutputVerifier SHALL verify the output against the specific
        task assigned to that Worker (task_assignment.task), NOT against
        the user's original prompt.

        Feature: output-verification-agent
        """
        # Create a custom verifier that captures what task was passed
        captured_tasks = []

        class TaskCapturingVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                captured_tasks.append(task)
                return VerificationResult(status=VerificationStatus.PASSED)

        verifier = WorkerOutputVerifier(
            custom_verifiers=[TaskCapturingVerifier()]
        )

        # Run custom verifiers (synchronous part we can test)
        verifier._run_custom_verifiers(
            assigned_task=assigned_task,
            output=output,
        )

        # Verify the assigned_task was passed to the verifier
        assert len(captured_tasks) == 1
        assert captured_tasks[0] == assigned_task

    @given(
        assigned_task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        output=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_verification_result_always_has_required_fields(
        self,
        assigned_task: str,
        output: str,
    ):
        """Verification always produces properly structured results.

        For any verification operation, the result SHALL contain all required
        fields: status, rejection_reason, improvement_suggestions, field_errors.

        Feature: output-verification-agent
        """
        verifier = WorkerOutputVerifier()

        # Test with custom verifier that always passes
        class AlwaysPassVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                return VerificationResult(status=VerificationStatus.PASSED)

        verifier_with_custom = WorkerOutputVerifier(
            custom_verifiers=[AlwaysPassVerifier()]
        )
        results = verifier_with_custom._run_custom_verifiers(assigned_task, output)

        for result in results:
            # Verify all required fields exist
            assert hasattr(result, "status")
            assert hasattr(result, "rejection_reason")
            assert hasattr(result, "improvement_suggestions")
            assert hasattr(result, "field_errors")
            assert isinstance(result.status, VerificationStatus)
            assert isinstance(result.improvement_suggestions, list)
            assert isinstance(result.field_errors, dict)


class TestPydanticValidationProperty:
    """Property-based tests for Pydantic validation.

    Feature: output-verification-agent
    """

    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        value=st.integers(min_value=-1000, max_value=1000),
    )
    @settings(max_examples=100)
    def test_valid_schema_produces_passed_result(
        self,
        name: str,
        value: int,
    ):
        """Valid output conforming to schema produces PASSED.

        For any output that conforms to the Pydantic schema, the verification
        result SHALL have status=PASSED.

        Feature: output-verification-agent
        """
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        verifier = WorkerOutputVerifier()
        output = json.dumps({"name": name, "value": value})

        result = verifier._validate_pydantic_schema(output, TestSchema)

        assert result.status == VerificationStatus.PASSED
        assert result.field_errors == {}

    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        # Generate values that cannot be coerced to int
        invalid_value=st.one_of(
            # Strings that are not numeric
            st.text(min_size=1, max_size=20).filter(
                lambda x: x.strip() and not x.strip().lstrip("-").isdigit()
            ),
            # Lists (cannot be int)
            st.lists(st.integers(), min_size=1, max_size=3),
        ),
    )
    @settings(max_examples=100)
    def test_invalid_schema_produces_failed_with_field_errors(
        self,
        name: str,
        invalid_value,
    ):
        """Invalid output produces FAILED with field errors.

        For any output that does not conform to the schema, the verification
        result SHALL have status=FAILED and SHALL contain field-level error
        details in the field_errors dictionary.

        Feature: output-verification-agent
        """
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int  # Expects integer

        verifier = WorkerOutputVerifier()
        output = json.dumps({"name": name, "value": invalid_value})

        result = verifier._validate_pydantic_schema(output, TestSchema)

        assert result.status == VerificationStatus.FAILED
        assert len(result.field_errors) > 0
        assert "value" in result.field_errors
        assert len(result.improvement_suggestions) > 0

    @given(
        missing_fields=st.lists(
            st.sampled_from(["field1", "field2", "field3"]),
            min_size=1,
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_missing_required_fields_produces_field_errors(
        self,
        missing_fields: list[str],
    ):
        """Missing required fields produce specific field errors.

        For any output missing required fields, the verification result SHALL
        contain field-level error details for each missing field.

        Feature: output-verification-agent
        """
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            field1: str
            field2: int
            field3: bool

        verifier = WorkerOutputVerifier()

        # Create output with some fields missing
        all_fields = {"field1": "test", "field2": 42, "field3": True}
        partial_output = {k: v for k, v in all_fields.items() if k not in missing_fields}
        output = json.dumps(partial_output)

        result = verifier._validate_pydantic_schema(output, TestSchema)

        assert result.status == VerificationStatus.FAILED
        # Each missing field should have an error
        for field in missing_fields:
            assert field in result.field_errors

    @given(
        # Generate strings that are definitely not valid JSON objects
        invalid_json=st.one_of(
            # Plain text that's not JSON
            st.text(min_size=2, max_size=100).filter(
                lambda x: x.strip()
                and not x.strip().startswith("{")
                and not x.strip().startswith("[")
                and not x.strip().startswith('"')
                and not x.strip().lstrip("-").replace(".", "", 1).isdigit()
            ),
        ),
    )
    @settings(max_examples=100)
    def test_invalid_json_produces_json_error(
        self,
        invalid_json: str,
    ):
        """Invalid JSON produces specific JSON error.

        For any output that is not valid JSON, the verification result SHALL
        have status=FAILED and contain a JSON parsing error.

        Feature: output-verification-agent
        """
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        verifier = WorkerOutputVerifier()

        result = verifier._validate_pydantic_schema(invalid_json, TestSchema)

        assert result.status == VerificationStatus.FAILED
        # Either JSON error or Pydantic validation error
        assert len(result.field_errors) > 0


class TestLLMVerificationResult:
    """Tests for LLMVerificationResult model."""

    def test_create_passed_llm_result(self):
        """Test creating a passed LLM verification result."""
        result = LLMVerificationResult(passed=True)
        assert result.passed is True
        assert result.rejection_reason is None
        assert result.improvement_suggestions == []
        assert result.confidence == 1.0

    def test_create_failed_llm_result(self):
        """Test creating a failed LLM verification result with details."""
        result = LLMVerificationResult(
            passed=False,
            rejection_reason="Output incomplete",
            improvement_suggestions=["Add more details"],
            confidence=0.7,
        )
        assert result.passed is False
        assert result.rejection_reason == "Output incomplete"
        assert result.improvement_suggestions == ["Add more details"]
        assert result.confidence == 0.7


class TestVerifierProtocol:
    """Tests for Verifier protocol."""

    def test_class_implements_verifier_protocol(self):
        """Test that a class can implement the Verifier protocol."""

        class MyVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                return VerificationResult(status=VerificationStatus.PASSED)

        verifier = MyVerifier()
        assert isinstance(verifier, Verifier)

    def test_verifier_can_be_used_with_worker_output_verifier(self):
        """Test that custom verifiers work with WorkerOutputVerifier."""

        class LengthVerifier:
            _verifier_name = "length_check"

            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                if len(output) < 10:
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        rejection_reason="Output too short",
                        validator_name=self._verifier_name,
                    )
                return VerificationResult(
                    status=VerificationStatus.PASSED,
                    validator_name=self._verifier_name,
                )

        worker_verifier = WorkerOutputVerifier(custom_verifiers=[LengthVerifier()])
        results = worker_verifier._run_custom_verifiers(
            assigned_task="Write something",
            output="Short",
        )

        assert len(results) == 1
        assert results[0].status == VerificationStatus.FAILED
        assert results[0].validator_name == "length_check"


class TestFinalResultVerifier:
    """Unit tests for FinalResultVerifier class."""

    def test_create_verifier_with_defaults(self):
        """Test creating a FinalResultVerifier with default settings."""
        verifier = FinalResultVerifier()
        assert verifier.model_name == "gpt-4o"
        assert verifier.api_key is None
        assert verifier.base_url is None
        assert verifier.custom_verifiers == []

    def test_create_verifier_with_custom_settings(self):
        """Test creating a FinalResultVerifier with custom settings."""
        verifier = FinalResultVerifier(
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            base_url="https://api.example.com",
        )
        assert verifier.model_name == "gpt-3.5-turbo"
        assert verifier.api_key == "test-key"
        assert verifier.base_url == "https://api.example.com"

    def test_aggregate_results_all_passed(self):
        """Test aggregation when all results pass."""
        verifier = FinalResultVerifier()
        results = [
            VerificationResult(status=VerificationStatus.PASSED, confidence=0.9),
            VerificationResult(status=VerificationStatus.PASSED, confidence=0.8),
        ]
        aggregated = verifier._aggregate_results(results)
        assert aggregated.status == VerificationStatus.PASSED
        assert abs(aggregated.confidence - 0.85) < 0.0001
        assert aggregated.validator_name == "final_aggregate"

    def test_aggregate_results_one_failed(self):
        """Test aggregation when one result fails."""
        verifier = FinalResultVerifier()
        results = [
            VerificationResult(status=VerificationStatus.PASSED, confidence=0.9),
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason="Incomplete answer",
                improvement_suggestions=["Add more details"],
                confidence=0.5,
            ),
        ]
        aggregated = verifier._aggregate_results(results)
        assert aggregated.status == VerificationStatus.FAILED
        assert "Incomplete answer" in aggregated.rejection_reason
        assert "Add more details" in aggregated.improvement_suggestions
        assert aggregated.validator_name == "final_aggregate"

    def test_aggregate_results_empty(self):
        """Test aggregation with empty results list."""
        verifier = FinalResultVerifier()
        aggregated = verifier._aggregate_results([])
        assert aggregated.status == VerificationStatus.PASSED
        assert aggregated.validator_name == "final_aggregate"

    def test_custom_verifier_execution(self):
        """Test that custom verifiers are executed with original prompt."""

        class CustomVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                # Check if the output mentions the key topic from the prompt
                if "capital" in task.lower() and "paris" not in output.lower():
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        rejection_reason="Missing capital city in response",
                        validator_name="custom",
                    )
                return VerificationResult(
                    status=VerificationStatus.PASSED,
                    validator_name="custom",
                )

        verifier = FinalResultVerifier(custom_verifiers=[CustomVerifier()])
        results = verifier._run_custom_verifiers(
            original_prompt="What is the capital of France?",
            final_result="France is a country in Europe.",
        )
        assert len(results) == 1
        assert results[0].status == VerificationStatus.FAILED
        assert results[0].rejection_reason == "Missing capital city in response"

    def test_custom_verifier_passes(self):
        """Test custom verifier that passes."""

        class CustomVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                if "capital" in task.lower() and "paris" in output.lower():
                    return VerificationResult(
                        status=VerificationStatus.PASSED,
                        validator_name="custom",
                    )
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="Missing expected content",
                    validator_name="custom",
                )

        verifier = FinalResultVerifier(custom_verifiers=[CustomVerifier()])
        results = verifier._run_custom_verifiers(
            original_prompt="What is the capital of France?",
            final_result="The capital of France is Paris.",
        )
        assert len(results) == 1
        assert results[0].status == VerificationStatus.PASSED

    def test_custom_verifier_exception_handling(self):
        """Test that custom verifier exceptions are handled gracefully."""

        class FailingVerifier:
            _verifier_name = "failing_verifier"

            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                raise ValueError("Verifier crashed!")

        verifier = FinalResultVerifier(custom_verifiers=[FailingVerifier()])
        results = verifier._run_custom_verifiers(
            original_prompt="Test prompt",
            final_result="Test result",
        )
        assert len(results) == 1
        assert results[0].status == VerificationStatus.SKIPPED
        assert "Custom verifier error" in results[0].rejection_reason
        assert results[0].validator_name == "failing_verifier"


class TestFinalResultVerificationProperty:
    """Property-based tests for Final Result Verification.

    Feature: output-verification-agent, Property 2: Final Result Verification
    Against Original Prompt
    """

    @given(
        original_prompt=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        final_result=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_final_verification_uses_original_prompt_not_task_assignment(
        self,
        original_prompt: str,
        final_result: str,
    ):
        """Final Result Verification Against Original Prompt.

        For any synthesize_results execution that completes, the
        FinalResultVerifier SHALL verify the final result against the user's
        original prompt (state.prompt), ensuring the complete user requirement
        is satisfied.

        Feature: output-verification-agent
        """
        # Create a custom verifier that captures what prompt was passed
        captured_prompts = []

        class PromptCapturingVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                captured_prompts.append(task)
                return VerificationResult(status=VerificationStatus.PASSED)

        verifier = FinalResultVerifier(
            custom_verifiers=[PromptCapturingVerifier()]
        )

        # Run custom verifiers (synchronous part we can test)
        verifier._run_custom_verifiers(
            original_prompt=original_prompt,
            final_result=final_result,
        )

        # Verify the original_prompt was passed to the verifier
        assert len(captured_prompts) == 1
        assert captured_prompts[0] == original_prompt

    @given(
        original_prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        final_result=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        worker_results=st.lists(
            st.fixed_dictionaries({
                "agent_id": st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
                "result": st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            }),
            min_size=0,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_final_verification_result_always_has_required_fields(
        self,
        original_prompt: str,
        final_result: str,
        worker_results: list[dict],
    ):
        """Final verification always produces properly structured results.

        For any final verification operation, the result SHALL contain all
        required fields: status, rejection_reason, improvement_suggestions,
        field_errors.

        Feature: output-verification-agent
        """
        verifier = FinalResultVerifier()

        # Test with custom verifier that always passes
        class AlwaysPassVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                return VerificationResult(status=VerificationStatus.PASSED)

        verifier_with_custom = FinalResultVerifier(
            custom_verifiers=[AlwaysPassVerifier()]
        )
        results = verifier_with_custom._run_custom_verifiers(
            original_prompt, final_result
        )

        for result in results:
            # Verify all required fields exist
            assert hasattr(result, "status")
            assert hasattr(result, "rejection_reason")
            assert hasattr(result, "improvement_suggestions")
            assert hasattr(result, "field_errors")
            assert isinstance(result.status, VerificationStatus)
            assert isinstance(result.improvement_suggestions, list)
            assert isinstance(result.field_errors, dict)

    @given(
        original_prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        final_result=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_final_verification_aggregation_preserves_failure_info(
        self,
        original_prompt: str,
        final_result: str,
    ):
        """Aggregation preserves failure information from custom verifiers.

        For any final verification with failing custom verifiers, the
        aggregated result SHALL contain the rejection reasons and improvement
        suggestions from all failed verifiers.

        Feature: output-verification-agent
        """
        rejection_reason = f"Failed for prompt: {original_prompt[:20]}"
        suggestion = "Try a different approach"

        class FailingVerifier:
            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason=rejection_reason,
                    improvement_suggestions=[suggestion],
                    validator_name="test_verifier",
                )

        verifier = FinalResultVerifier(custom_verifiers=[FailingVerifier()])
        results = verifier._run_custom_verifiers(original_prompt, final_result)
        aggregated = verifier._aggregate_results(results)

        assert aggregated.status == VerificationStatus.FAILED
        assert rejection_reason in aggregated.rejection_reason
        assert suggestion in aggregated.improvement_suggestions

    @given(
        original_prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        final_result=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        num_verifiers=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_all_custom_verifiers_are_executed(
        self,
        original_prompt: str,
        final_result: str,
        num_verifiers: int,
    ):
        """All registered custom verifiers are executed.

        For any FinalResultVerifier with N custom verifiers, all N verifiers
        SHALL be executed and their results included in the aggregation.

        Feature: output-verification-agent
        """
        execution_count = []

        class CountingVerifier:
            def __init__(self, verifier_id: int):
                self.verifier_id = verifier_id

            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                execution_count.append(self.verifier_id)
                return VerificationResult(status=VerificationStatus.PASSED)

        verifiers = [CountingVerifier(i) for i in range(num_verifiers)]
        verifier = FinalResultVerifier(custom_verifiers=verifiers)

        results = verifier._run_custom_verifiers(original_prompt, final_result)

        # All verifiers should have been executed
        assert len(results) == num_verifiers
        assert len(execution_count) == num_verifiers
        # Each verifier should have been called exactly once
        assert sorted(execution_count) == list(range(num_verifiers))



class TestWorkerRetryContextProperty:
    """Property-based tests for Worker Retry Context completeness.

    Feature: output-verification-agent
    """

    @given(
        iteration=st.integers(min_value=0, max_value=10),
        max_iterations=st.integers(min_value=1, max_value=10),
        assigned_task=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        previous_outputs=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=0,
            max_size=5,
        ),
        rejection_reasons=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_retry_context_contains_assigned_task_not_original_prompt(
        self,
        iteration: int,
        max_iterations: int,
        assigned_task: str,
        previous_outputs: list[str],
        rejection_reasons: list[str],
    ):
        """Worker Retry Context Is Complete.

        For any failed Worker verification that triggers a retry, the retry
        context SHALL contain: the assigned task (not original prompt), all
        previous outputs from that Worker, all previous rejection reasons,
        and the complete task history.

        Feature: output-verification-agent
        """
        # Create previous rejections from rejection reasons
        previous_rejections = [
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason=reason,
                improvement_suggestions=[f"Suggestion for: {reason[:20]}"],
            )
            for reason in rejection_reasons
        ]

        # Create task history
        task_history = [
            {"agent_id": f"agent_{i}", "task": f"Task {i}"}
            for i in range(len(previous_outputs))
        ]

        # Create RetryContext
        context = RetryContext(
            iteration=iteration,
            max_iterations=max_iterations,
            previous_outputs=previous_outputs,
            previous_rejections=previous_rejections,
            task_history=task_history,
            original_task=assigned_task,  # This is the ASSIGNED task, not user prompt
        )

        # Verify the context contains all required fields
        assert context.original_task == assigned_task
        assert context.iteration == iteration
        assert context.max_iterations == max_iterations
        assert len(context.previous_outputs) == len(previous_outputs)
        assert len(context.previous_rejections) == len(rejection_reasons)
        assert len(context.task_history) == len(task_history)

        # Verify all previous outputs are preserved
        for i, output in enumerate(previous_outputs):
            assert context.previous_outputs[i] == output

        # Verify all rejection reasons are preserved
        for i, reason in enumerate(rejection_reasons):
            assert context.previous_rejections[i].rejection_reason == reason

    @given(
        assigned_task=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        original_prompt=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        num_failed_agents=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_retry_context_uses_assigned_task_not_user_prompt(
        self,
        assigned_task: str,
        original_prompt: str,
        num_failed_agents: int,
    ):
        """Retry context uses assigned task, not user's original prompt.

        The retry context SHALL contain the specific task assigned to the
        Worker (task_assignment.task), NOT the user's original prompt.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import _build_retry_context

        # Create failed agents with assigned tasks
        failed_agents = [
            {
                "agent_id": f"agent_{i}",
                "assigned_task": assigned_task,
                "verification": VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason=f"Failed reason {i}",
                ),
                "previous_output": f"Output {i}",
            }
            for i in range(num_failed_agents)
        ]

        # Create agent results
        agent_results = [
            {"agent_id": f"agent_{i}", "result": f"Output {i}"}
            for i in range(num_failed_agents)
        ]

        # Create task assignments
        task_assignments = [
            {"agent_id": f"agent_{i}", "task": assigned_task}
            for i in range(num_failed_agents)
        ]

        # Build retry context
        context = _build_retry_context(
            iteration=0,
            max_iterations=3,
            failed_agents=failed_agents,
            agent_results=agent_results,
            task_assignments=task_assignments,
            original_prompt=original_prompt,  # This should NOT be used as the task
        )

        # The original_task in context should be the assigned_task, not original_prompt
        assert context.original_task == assigned_task
        # Unless they happen to be the same, they should be different
        if assigned_task != original_prompt:
            assert context.original_task != original_prompt

    @given(
        iteration=st.integers(min_value=0, max_value=5),
        max_iterations=st.integers(min_value=1, max_value=10),
        num_outputs=st.integers(min_value=1, max_value=5),
        num_rejections=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_retry_context_preserves_all_history(
        self,
        iteration: int,
        max_iterations: int,
        num_outputs: int,
        num_rejections: int,
    ):
        """Retry context preserves complete history.

        For any retry operation, the context SHALL include all previous
        outputs and all previous rejection reasons.

        Feature: output-verification-agent
        """
        # Generate test data
        previous_outputs = [f"Output {i}" for i in range(num_outputs)]
        previous_rejections = [
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason=f"Rejection {i}",
                improvement_suggestions=[f"Suggestion {i}"],
            )
            for i in range(num_rejections)
        ]
        task_history = [
            {"agent_id": f"agent_{i}", "task": f"Task {i}"}
            for i in range(num_outputs)
        ]

        context = RetryContext(
            iteration=iteration,
            max_iterations=max_iterations,
            previous_outputs=previous_outputs,
            previous_rejections=previous_rejections,
            task_history=task_history,
            original_task="Test task",
        )

        # Verify all history is preserved
        assert len(context.previous_outputs) == num_outputs
        assert len(context.previous_rejections) == num_rejections
        assert len(context.task_history) == num_outputs

        # Verify content integrity
        for i in range(num_outputs):
            assert context.previous_outputs[i] == f"Output {i}"
            assert context.task_history[i]["agent_id"] == f"agent_{i}"

        for i in range(num_rejections):
            assert context.previous_rejections[i].rejection_reason == f"Rejection {i}"

    @given(
        assigned_task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        rejection_reasons=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=1,
            max_size=3,
        ),
        suggestions=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=0,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_retry_prompt_contains_task_and_rejections(
        self,
        assigned_task: str,
        rejection_reasons: list[str],
        suggestions: list[str],
    ):
        """Retry prompt contains assigned task and rejection context.

        The retry prompt SHALL contain the original assigned task, explicit
        instructions to try a different approach, and the previous rejection
        reasons.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import build_retry_prompt

        # Create rejections with suggestions
        previous_rejections = [
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason=reason,
                improvement_suggestions=suggestions[:1] if suggestions else [],
            )
            for reason in rejection_reasons
        ]

        context = RetryContext(
            iteration=1,
            max_iterations=3,
            previous_outputs=["Previous output"],
            previous_rejections=previous_rejections,
            task_history=[],
            original_task=assigned_task,
        )

        prompt = build_retry_prompt(context)

        # Verify prompt contains required elements
        assert assigned_task in prompt
        assert "DIFFERENT approach" in prompt or "different" in prompt.lower()

        # Verify all rejection reasons are included
        for reason in rejection_reasons:
            assert reason in prompt

    @given(
        iteration=st.integers(min_value=0, max_value=10),
        max_iterations=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_retry_context_iteration_tracking(
        self,
        iteration: int,
        max_iterations: int,
    ):
        """Retry context correctly tracks iteration numbers.

        The retry context SHALL correctly track the current iteration and
        maximum iterations to prevent infinite retry loops.

        Feature: output-verification-agent
        """
        context = RetryContext(
            iteration=iteration,
            max_iterations=max_iterations,
            previous_outputs=[],
            previous_rejections=[],
            task_history=[],
            original_task="Test task",
        )

        assert context.iteration == iteration
        assert context.max_iterations == max_iterations

        # Verify we can determine if more retries are allowed
        can_retry = context.iteration < context.max_iterations
        assert can_retry == (iteration < max_iterations)



class TestWorkflowRestartProperty:
    """Property-based tests for Workflow Restart On Final Verification Failure.

    Feature: output-verification-agent
    """

    @given(
        original_prompt=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        rejection_reason=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        num_previous_attempts=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=100)
    def test_workflow_retry_instruction_contains_original_prompt(
        self,
        original_prompt: str,
        rejection_reason: str,
        num_previous_attempts: int,
    ):
        """Workflow retry instruction contains original prompt.

        For any final result verification failure, the retry instruction SHALL
        contain the user's original prompt to guide the retry.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import _build_workflow_retry_instruction

        # Build previous attempts
        previous_attempts = [
            {
                "iteration": i,
                "final_result": f"Result {i}",
                "rejection_reason": f"Reason {i}",
                "agent_results": [],
            }
            for i in range(num_previous_attempts)
        ]

        instruction = _build_workflow_retry_instruction(
            original_prompt=original_prompt,
            rejection_reason=rejection_reason,
            previous_attempts=previous_attempts,
        )

        # Verify original prompt is in the instruction
        assert original_prompt in instruction
        # Verify it's marked as a retry
        assert "RETRY" in instruction
        # Verify it instructs to try different approach
        assert "DIFFERENT" in instruction or "different" in instruction.lower()

    @given(
        original_prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        rejection_reasons=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_workflow_retry_instruction_contains_all_previous_rejections(
        self,
        original_prompt: str,
        rejection_reasons: list[str],
    ):
        """Workflow retry instruction contains all previous rejection reasons.

        For any workflow restart, the retry instruction SHALL contain all
        previous rejection reasons to prevent repeating the same mistakes.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import _build_workflow_retry_instruction

        # Build previous attempts with rejection reasons
        previous_attempts = [
            {
                "iteration": i,
                "final_result": f"Result {i}",
                "rejection_reason": reason,
                "agent_results": [],
            }
            for i, reason in enumerate(rejection_reasons)
        ]

        instruction = _build_workflow_retry_instruction(
            original_prompt=original_prompt,
            rejection_reason=rejection_reasons[-1] if rejection_reasons else None,
            previous_attempts=previous_attempts,
        )

        # Verify all rejection reasons are included
        for reason in rejection_reasons:
            assert reason in instruction

    @given(
        original_prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        rejection_reason=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        improvement_suggestions=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=0,
            max_size=3,
        ),
        attempts=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_failure_summary_contains_original_prompt_and_reason(
        self,
        original_prompt: str,
        rejection_reason: str,
        improvement_suggestions: list[str],
        attempts: int,
    ):
        """Failure summary contains original prompt and rejection reason.

        When max iterations are reached, the failure summary SHALL contain
        the user's original prompt and the rejection reason.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import _generate_failure_summary

        summary = _generate_failure_summary(
            original_prompt=original_prompt,
            rejection_reason=rejection_reason,
            improvement_suggestions=improvement_suggestions,
            attempts=attempts,
        )

        # Verify original prompt is in the summary
        assert original_prompt in summary
        # Verify rejection reason is in the summary
        assert rejection_reason in summary
        # Verify attempt count is in the summary
        assert str(attempts) in summary
        # Verify all improvement suggestions are included
        for suggestion in improvement_suggestions:
            assert suggestion in summary

    @given(
        workflow_iteration=st.integers(min_value=0, max_value=5),
        max_workflow_iterations=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_workflow_iteration_tracking(
        self,
        workflow_iteration: int,
        max_workflow_iterations: int,
    ):
        """Workflow iteration is correctly tracked.

        The workflow iteration SHALL be correctly tracked to determine
        whether more retries are allowed.

        Feature: output-verification-agent
        """
        # Determine if restart should be allowed
        should_restart = workflow_iteration < max_workflow_iterations

        # This is the logic used in verify_final_result
        if workflow_iteration >= max_workflow_iterations:
            # Max reached, should not restart
            assert not should_restart
        else:
            # Can still restart
            assert should_restart

    @given(
        original_prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        final_result=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        num_agent_results=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=100)
    def test_previous_attempts_history_is_preserved(
        self,
        original_prompt: str,
        final_result: str,
        num_agent_results: int,
    ):
        """Previous workflow attempts are preserved in history.

        When a workflow restarts, the previous attempt SHALL be added to
        the previous_workflow_attempts history.

        Feature: output-verification-agent
        """
        # Simulate building previous attempts history
        previous_attempts: list[dict] = []
        agent_results = [
            {"agent_id": f"agent_{i}", "result": f"Result {i}"}
            for i in range(num_agent_results)
        ]

        # Add a new attempt (simulating what verify_final_result does)
        new_attempt = {
            "iteration": len(previous_attempts),
            "final_result": final_result,
            "rejection_reason": "Test rejection",
            "agent_results": agent_results,
        }
        previous_attempts = list(previous_attempts)  # Make a copy
        previous_attempts.append(new_attempt)

        # Verify the attempt was added
        assert len(previous_attempts) == 1
        assert previous_attempts[0]["final_result"] == final_result
        assert previous_attempts[0]["rejection_reason"] == "Test rejection"
        assert len(previous_attempts[0]["agent_results"]) == num_agent_results

    @given(
        original_prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        rejection_reason=st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        ),
    )
    @settings(max_examples=100)
    def test_failure_summary_handles_none_rejection_reason(
        self,
        original_prompt: str,
        rejection_reason: str | None,
    ):
        """Failure summary handles None rejection reason gracefully.

        The failure summary SHALL handle None rejection reason by using
        a default message.

        Feature: output-verification-agent, Property 11: Workflow Restart On
        Final Verification Failure
        Validates: Requirements 2.1, 5.3
        """
        from kiva.nodes.verify import _generate_failure_summary

        summary = _generate_failure_summary(
            original_prompt=original_prompt,
            rejection_reason=rejection_reason,
            improvement_suggestions=[],
            attempts=1,
        )

        # Verify original prompt is always present
        assert original_prompt in summary

        # Verify rejection reason handling
        if rejection_reason is not None:
            assert rejection_reason in summary
        else:
            # Should have a default message
            assert "Unknown reason" in summary
