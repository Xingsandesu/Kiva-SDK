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
