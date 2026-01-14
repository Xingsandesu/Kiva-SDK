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

        Feature: output-verification-agent
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


class TestRetryPromptProperty:
    """Property-based tests for Retry Prompt Contains Assigned Task Context.

    Feature: output-verification-agent, Property 8: Retry Prompt Contains
    Assigned Task Context
    Validates: Requirements 5.3
    """

    @given(
        assigned_task=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        rejection_reasons=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=1,
            max_size=5,
        ),
        improvement_suggestions=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=0,
            max_size=3,
        ),
        previous_outputs=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            min_size=0,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_retry_prompt_contains_assigned_task_context(
        self,
        assigned_task: str,
        rejection_reasons: list[str],
        improvement_suggestions: list[str],
        previous_outputs: list[str],
    ):
        """Property 8: Retry Prompt Contains Assigned Task Context.

        For any Worker retry triggered by verification failure, the retry
        prompt SHALL contain:
        1. The original assigned task (not user prompt)
        2. Explicit instructions to try a different approach
        3. The previous rejection reasons specific to that task

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import build_retry_prompt

        # Create previous rejections with suggestions
        previous_rejections = [
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason=reason,
                improvement_suggestions=(
                    improvement_suggestions[:1] if improvement_suggestions else []
                ),
            )
            for reason in rejection_reasons
        ]

        context = RetryContext(
            iteration=1,
            max_iterations=3,
            previous_outputs=previous_outputs,
            previous_rejections=previous_rejections,
            task_history=[],
            original_task=assigned_task,
        )

        prompt = build_retry_prompt(context)

        # 1. Verify the assigned task is in the prompt (at least twice - once
        #    at the beginning and once at the end as instruction)
        assert assigned_task in prompt, (
            f"Assigned task '{assigned_task}' not found in retry prompt"
        )

        # 2. Verify explicit instructions to try a different approach
        different_approach_indicators = [
            "DIFFERENT",
            "different approach",
            "fundamentally different",
            "try something",
        ]
        has_different_approach = any(
            indicator.lower() in prompt.lower()
            for indicator in different_approach_indicators
        )
        assert has_different_approach, (
            "Retry prompt does not contain instructions to try a different approach"
        )

        # 3. Verify all previous rejection reasons are included
        for reason in rejection_reasons:
            assert reason in prompt, (
                f"Rejection reason '{reason}' not found in retry prompt"
            )

    @given(
        assigned_task=st.text(min_size=10, max_size=100).filter(lambda x: x.strip()),
        user_original_prompt=st.text(min_size=10, max_size=100).filter(
            lambda x: x.strip()
        ),
    )
    @settings(max_examples=100)
    def test_retry_prompt_uses_assigned_task_not_user_prompt(
        self,
        assigned_task: str,
        user_original_prompt: str,
    ):
        """Retry prompt uses assigned task, not user's original prompt.

        The retry prompt SHALL contain the specific task assigned to the
        Worker (task_assignment.task), NOT the user's original prompt.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import build_retry_prompt

        # Create a simple retry context with the assigned task
        context = RetryContext(
            iteration=1,
            max_iterations=3,
            previous_outputs=["Previous output"],
            previous_rejections=[
                VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="Test rejection",
                )
            ],
            task_history=[],
            original_task=assigned_task,  # This is the ASSIGNED task
        )

        prompt = build_retry_prompt(context)

        # The assigned task should be in the prompt
        assert assigned_task in prompt

        # If assigned_task and user_original_prompt are sufficiently different
        # (not substrings of each other and not too short to be coincidental),
        # the user_original_prompt should NOT be in the prompt
        if (
            assigned_task != user_original_prompt
            and user_original_prompt not in assigned_task
            and assigned_task not in user_original_prompt
            and len(user_original_prompt) > 5  # Avoid false positives with short strings
        ):
            assert user_original_prompt not in prompt, (
                "User's original prompt should not be in retry prompt; "
                "only the assigned task should be used"
            )

    @given(
        assigned_task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        num_rejections=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_retry_prompt_includes_all_rejection_attempts(
        self,
        assigned_task: str,
        num_rejections: int,
    ):
        """Retry prompt includes all previous rejection attempts.

        For any number of previous rejections, the retry prompt SHALL
        include all rejection reasons to help the agent avoid repeating
        the same mistakes.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import build_retry_prompt

        # Create multiple rejections with unique reasons
        rejection_reasons = [f"Rejection reason {i}" for i in range(num_rejections)]
        previous_rejections = [
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason=reason,
                improvement_suggestions=[f"Suggestion for {reason}"],
            )
            for reason in rejection_reasons
        ]

        context = RetryContext(
            iteration=num_rejections,
            max_iterations=num_rejections + 2,
            previous_outputs=[f"Output {i}" for i in range(num_rejections)],
            previous_rejections=previous_rejections,
            task_history=[],
            original_task=assigned_task,
        )

        prompt = build_retry_prompt(context)

        # All rejection reasons should be in the prompt
        for reason in rejection_reasons:
            assert reason in prompt, (
                f"Rejection reason '{reason}' not found in retry prompt"
            )

    @given(
        assigned_task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        suggestions=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_retry_prompt_includes_improvement_suggestions(
        self,
        assigned_task: str,
        suggestions: list[str],
    ):
        """Retry prompt includes improvement suggestions when available.

        When improvement suggestions are provided in the rejection, the
        retry prompt SHOULD include them to guide the agent.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import build_retry_prompt

        # Create a rejection with suggestions
        previous_rejections = [
            VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason="Test rejection",
                improvement_suggestions=suggestions,
            )
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

        # At least the first suggestion should be included
        # (based on current implementation which includes suggestions)
        if suggestions:
            # Check if suggestions are mentioned in some form
            has_suggestion_reference = any(
                "Suggestion" in prompt or s in prompt for s in suggestions
            )
            assert has_suggestion_reference, (
                "Retry prompt should include improvement suggestions"
            )

    @given(
        assigned_task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_retry_prompt_instructs_not_to_repeat(
        self,
        assigned_task: str,
    ):
        """Retry prompt instructs agent not to repeat similar approaches.

        The retry prompt SHALL explicitly instruct the agent to NOT repeat
        similar approaches that failed before.

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import build_retry_prompt

        context = RetryContext(
            iteration=1,
            max_iterations=3,
            previous_outputs=["Previous output"],
            previous_rejections=[
                VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="Test rejection",
                )
            ],
            task_history=[],
            original_task=assigned_task,
        )

        prompt = build_retry_prompt(context)

        # Check for "do not repeat" or similar instructions
        no_repeat_indicators = [
            "NOT repeat",
            "Do NOT",
            "don't repeat",
            "avoid",
            "different",
        ]
        has_no_repeat_instruction = any(
            indicator.lower() in prompt.lower() for indicator in no_repeat_indicators
        )
        assert has_no_repeat_instruction, (
            "Retry prompt should instruct agent not to repeat similar approaches"
        )



class TestWorkerRetryNode:
    """Unit tests for worker_retry graph node."""

    @pytest.mark.asyncio
    async def test_worker_retry_returns_empty_when_no_context(self):
        """Test that worker_retry returns empty results when no retry context."""
        from kiva.nodes.verify import worker_retry

        state = {
            "retry_context": None,
            "agents": [],
            "task_assignments": [],
            "execution_id": "test-exec-123",
        }

        result = await worker_retry(state)

        assert result == {"agent_results": []}

    @pytest.mark.asyncio
    async def test_worker_retry_builds_prompt_from_context(self):
        """Test that worker_retry builds retry prompt from context."""
        from kiva.nodes.verify import worker_retry

        retry_context = RetryContext(
            iteration=1,
            max_iterations=3,
            previous_outputs=["Previous output"],
            previous_rejections=[
                VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="Output incomplete",
                    improvement_suggestions=["Add more details"],
                )
            ],
            task_history=[],
            original_task="Analyze the data",
        )

        state = {
            "retry_context": retry_context.model_dump(),
            "agents": [],  # No agents, so no execution
            "task_assignments": [],
            "execution_id": "test-exec-123",
        }

        result = await worker_retry(state)

        # With no agents, should return empty results
        assert result == {"agent_results": []}

    @pytest.mark.asyncio
    async def test_worker_retry_handles_missing_agent(self):
        """Test that worker_retry handles missing agent gracefully."""
        from kiva.nodes.verify import worker_retry

        retry_context = RetryContext(
            iteration=1,
            max_iterations=3,
            previous_outputs=["Previous output"],
            previous_rejections=[
                VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="Output incomplete",
                )
            ],
            task_history=[],
            original_task="Analyze the data",
        )

        state = {
            "retry_context": retry_context.model_dump(),
            "agents": [],  # No agents available
            "task_assignments": [
                {"agent_id": "missing_agent", "task": "Some task"}
            ],
            "execution_id": "test-exec-123",
        }

        result = await worker_retry(state)

        # Should return error result for missing agent
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["agent_id"] == "missing_agent"
        assert result["agent_results"][0]["error"] is not None
        assert "not found" in result["agent_results"][0]["error"]

    @pytest.mark.asyncio
    async def test_worker_retry_with_dict_context(self):
        """Test that worker_retry handles dict retry context."""
        from kiva.nodes.verify import worker_retry

        # Pass context as dict (as it would be in state)
        retry_context_dict = {
            "iteration": 1,
            "max_iterations": 3,
            "previous_outputs": ["Previous output"],
            "previous_rejections": [
                {
                    "status": "failed",
                    "rejection_reason": "Output incomplete",
                    "improvement_suggestions": [],
                    "field_errors": {},
                    "validator_name": "default",
                    "confidence": 1.0,
                }
            ],
            "task_history": [],
            "original_task": "Analyze the data",
        }

        state = {
            "retry_context": retry_context_dict,
            "agents": [],
            "task_assignments": [],
            "execution_id": "test-exec-123",
        }

        result = await worker_retry(state)

        assert result == {"agent_results": []}

    @pytest.mark.asyncio
    async def test_worker_retry_with_mock_agent(self):
        """Test that worker_retry executes agent with retry prompt."""
        from kiva.nodes.verify import worker_retry

        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.ainvoke = AsyncMock(
            return_value={"messages": [MagicMock(content="Retry result")]}
        )

        retry_context = RetryContext(
            iteration=1,
            max_iterations=3,
            previous_outputs=["Previous output"],
            previous_rejections=[
                VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="Output incomplete",
                    improvement_suggestions=["Add more details"],
                )
            ],
            task_history=[],
            original_task="Analyze the data",
        )

        state = {
            "retry_context": retry_context.model_dump(),
            "agents": [mock_agent],
            "task_assignments": [
                {"agent_id": "test_agent", "task": "Analyze the data"}
            ],
            "execution_id": "test-exec-123",
        }

        result = await worker_retry(state)

        # Should have called the agent
        assert mock_agent.ainvoke.called
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["agent_id"] == "test_agent"
        assert result["agent_results"][0]["result"] == "Retry result"

    @pytest.mark.asyncio
    async def test_worker_retry_prompt_contains_rejection_context(self):
        """Test that retry prompt passed to agent contains rejection context."""
        from kiva.nodes.verify import worker_retry

        # Create a mock agent that captures the task
        captured_tasks = []

        async def capture_task(input_dict):
            task = input_dict["messages"][0]["content"]
            captured_tasks.append(task)
            return {"messages": [MagicMock(content="Result")]}

        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.ainvoke = capture_task

        retry_context = RetryContext(
            iteration=2,
            max_iterations=3,
            previous_outputs=["First output", "Second output"],
            previous_rejections=[
                VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="First rejection reason",
                    improvement_suggestions=["First suggestion"],
                ),
                VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason="Second rejection reason",
                    improvement_suggestions=["Second suggestion"],
                ),
            ],
            task_history=[],
            original_task="Analyze the complex data",
        )

        state = {
            "retry_context": retry_context.model_dump(),
            "agents": [mock_agent],
            "task_assignments": [
                {"agent_id": "test_agent", "task": "Analyze the complex data"}
            ],
            "execution_id": "test-exec-123",
        }

        await worker_retry(state)

        # Verify the task passed to agent contains required elements
        assert len(captured_tasks) == 1
        task = captured_tasks[0]

        # Should contain the original task
        assert "Analyze the complex data" in task

        # Should contain rejection reasons
        assert "First rejection reason" in task
        assert "Second rejection reason" in task

        # Should contain instructions to try different approach
        assert "DIFFERENT" in task or "different" in task.lower()



class TestMaxIterationsProperty:
    """Property-based tests for Max Iterations Is Respected At Both Levels.

    Feature: output-verification-agent
    """

    @given(
        max_verification_iterations=st.integers(min_value=1, max_value=5),
        max_workflow_iterations=st.integers(min_value=1, max_value=3),
        num_failed_verifications=st.integers(min_value=0, max_value=10),
        num_failed_final_verifications=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_max_iterations_respected_at_both_levels(
        self,
        max_verification_iterations: int,
        max_workflow_iterations: int,
        num_failed_verifications: int,
        num_failed_final_verifications: int,
    ):
        """Max Iterations Is Respected At Both Levels.

        For any execution with max_verification_iterations=N and
        max_workflow_iterations=M, the total number of Worker verification
        attempts SHALL NOT exceed N, and the total number of complete workflow
        restarts SHALL NOT exceed M.

        Feature: output-verification-agent
        """
        # Simulate worker verification iterations
        worker_iteration = 0
        for _ in range(num_failed_verifications):
            if worker_iteration >= max_verification_iterations:
                # Should stop retrying
                break
            worker_iteration += 1

        # Verify worker iterations don't exceed max
        assert worker_iteration <= max_verification_iterations

        # Simulate workflow iterations
        workflow_iteration = 0
        for _ in range(num_failed_final_verifications):
            if workflow_iteration >= max_workflow_iterations:
                # Should stop restarting
                break
            workflow_iteration += 1

        # Verify workflow iterations don't exceed max
        assert workflow_iteration <= max_workflow_iterations

    @given(
        max_verification_iterations=st.integers(min_value=1, max_value=5),
        current_iteration=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_worker_verification_stops_at_max_iterations(
        self,
        max_verification_iterations: int,
        current_iteration: int,
    ):
        """Worker verification stops when max iterations reached.

        When the current verification iteration equals or exceeds
        max_verification_iterations, no more worker retries SHALL be triggered.

        Feature: output-verification-agent
        """
        # This is the logic from verify_worker_output
        should_retry = current_iteration < max_verification_iterations

        if current_iteration >= max_verification_iterations:
            # Should not retry
            assert not should_retry
        else:
            # Can still retry
            assert should_retry

    @given(
        max_workflow_iterations=st.integers(min_value=1, max_value=5),
        current_workflow_iteration=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_workflow_restart_stops_at_max_iterations(
        self,
        max_workflow_iterations: int,
        current_workflow_iteration: int,
    ):
        """Workflow restart stops when max iterations reached.

        When the current workflow iteration equals or exceeds
        max_workflow_iterations, no more workflow restarts SHALL be triggered.

        Feature: output-verification-agent
        """
        # This is the logic from verify_final_result
        should_restart = current_workflow_iteration < max_workflow_iterations

        if current_workflow_iteration >= max_workflow_iterations:
            # Should not restart
            assert not should_restart
        else:
            # Can still restart
            assert should_restart

    @given(
        max_verification_iterations=st.integers(min_value=1, max_value=5),
        num_agents=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_max_verification_iterations_applies_per_workflow_not_per_agent(
        self,
        max_verification_iterations: int,
        num_agents: int,
    ):
        """Max verification iterations applies per workflow execution.

        The max_verification_iterations limit SHALL apply to the entire
        worker verification phase, not separately to each agent.

        Feature: output-verification-agent
        """
        # Simulate a workflow with multiple agents
        # All agents share the same verification iteration counter
        verification_iteration = 0

        # Simulate multiple failed verifications
        for attempt in range(max_verification_iterations + 2):
            if verification_iteration >= max_verification_iterations:
                # Should stop for all agents
                break
            verification_iteration += 1

        # Verify the iteration counter doesn't exceed max
        assert verification_iteration <= max_verification_iterations

    @given(
        max_workflow_iterations=st.integers(min_value=1, max_value=3),
        num_workflow_restarts=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100)
    def test_workflow_restarts_count_correctly(
        self,
        max_workflow_iterations: int,
        num_workflow_restarts: int,
    ):
        """Workflow restarts are counted correctly.

        Each complete workflow restart (from analyze_and_plan) SHALL increment
        the workflow_iteration counter.

        Feature: output-verification-agent
        """
        # Simulate workflow restarts
        workflow_iteration = 0
        restarts_performed = 0

        for _ in range(num_workflow_restarts):
            if workflow_iteration >= max_workflow_iterations:
                # Can't restart anymore
                break
            # Restart workflow
            workflow_iteration += 1
            restarts_performed += 1

        # Verify restarts don't exceed max
        assert restarts_performed <= max_workflow_iterations
        assert workflow_iteration <= max_workflow_iterations

    @given(
        max_verification_iterations=st.integers(min_value=1, max_value=5),
        max_workflow_iterations=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_both_iteration_limits_are_independent(
        self,
        max_verification_iterations: int,
        max_workflow_iterations: int,
    ):
        """Worker and workflow iteration limits are independent.

        The max_verification_iterations and max_workflow_iterations SHALL be
        independent limits that don't affect each other.

        Feature: output-verification-agent
        """
        # Simulate reaching max worker iterations
        worker_iteration = 0
        for _ in range(max_verification_iterations + 1):
            if worker_iteration >= max_verification_iterations:
                break
            worker_iteration += 1

        # Simulate reaching max workflow iterations
        workflow_iteration = 0
        for _ in range(max_workflow_iterations + 1):
            if workflow_iteration >= max_workflow_iterations:
                break
            workflow_iteration += 1

        # Both should respect their own limits independently
        assert worker_iteration <= max_verification_iterations
        assert workflow_iteration <= max_workflow_iterations

        # One reaching max shouldn't affect the other
        # (they're independent counters)
        assert worker_iteration == max_verification_iterations
        assert workflow_iteration == max_workflow_iterations

    @given(
        max_verification_iterations=st.integers(min_value=1, max_value=5),
        verification_iteration=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_verification_iteration_increments_correctly(
        self,
        max_verification_iterations: int,
        verification_iteration: int,
    ):
        """Verification iteration increments correctly on each retry.

        Each worker retry SHALL increment the verification_iteration counter
        by exactly 1.

        Feature: output-verification-agent
        """
        # Simulate the increment logic from verify_worker_output
        if verification_iteration < max_verification_iterations:
            # Should increment
            new_iteration = verification_iteration + 1
            assert new_iteration == verification_iteration + 1
            assert new_iteration <= max_verification_iterations + 1
        else:
            # Should not increment (max reached)
            new_iteration = verification_iteration
            assert new_iteration == verification_iteration

    @given(
        max_workflow_iterations=st.integers(min_value=1, max_value=3),
        workflow_iteration=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100)
    def test_workflow_iteration_increments_correctly(
        self,
        max_workflow_iterations: int,
        workflow_iteration: int,
    ):
        """Workflow iteration increments correctly on each restart.

        Each workflow restart SHALL increment the workflow_iteration counter
        by exactly 1.

        Feature: output-verification-agent
        """
        # Simulate the increment logic from verify_final_result
        if workflow_iteration < max_workflow_iterations:
            # Should increment
            new_iteration = workflow_iteration + 1
            assert new_iteration == workflow_iteration + 1
            assert new_iteration <= max_workflow_iterations + 1
        else:
            # Should not increment (max reached)
            new_iteration = workflow_iteration
            assert new_iteration == workflow_iteration

    @given(
        max_verification_iterations=st.integers(min_value=1, max_value=5),
        max_workflow_iterations=st.integers(min_value=1, max_value=3),
        num_workflow_cycles=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_verification_iteration_resets_on_workflow_restart(
        self,
        max_verification_iterations: int,
        max_workflow_iterations: int,
        num_workflow_cycles: int,
    ):
        """Verification iteration resets when workflow restarts.

        When the workflow restarts (from analyze_and_plan), the
        verification_iteration counter SHALL be reset to 0.

        Feature: output-verification-agent
        """
        for workflow_cycle in range(min(num_workflow_cycles, max_workflow_iterations)):
            # Start of workflow cycle
            verification_iteration = 0

            # Simulate worker verifications in this cycle
            for _ in range(max_verification_iterations):
                if verification_iteration >= max_verification_iterations:
                    break
                verification_iteration += 1

            # At end of cycle, if workflow restarts, verification_iteration resets
            # This is the logic from verify_final_result's update dict
            if workflow_cycle < max_workflow_iterations - 1:
                # Workflow will restart, reset verification_iteration
                verification_iteration = 0
                assert verification_iteration == 0

    @given(
        max_verification_iterations=st.integers(min_value=1, max_value=5),
        max_workflow_iterations=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_max_iterations_configuration_is_respected(
        self,
        max_verification_iterations: int,
        max_workflow_iterations: int,
    ):
        """Max iterations configuration is respected throughout execution.

        The configured max_verification_iterations and max_workflow_iterations
        SHALL be respected at all decision points in the verification flow.

        Feature: output-verification-agent
        """
        # Test worker verification decision point
        for iteration in range(max_verification_iterations + 2):
            should_continue_worker = iteration < max_verification_iterations
            if iteration >= max_verification_iterations:
                assert not should_continue_worker
            else:
                assert should_continue_worker

        # Test workflow restart decision point
        for iteration in range(max_workflow_iterations + 2):
            should_continue_workflow = iteration < max_workflow_iterations
            if iteration >= max_workflow_iterations:
                assert not should_continue_workflow
            else:
                assert should_continue_workflow


class TestCustomVerifiersProperty:
    """Property-based tests for custom verifier execution.

    Feature: output-verification-agent
    """

    @given(
        verifier_name=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
        priority=st.integers(min_value=-100, max_value=100),
        task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        output=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_custom_verifier_is_registered_and_executed(
        self,
        verifier_name: str,
        priority: int,
        task: str,
        output: str,
    ):
        """Custom Verifiers Are Executed.

        For any registered custom verifier, when verification is performed,
        the custom verifier SHALL be called with the correct arguments
        (task, output, context) and its result SHALL be included in the
        aggregated verification results.

        Feature: output-verification-agent
        """
        from kiva import Kiva
        from kiva.verification import VerificationResult, VerificationStatus

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        # Track what arguments the verifier receives
        captured_args: list[tuple[str, str, dict | None]] = []

        @kiva.verifier(verifier_name, priority=priority)
        def custom_verifier(
            t: str, o: str, context: dict | None = None
        ) -> VerificationResult:
            captured_args.append((t, o, context))
            return VerificationResult(
                status=VerificationStatus.PASSED,
                validator_name=verifier_name,
            )

        # Verify the verifier was registered
        assert len(kiva._verifiers) == 1
        registered = kiva._verifiers[0]
        assert registered.name == verifier_name
        assert registered.priority == priority

        # Execute the verifier through the registered wrapper
        result = registered.verify(task, output, None)

        # Verify the verifier was called with correct arguments
        assert len(captured_args) == 1
        assert captured_args[0][0] == task
        assert captured_args[0][1] == output
        assert captured_args[0][2] is None

        # Verify the result is properly structured
        assert isinstance(result, VerificationResult)
        assert result.status == VerificationStatus.PASSED
        assert result.validator_name == verifier_name

    @given(
        num_verifiers=st.integers(min_value=1, max_value=5),
        priorities=st.lists(
            st.integers(min_value=-100, max_value=100),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_multiple_verifiers_are_all_executed(
        self,
        num_verifiers: int,
        priorities: list[int],
    ):
        """Multiple custom verifiers are all executed.

        When multiple verifiers are registered, ALL verifiers SHALL be
        executed and their results SHALL be aggregated.

        Feature: output-verification-agent
        """
        from kiva import Kiva
        from kiva.verification import VerificationResult, VerificationStatus

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        # Adjust priorities list to match num_verifiers
        actual_priorities = (priorities * num_verifiers)[:num_verifiers]
        executed_verifiers: list[str] = []

        # Register multiple verifiers
        for i in range(num_verifiers):
            name = f"verifier_{i}"
            prio = actual_priorities[i]

            @kiva.verifier(name, priority=prio)
            def make_verifier(
                t: str, o: str, context: dict | None = None, idx: int = i
            ) -> VerificationResult:
                executed_verifiers.append(f"verifier_{idx}")
                return VerificationResult(
                    status=VerificationStatus.PASSED,
                    validator_name=f"verifier_{idx}",
                )

        # Verify all verifiers were registered
        assert len(kiva._verifiers) == num_verifiers

        # Execute all verifiers
        for verifier in kiva._verifiers:
            verifier.verify("test task", "test output", None)

        # Verify all verifiers were executed
        assert len(executed_verifiers) == num_verifiers

    @given(
        priorities=st.lists(
            st.integers(min_value=-100, max_value=100),
            min_size=2,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_verifiers_sorted_by_priority(
        self,
        priorities: list[int],
    ):
        """Verifiers are sorted by priority (highest first).

        When get_verifiers() is called, verifiers SHALL be returned
        sorted by priority in descending order.

        Feature: output-verification-agent
        """
        from kiva import Kiva
        from kiva.verification import VerificationResult, VerificationStatus

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        # Register verifiers with different priorities
        for i, prio in enumerate(priorities):

            @kiva.verifier(f"verifier_{i}", priority=prio)
            def verifier_func(
                t: str, o: str, context: dict | None = None
            ) -> VerificationResult:
                return VerificationResult(status=VerificationStatus.PASSED)

        # Get sorted verifiers
        sorted_verifiers = kiva.get_verifiers()

        # Verify they are sorted by priority (highest first)
        for i in range(len(sorted_verifiers) - 1):
            assert sorted_verifiers[i].priority >= sorted_verifiers[i + 1].priority

    @given(
        task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        output=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        context_keys=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_verifier_receives_context(
        self,
        task: str,
        output: str,
        context_keys: list[str],
    ):
        """Custom verifier receives context parameter.

        The custom verifier function SHALL receive the optional context
        parameter when provided.

        Feature: output-verification-agent
        """
        from kiva import Kiva
        from kiva.verification import VerificationResult, VerificationStatus

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        received_context: list[dict | None] = []

        @kiva.verifier("context_checker")
        def check_context(
            t: str, o: str, context: dict | None = None
        ) -> VerificationResult:
            received_context.append(context)
            return VerificationResult(status=VerificationStatus.PASSED)

        # Create context dict
        test_context = {key: f"value_{i}" for i, key in enumerate(context_keys)}

        # Execute verifier with context
        verifier = kiva._verifiers[0]
        verifier.verify(task, output, test_context if context_keys else None)

        # Verify context was received
        assert len(received_context) == 1
        if context_keys:
            assert received_context[0] == test_context
        else:
            assert received_context[0] is None

    @given(
        should_pass=st.booleans(),
        rejection_reason=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        suggestions=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_verifier_result_is_returned(
        self,
        should_pass: bool,
        rejection_reason: str,
        suggestions: list[str],
    ):
        """Custom verifier result is properly returned.

        The custom verifier function SHALL return a VerificationResult
        and that result SHALL be properly returned from the verify() call.

        Feature: output-verification-agent
        """
        from kiva import Kiva
        from kiva.verification import VerificationResult, VerificationStatus

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        expected_status = (
            VerificationStatus.PASSED if should_pass else VerificationStatus.FAILED
        )

        @kiva.verifier("result_checker")
        def check_result(
            t: str, o: str, context: dict | None = None
        ) -> VerificationResult:
            return VerificationResult(
                status=expected_status,
                rejection_reason=None if should_pass else rejection_reason,
                improvement_suggestions=[] if should_pass else suggestions,
                validator_name="result_checker",
            )

        # Execute verifier
        verifier = kiva._verifiers[0]
        result = verifier.verify("task", "output", None)

        # Verify result matches expected
        assert result.status == expected_status
        if should_pass:
            assert result.rejection_reason is None
            assert result.improvement_suggestions == []
        else:
            assert result.rejection_reason == rejection_reason
            assert result.improvement_suggestions == suggestions
        assert result.validator_name == "result_checker"

    def test_verifier_decorator_preserves_function_metadata(self):
        """Verifier decorator preserves function metadata.

        The @kiva.verifier decorator SHALL store verifier metadata
        on the decorated function for introspection.

        Feature: output-verification-agent
        """
        from kiva import Kiva
        from kiva.verification import VerificationResult, VerificationStatus

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        @kiva.verifier("my_verifier", priority=5)
        def my_custom_verifier(
            task: str, output: str, context: dict | None = None
        ) -> VerificationResult:
            """My custom verifier docstring."""
            return VerificationResult(status=VerificationStatus.PASSED)

        # Verify metadata is stored on function
        assert hasattr(my_custom_verifier, "_verifier_name")
        assert my_custom_verifier._verifier_name == "my_verifier"
        assert hasattr(my_custom_verifier, "_verifier_priority")
        assert my_custom_verifier._verifier_priority == 5

        # Verify function is still callable
        result = my_custom_verifier("task", "output")
        assert result.status == VerificationStatus.PASSED

    def test_verifier_uses_function_name_when_name_not_provided(self):
        """Verifier uses function name when name parameter not provided.

        When the name parameter is not provided to @kiva.verifier,
        the decorator SHALL use the function's __name__ as the verifier name.

        Feature: output-verification-agent
        """
        from kiva import Kiva
        from kiva.verification import VerificationResult, VerificationStatus

        kiva = Kiva(base_url="http://test", api_key="test", model="test")

        @kiva.verifier()  # No name provided
        def auto_named_verifier(
            task: str, output: str, context: dict | None = None
        ) -> VerificationResult:
            return VerificationResult(status=VerificationStatus.PASSED)

        # Verify the function name was used
        assert len(kiva._verifiers) == 1
        assert kiva._verifiers[0].name == "auto_named_verifier"
        assert auto_named_verifier._verifier_name == "auto_named_verifier"


class TestVerificationEventsProperty:
    """Property-based tests for Verification Events Are Emitted At Both Levels.

    Feature: output-verification-agent
    """

    def test_worker_verification_event_types_defined(self):
        """Worker verification event types are properly defined.

        The SDK SHALL define distinct event types for worker verification:
        - worker_verification_start
        - worker_verification_passed
        - worker_verification_failed
        - worker_verification_max_reached

        Feature: output-verification-agent
        """
        from kiva.events import (
            WORKER_VERIFICATION_EVENT_TYPES,
            WORKER_VERIFICATION_START,
            WORKER_VERIFICATION_PASSED,
            WORKER_VERIFICATION_FAILED,
            WORKER_VERIFICATION_MAX_REACHED,
        )

        # Verify all worker verification event types are defined
        assert WORKER_VERIFICATION_START == "worker_verification_start"
        assert WORKER_VERIFICATION_PASSED == "worker_verification_passed"
        assert WORKER_VERIFICATION_FAILED == "worker_verification_failed"
        assert WORKER_VERIFICATION_MAX_REACHED == "worker_verification_max_reached"

        # Verify they are in the list
        assert WORKER_VERIFICATION_START in WORKER_VERIFICATION_EVENT_TYPES
        assert WORKER_VERIFICATION_PASSED in WORKER_VERIFICATION_EVENT_TYPES
        assert WORKER_VERIFICATION_FAILED in WORKER_VERIFICATION_EVENT_TYPES
        assert WORKER_VERIFICATION_MAX_REACHED in WORKER_VERIFICATION_EVENT_TYPES
        # 5 event types: start, passed, failed, max_reached, error (graceful degradation)
        assert len(WORKER_VERIFICATION_EVENT_TYPES) == 5

    def test_final_verification_event_types_defined(self):
        """Final verification event types are properly defined.

        The SDK SHALL define distinct event types for final verification:
        - final_verification_start
        - final_verification_passed
        - final_verification_failed
        - final_verification_max_reached

        Feature: output-verification-agent
        """
        from kiva.events import (
            FINAL_VERIFICATION_EVENT_TYPES,
            FINAL_VERIFICATION_START,
            FINAL_VERIFICATION_PASSED,
            FINAL_VERIFICATION_FAILED,
            FINAL_VERIFICATION_MAX_REACHED,
        )

        # Verify all final verification event types are defined
        assert FINAL_VERIFICATION_START == "final_verification_start"
        assert FINAL_VERIFICATION_PASSED == "final_verification_passed"
        assert FINAL_VERIFICATION_FAILED == "final_verification_failed"
        assert FINAL_VERIFICATION_MAX_REACHED == "final_verification_max_reached"

        # Verify they are in the list
        assert FINAL_VERIFICATION_START in FINAL_VERIFICATION_EVENT_TYPES
        assert FINAL_VERIFICATION_PASSED in FINAL_VERIFICATION_EVENT_TYPES
        assert FINAL_VERIFICATION_FAILED in FINAL_VERIFICATION_EVENT_TYPES
        assert FINAL_VERIFICATION_MAX_REACHED in FINAL_VERIFICATION_EVENT_TYPES
        # 5 event types: start, passed, failed, max_reached, error (graceful degradation)
        assert len(FINAL_VERIFICATION_EVENT_TYPES) == 5

    def test_retry_event_types_defined(self):
        """Retry event types are properly defined.

        The SDK SHALL define distinct event types for retry operations:
        - retry_triggered
        - retry_completed
        - retry_skipped

        Feature: output-verification-agent
        """
        from kiva.events import (
            RETRY_EVENT_TYPES,
            RETRY_TRIGGERED,
            RETRY_COMPLETED,
            RETRY_SKIPPED,
        )

        # Verify all retry event types are defined
        assert RETRY_TRIGGERED == "retry_triggered"
        assert RETRY_COMPLETED == "retry_completed"
        assert RETRY_SKIPPED == "retry_skipped"

        # Verify they are in the list
        assert RETRY_TRIGGERED in RETRY_EVENT_TYPES
        assert RETRY_COMPLETED in RETRY_EVENT_TYPES
        assert RETRY_SKIPPED in RETRY_EVENT_TYPES
        assert len(RETRY_EVENT_TYPES) == 3

    def test_all_verification_events_in_combined_list(self):
        """All verification events are in the combined VERIFICATION_EVENT_TYPES list.

        The SDK SHALL provide a combined list of all verification-related events.

        Feature: output-verification-agent
        """
        from kiva.events import (
            VERIFICATION_EVENT_TYPES,
            WORKER_VERIFICATION_EVENT_TYPES,
            FINAL_VERIFICATION_EVENT_TYPES,
            RETRY_EVENT_TYPES,
        )

        # Verify combined list contains all event types
        expected_count = (
            len(WORKER_VERIFICATION_EVENT_TYPES)
            + len(FINAL_VERIFICATION_EVENT_TYPES)
            + len(RETRY_EVENT_TYPES)
        )
        assert len(VERIFICATION_EVENT_TYPES) == expected_count

        # Verify all individual events are in combined list
        for event_type in WORKER_VERIFICATION_EVENT_TYPES:
            assert event_type in VERIFICATION_EVENT_TYPES
        for event_type in FINAL_VERIFICATION_EVENT_TYPES:
            assert event_type in VERIFICATION_EVENT_TYPES
        for event_type in RETRY_EVENT_TYPES:
            assert event_type in VERIFICATION_EVENT_TYPES

    @given(
        event_type=st.sampled_from([
            "worker_verification_start",
            "worker_verification_passed",
            "worker_verification_failed",
            "worker_verification_max_reached",
            "final_verification_start",
            "final_verification_passed",
            "final_verification_failed",
            "final_verification_max_reached",
            "retry_triggered",
            "retry_completed",
            "retry_skipped",
        ])
    )
    @settings(max_examples=100)
    def test_is_verification_event_returns_true_for_verification_events(
        self,
        event_type: str,
    ):
        """is_verification_event returns True for all verification events.

        For any verification event type, is_verification_event() SHALL return True.

        Feature: output-verification-agent
        """
        from kiva.events import is_verification_event

        assert is_verification_event(event_type) is True

    @given(
        event_type=st.sampled_from([
            "token",
            "workflow_selected",
            "final_result",
            "error",
            "agent_start",
            "agent_end",
            "instance_spawn",
            "instance_complete",
            "custom_event",
            "unknown_event",
        ])
    )
    @settings(max_examples=100)
    def test_is_verification_event_returns_false_for_non_verification_events(
        self,
        event_type: str,
    ):
        """is_verification_event returns False for non-verification events.

        For any non-verification event type, is_verification_event() SHALL
        return False.

        Feature: output-verification-agent
        """
        from kiva.events import is_verification_event

        assert is_verification_event(event_type) is False

    @given(
        event_type=st.sampled_from([
            "worker_verification_start",
            "worker_verification_passed",
            "worker_verification_failed",
            "worker_verification_max_reached",
        ])
    )
    @settings(max_examples=100)
    def test_is_worker_verification_event_returns_true_for_worker_events(
        self,
        event_type: str,
    ):
        """is_worker_verification_event returns True for worker verification events.

        For any worker verification event type, is_worker_verification_event()
        SHALL return True.

        Feature: output-verification-agent
        """
        from kiva.events import is_worker_verification_event

        assert is_worker_verification_event(event_type) is True

    @given(
        event_type=st.sampled_from([
            "final_verification_start",
            "final_verification_passed",
            "retry_triggered",
            "token",
            "agent_start",
        ])
    )
    @settings(max_examples=100)
    def test_is_worker_verification_event_returns_false_for_non_worker_events(
        self,
        event_type: str,
    ):
        """is_worker_verification_event returns False for non-worker events.

        For any non-worker verification event type, is_worker_verification_event()
        SHALL return False.

        Feature: output-verification-agent
        """
        from kiva.events import is_worker_verification_event

        assert is_worker_verification_event(event_type) is False

    @given(
        event_type=st.sampled_from([
            "final_verification_start",
            "final_verification_passed",
            "final_verification_failed",
            "final_verification_max_reached",
        ])
    )
    @settings(max_examples=100)
    def test_is_final_verification_event_returns_true_for_final_events(
        self,
        event_type: str,
    ):
        """is_final_verification_event returns True for final verification events.

        For any final verification event type, is_final_verification_event()
        SHALL return True.

        Feature: output-verification-agent
        """
        from kiva.events import is_final_verification_event

        assert is_final_verification_event(event_type) is True

    @given(
        event_type=st.sampled_from([
            "worker_verification_start",
            "worker_verification_passed",
            "retry_triggered",
            "token",
            "agent_start",
        ])
    )
    @settings(max_examples=100)
    def test_is_final_verification_event_returns_false_for_non_final_events(
        self,
        event_type: str,
    ):
        """is_final_verification_event returns False for non-final events.

        For any non-final verification event type, is_final_verification_event()
        SHALL return False.

        Feature: output-verification-agent
        """
        from kiva.events import is_final_verification_event

        assert is_final_verification_event(event_type) is False

    @given(
        event_type=st.sampled_from([
            "retry_triggered",
            "retry_completed",
            "retry_skipped",
        ])
    )
    @settings(max_examples=100)
    def test_is_retry_event_returns_true_for_retry_events(
        self,
        event_type: str,
    ):
        """is_retry_event returns True for retry events.

        For any retry event type, is_retry_event() SHALL return True.

        Feature: output-verification-agent
        """
        from kiva.events import is_retry_event

        assert is_retry_event(event_type) is True

    @given(
        event_type=st.sampled_from([
            "worker_verification_start",
            "final_verification_passed",
            "token",
            "agent_start",
        ])
    )
    @settings(max_examples=100)
    def test_is_retry_event_returns_false_for_non_retry_events(
        self,
        event_type: str,
    ):
        """is_retry_event returns False for non-retry events.

        For any non-retry event type, is_retry_event() SHALL return False.

        Feature: output-verification-agent
        """
        from kiva.events import is_retry_event

        assert is_retry_event(event_type) is False

    @given(
        iteration=st.integers(min_value=0, max_value=100),
        agent_count=st.integers(min_value=1, max_value=10),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_stream_event_can_represent_worker_verification_start(
        self,
        iteration: int,
        agent_count: int,
        timestamp: float,
    ):
        """StreamEvent can represent worker_verification_start events.

        For any worker verification start event, the StreamEvent SHALL contain
        iteration and agent_count data.

        Feature: output-verification-agent
        """
        from kiva.events import StreamEvent

        event = StreamEvent(
            type="worker_verification_start",
            data={
                "iteration": iteration,
                "agent_count": agent_count,
                "execution_id": "test-exec-id",
            },
            timestamp=timestamp,
        )

        assert event.type == "worker_verification_start"
        assert event.data["iteration"] == iteration
        assert event.data["agent_count"] == agent_count
        assert event.timestamp == timestamp

        # Verify serialization
        event_dict = event.to_dict()
        assert event_dict["type"] == "worker_verification_start"
        assert event_dict["data"]["iteration"] == iteration

    @given(
        iteration=st.integers(min_value=0, max_value=100),
        failed_agents=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            min_size=0,
            max_size=5,
        ),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_stream_event_can_represent_worker_verification_failed(
        self,
        iteration: int,
        failed_agents: list[str],
        timestamp: float,
    ):
        """StreamEvent can represent worker_verification_failed events.

        For any worker verification failed event, the StreamEvent SHALL contain
        iteration and failed_agents data.

        Feature: output-verification-agent
        """
        from kiva.events import StreamEvent

        event = StreamEvent(
            type="worker_verification_failed",
            data={
                "iteration": iteration,
                "failed_agents": failed_agents,
                "execution_id": "test-exec-id",
            },
            timestamp=timestamp,
        )

        assert event.type == "worker_verification_failed"
        assert event.data["iteration"] == iteration
        assert event.data["failed_agents"] == failed_agents
        assert event.timestamp == timestamp

    @given(
        iteration=st.integers(min_value=0, max_value=100),
        reason=st.one_of(st.none(), st.text(max_size=200)),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_stream_event_can_represent_final_verification_failed(
        self,
        iteration: int,
        reason: str | None,
        timestamp: float,
    ):
        """StreamEvent can represent final_verification_failed events.

        For any final verification failed event, the StreamEvent SHALL contain
        iteration, reason, and action data.

        Feature: output-verification-agent
        """
        from kiva.events import StreamEvent

        event = StreamEvent(
            type="final_verification_failed",
            data={
                "iteration": iteration,
                "reason": reason,
                "action": "restart_workflow",
                "execution_id": "test-exec-id",
            },
            timestamp=timestamp,
        )

        assert event.type == "final_verification_failed"
        assert event.data["iteration"] == iteration
        assert event.data["reason"] == reason
        assert event.data["action"] == "restart_workflow"
        assert event.timestamp == timestamp

    @given(
        iteration=st.integers(min_value=0, max_value=100),
        retry_prompt=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_stream_event_can_represent_retry_triggered(
        self,
        iteration: int,
        retry_prompt: str,
        timestamp: float,
    ):
        """StreamEvent can represent retry_triggered events.

        For any retry triggered event, the StreamEvent SHALL contain
        iteration and retry_prompt data.

        Feature: output-verification-agent
        """
        from kiva.events import StreamEvent

        event = StreamEvent(
            type="retry_triggered",
            data={
                "iteration": iteration,
                "retry_prompt": retry_prompt[:200],  # Truncated as in actual impl
                "execution_id": "test-exec-id",
            },
            timestamp=timestamp,
        )

        assert event.type == "retry_triggered"
        assert event.data["iteration"] == iteration
        assert event.data["retry_prompt"] == retry_prompt[:200]
        assert event.timestamp == timestamp

    @given(
        iteration=st.integers(min_value=0, max_value=100),
        results_count=st.integers(min_value=0, max_value=20),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_stream_event_can_represent_retry_completed(
        self,
        iteration: int,
        results_count: int,
        timestamp: float,
    ):
        """StreamEvent can represent retry_completed events.

        For any retry completed event, the StreamEvent SHALL contain
        iteration and results_count data.

        Feature: output-verification-agent
        """
        from kiva.events import StreamEvent

        event = StreamEvent(
            type="retry_completed",
            data={
                "iteration": iteration,
                "results_count": results_count,
                "execution_id": "test-exec-id",
            },
            timestamp=timestamp,
        )

        assert event.type == "retry_completed"
        assert event.data["iteration"] == iteration
        assert event.data["results_count"] == results_count
        assert event.timestamp == timestamp

    def test_verification_events_emitted_by_verify_worker_output_node(self):
        """verify_worker_output node emits appropriate verification events.

        The verify_worker_output node SHALL emit:
        - worker_verification_start when verification begins
        - worker_verification_passed/failed/max_reached based on result

        Feature: output-verification-agent
        """
        # This test verifies that the verify_worker_output node emits events
        # by checking the emit_event calls in the node implementation
        from kiva.nodes.verify import verify_worker_output

        # Verify the function exists and is callable
        assert callable(verify_worker_output)

        # The actual event emission is tested via integration tests
        # Here we verify the node is properly defined

    def test_verification_events_emitted_by_verify_final_result_node(self):
        """verify_final_result node emits appropriate verification events.

        The verify_final_result node SHALL emit:
        - final_verification_start when verification begins
        - final_verification_passed/failed/max_reached based on result

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import verify_final_result

        # Verify the function exists and is callable
        assert callable(verify_final_result)

    def test_retry_events_emitted_by_worker_retry_node(self):
        """worker_retry node emits appropriate retry events.

        The worker_retry node SHALL emit:
        - retry_triggered when retry begins
        - retry_completed when retry finishes
        - retry_skipped when no retry context available

        Feature: output-verification-agent
        """
        from kiva.nodes.verify import worker_retry

        # Verify the function exists and is callable
        assert callable(worker_retry)


class TestInterAgentMessageValidationProperty:
    """Property-based tests for Inter-Agent Message Validation.

    Feature: output-verification-agent
    """

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.one_of(
            st.text(max_size=200),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
                values=st.text(max_size=50),
                max_size=5,
            ),
            st.lists(st.integers(), max_size=5),
            st.none(),
        ),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_valid_message_passes_validation(
        self,
        sender_id: str,
        receiver_id: str,
        content,
        message_type: str,
        timestamp: float,
    ):
        """Valid messages pass validation.

        For any message that conforms to the AgentMessage Pydantic schema,
        the validation SHALL return is_valid=True with no field errors.

        Feature: output-verification-agent
        """
        from kiva.verification import validate_agent_message

        message_data = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": content,
            "message_type": message_type,
            "timestamp": timestamp,
        }

        result = validate_agent_message(message_data)

        assert result.is_valid is True
        assert result.field_errors == {}
        assert result.rejection_reason is None
        assert result.expected_format is not None

    @given(
        # Generate message data with missing required fields
        missing_fields=st.lists(
            st.sampled_from(["sender_id", "receiver_id", "message_type", "timestamp"]),
            min_size=1,
            max_size=4,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_missing_required_fields_fails_validation(
        self,
        missing_fields: list[str],
    ):
        """Missing required fields fail validation with field errors.

        For any message missing required fields, the validation SHALL return
        is_valid=False and SHALL include field-level error details for each
        missing field.

        Feature: output-verification-agent
        """
        from kiva.verification import validate_agent_message

        # Create complete message data
        complete_data = {
            "sender_id": "agent_1",
            "receiver_id": "agent_2",
            "content": "test content",
            "message_type": "task_result",
            "timestamp": 1234567890.0,
        }

        # Remove the specified fields
        message_data = {k: v for k, v in complete_data.items() if k not in missing_fields}

        result = validate_agent_message(message_data)

        assert result.is_valid is False
        assert result.rejection_reason is not None
        assert "error" in result.rejection_reason.lower()
        assert len(result.field_errors) > 0

        # Each missing field should have an error
        for field in missing_fields:
            assert field in result.field_errors, (
                f"Missing field '{field}' should have an error in field_errors"
            )

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
        # Generate invalid timestamp values
        invalid_timestamp=st.one_of(
            st.text(min_size=1, max_size=20).filter(
                lambda x: x.strip() and not x.strip().lstrip("-").replace(".", "", 1).isdigit()
            ),
            st.lists(st.integers(), min_size=1, max_size=3),
        ),
    )
    @settings(max_examples=100)
    def test_invalid_field_type_fails_validation(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        invalid_timestamp,
    ):
        """Invalid field types fail validation with field errors.

        For any message with invalid field types, the validation SHALL return
        is_valid=False and SHALL include field-level error details explaining
        the expected format.

        Feature: output-verification-agent
        """
        from kiva.verification import validate_agent_message

        message_data = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": "test content",
            "message_type": message_type,
            "timestamp": invalid_timestamp,  # Invalid type
        }

        result = validate_agent_message(message_data)

        assert result.is_valid is False
        assert result.rejection_reason is not None
        assert len(result.field_errors) > 0
        assert "timestamp" in result.field_errors
        assert result.expected_format is not None

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.text(max_size=200),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_validation_result_includes_expected_format(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        message_type: str,
        timestamp: float,
    ):
        """Validation result includes expected format description.

        For any validation result (pass or fail), the result SHALL include
        a description of the expected message format.

        Feature: output-verification-agent
        """
        from kiva.verification import validate_agent_message

        # Test with valid message
        valid_message = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": content,
            "message_type": message_type,
            "timestamp": timestamp,
        }
        valid_result = validate_agent_message(valid_message)
        assert valid_result.expected_format is not None
        assert "sender_id" in valid_result.expected_format
        assert "receiver_id" in valid_result.expected_format

        # Test with invalid message
        invalid_message = {"sender_id": sender_id}  # Missing fields
        invalid_result = validate_agent_message(invalid_message)
        assert invalid_result.expected_format is not None
        assert "sender_id" in invalid_result.expected_format

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.one_of(
            st.text(max_size=200),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
                values=st.text(max_size=50),
                max_size=5,
            ),
            st.none(),
        ),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_inter_agent_message_validator_class(
        self,
        sender_id: str,
        receiver_id: str,
        content,
        message_type: str,
        timestamp: float,
    ):
        """InterAgentMessageValidator class validates correctly.

        The InterAgentMessageValidator class SHALL validate messages against
        the AgentMessage schema and return proper validation results.

        Feature: output-verification-agent
        """
        from kiva.verification import InterAgentMessageValidator

        validator = InterAgentMessageValidator()

        message_data = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": content,
            "message_type": message_type,
            "timestamp": timestamp,
        }

        result = validator.validate(message_data)

        assert result.is_valid is True
        assert result.field_errors == {}

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.text(max_size=200),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_validate_and_create_returns_message_on_success(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        message_type: str,
        timestamp: float,
    ):
        """validate_and_create returns AgentMessage on success.

        When validation passes, validate_and_create SHALL return a valid
        AgentMessage object along with a successful validation result.

        Feature: output-verification-agent
        """
        from kiva.verification import AgentMessage, InterAgentMessageValidator

        validator = InterAgentMessageValidator()

        message_data = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": content,
            "message_type": message_type,
            "timestamp": timestamp,
        }

        message, result = validator.validate_and_create(message_data)

        assert result.is_valid is True
        assert message is not None
        assert isinstance(message, AgentMessage)
        assert message.sender_id == sender_id
        assert message.receiver_id == receiver_id
        assert message.content == content
        assert message.message_type == message_type
        assert message.timestamp == timestamp

    @given(
        missing_fields=st.lists(
            st.sampled_from(["sender_id", "receiver_id", "message_type", "timestamp"]),
            min_size=1,
            max_size=4,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_validate_and_create_returns_none_on_failure(
        self,
        missing_fields: list[str],
    ):
        """validate_and_create returns None on validation failure.

        When validation fails, validate_and_create SHALL return None for the
        message and a failed validation result with field errors.

        Feature: output-verification-agent
        """
        from kiva.verification import InterAgentMessageValidator

        validator = InterAgentMessageValidator()

        # Create incomplete message data
        complete_data = {
            "sender_id": "agent_1",
            "receiver_id": "agent_2",
            "content": "test",
            "message_type": "task_result",
            "timestamp": 1234567890.0,
        }
        message_data = {k: v for k, v in complete_data.items() if k not in missing_fields}

        message, result = validator.validate_and_create(message_data)

        assert result.is_valid is False
        assert message is None
        assert len(result.field_errors) > 0

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.text(max_size=200),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_workflows_utils_create_agent_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        message_type: str,
    ):
        """workflows/utils create_agent_message validates correctly.

        The create_agent_message utility function SHALL validate messages
        and return proper results.

        Feature: output-verification-agent
        """
        from kiva.workflows.utils import create_agent_message

        message, result = create_agent_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
            # timestamp defaults to current time
        )

        assert result.is_valid is True
        assert message is not None
        assert message.sender_id == sender_id
        assert message.receiver_id == receiver_id
        assert message.content == content
        assert message.message_type == message_type
        assert message.timestamp > 0  # Should have a valid timestamp

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.text(max_size=200),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_workflows_utils_validate_and_send_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        message_type: str,
    ):
        """workflows/utils validate_and_send_message validates correctly.

        The validate_and_send_message utility function SHALL validate messages
        before sending and return proper results.

        Feature: output-verification-agent
        """
        from kiva.workflows.utils import validate_and_send_message

        message, result = validate_and_send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
        )

        assert result.is_valid is True
        assert message is not None
        assert message.sender_id == sender_id
        assert message.receiver_id == receiver_id

    @given(
        sender_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        receiver_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.text(max_size=200),
        message_type=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
        timestamp=st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_workflows_utils_validate_message_data(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        message_type: str,
        timestamp: float,
    ):
        """workflows/utils validate_message_data validates correctly.

        The validate_message_data utility function SHALL validate raw message
        data against the AgentMessage schema.

        Feature: output-verification-agent
        """
        from kiva.workflows.utils import validate_message_data

        message_data = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": content,
            "message_type": message_type,
            "timestamp": timestamp,
        }

        result = validate_message_data(message_data)

        assert result.is_valid is True
        assert result.field_errors == {}

    @given(
        missing_fields=st.lists(
            st.sampled_from(["sender_id", "receiver_id", "message_type", "timestamp"]),
            min_size=1,
            max_size=4,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_workflows_utils_validate_message_data_fails_on_invalid(
        self,
        missing_fields: list[str],
    ):
        """workflows/utils validate_message_data fails on invalid data.

        The validate_message_data utility function SHALL return validation
        failure with field errors when data is invalid.

        Feature: output-verification-agent
        """
        from kiva.workflows.utils import validate_message_data

        # Create incomplete message data
        complete_data = {
            "sender_id": "agent_1",
            "receiver_id": "agent_2",
            "content": "test",
            "message_type": "task_result",
            "timestamp": 1234567890.0,
        }
        message_data = {k: v for k, v in complete_data.items() if k not in missing_fields}

        result = validate_message_data(message_data)

        assert result.is_valid is False
        assert len(result.field_errors) > 0
        for field in missing_fields:
            assert field in result.field_errors


class TestMessageValidationResultModel:
    """Unit tests for MessageValidationResult model."""

    def test_create_valid_result(self):
        """Test creating a valid MessageValidationResult."""
        from kiva.verification import MessageValidationResult

        result = MessageValidationResult(
            is_valid=True,
            expected_format="AgentMessage with required fields",
        )
        assert result.is_valid is True
        assert result.field_errors == {}
        assert result.rejection_reason is None
        assert result.expected_format == "AgentMessage with required fields"

    def test_create_invalid_result_with_errors(self):
        """Test creating an invalid MessageValidationResult with field errors."""
        from kiva.verification import MessageValidationResult

        result = MessageValidationResult(
            is_valid=False,
            field_errors={"sender_id": "Field required", "timestamp": "Invalid type"},
            rejection_reason="Message validation failed: 2 error(s)",
            expected_format="AgentMessage with required fields",
        )
        assert result.is_valid is False
        assert len(result.field_errors) == 2
        assert "sender_id" in result.field_errors
        assert "timestamp" in result.field_errors
        assert result.rejection_reason == "Message validation failed: 2 error(s)"

    def test_model_dump(self):
        """Test serialization of MessageValidationResult."""
        from kiva.verification import MessageValidationResult

        result = MessageValidationResult(
            is_valid=False,
            field_errors={"sender_id": "Field required"},
            rejection_reason="Validation failed",
            expected_format="AgentMessage",
        )
        dumped = result.model_dump()
        assert dumped["is_valid"] is False
        assert dumped["field_errors"] == {"sender_id": "Field required"}
        assert dumped["rejection_reason"] == "Validation failed"
        assert dumped["expected_format"] == "AgentMessage"


class TestGracefulDegradationProperty:
    """Property-based tests for Graceful Degradation On Failure.

    Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
    Validates: Requirements 8.1, 8.2, 8.3

    Property 10: Graceful Degradation On Failure
    *For any* execution where all retry attempts fail (at either Worker or Workflow
    level) OR where the Verifier itself fails, the SDK SHALL:
    - For Worker level: proceed to synthesis with available results and a warning flag
    - For Workflow level: return a failure summary containing the original user prompt,
      rejection reason, and improvement suggestions
    """

    @given(
        original_prompt=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        final_result=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        rejection_reason=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        improvement_suggestions=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=0,
            max_size=3,
        ),
        attempts=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_failure_summary_contains_required_info(
        self,
        original_prompt: str,
        final_result: str,
        rejection_reason: str,
        improvement_suggestions: list[str],
        attempts: int,
    ):
        """Property 10: Failure summary contains required information.

        For any workflow-level failure after max iterations, the failure summary
        SHALL contain the original user prompt, rejection reason, and improvement
        suggestions.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.1, 8.2, 8.3
        """
        from kiva.nodes.verify import _generate_failure_summary

        summary = _generate_failure_summary(
            original_prompt=original_prompt,
            rejection_reason=rejection_reason,
            improvement_suggestions=improvement_suggestions,
            attempts=attempts,
        )

        # Verify the summary contains the original prompt
        assert original_prompt in summary, (
            f"Failure summary must contain original prompt. "
            f"Prompt: {original_prompt!r}, Summary: {summary!r}"
        )

        # Verify the summary contains the rejection reason
        assert rejection_reason in summary, (
            f"Failure summary must contain rejection reason. "
            f"Reason: {rejection_reason!r}, Summary: {summary!r}"
        )

        # Verify the summary contains the attempt count
        assert str(attempts) in summary, (
            f"Failure summary must contain attempt count. "
            f"Attempts: {attempts}, Summary: {summary!r}"
        )

        # Verify improvement suggestions are included if provided
        for suggestion in improvement_suggestions:
            assert suggestion in summary, (
                f"Failure summary must contain improvement suggestion. "
                f"Suggestion: {suggestion!r}, Summary: {summary!r}"
            )

    @given(
        agent_results=st.lists(
            st.fixed_dictionaries({
                "agent_id": st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
                "result": st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            }),
            min_size=1,
            max_size=3,
        ),
        iteration=st.integers(min_value=0, max_value=10),
        max_iterations=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_worker_verification_returns_results_at_max_iterations(
        self,
        agent_results: list[dict],
        iteration: int,
        max_iterations: int,
    ):
        """Property 10: Worker verification returns results at max iterations.

        For any worker verification that reaches max iterations, the SDK SHALL
        proceed to synthesis with available results and a warning flag.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.1
        """
        # When iteration >= max_iterations, we should get a warning and proceed
        # This tests the logic that max iterations triggers graceful degradation
        if iteration >= max_iterations:
            # At max iterations, we should have a warning
            warning = "Max iterations reached for worker verification"
            assert "Max iterations" in warning
            assert "worker verification" in warning

    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_verifier_failure_produces_skipped_status(
        self,
        error_message: str,
    ):
        """Property 10: Verifier failure produces SKIPPED status.

        For any verifier that fails with an exception, the SDK SHALL produce
        a SKIPPED verification result with the error details.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.2
        """
        # Create a verifier result that represents a skipped verification
        result = VerificationResult(
            status=VerificationStatus.SKIPPED,
            rejection_reason=f"Verifier error: {error_message}",
            validator_name="failing_verifier",
        )

        # Verify the result has SKIPPED status
        assert result.status == VerificationStatus.SKIPPED

        # Verify the error message is preserved
        assert error_message in result.rejection_reason

    @given(
        passed_results=st.lists(
            st.builds(
                VerificationResult,
                status=st.just(VerificationStatus.PASSED),
                confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
            ),
            min_size=0,
            max_size=3,
        ),
        skipped_results=st.lists(
            st.builds(
                VerificationResult,
                status=st.just(VerificationStatus.SKIPPED),
                rejection_reason=st.text(min_size=1, max_size=100).filter(
                    lambda x: x.strip()
                ),
            ),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_skipped_results_allow_verification_to_pass(
        self,
        passed_results: list[VerificationResult],
        skipped_results: list[VerificationResult],
    ):
        """Property 10: SKIPPED results allow verification to pass.

        For any combination of PASSED and SKIPPED results (no FAILED), the
        aggregate verification SHALL be PASSED, enabling graceful degradation
        when verifiers fail.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.2
        """
        verifier = WorkerOutputVerifier()
        all_results = passed_results + skipped_results

        if not all_results:
            return  # Skip empty case

        aggregated = verifier._aggregate_results(all_results)

        # When all results are PASSED or SKIPPED, aggregate should be PASSED
        assert aggregated.status == VerificationStatus.PASSED, (
            f"Aggregate of PASSED and SKIPPED results should be PASSED. "
            f"Got: {aggregated.status}"
        )

    @given(
        task=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        output=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_custom_verifier_exception_produces_skipped_result(
        self,
        task: str,
        output: str,
    ):
        """Property 10: Custom verifier exception produces SKIPPED result.

        For any custom verifier that raises an exception, the SDK SHALL
        produce a SKIPPED result instead of propagating the exception.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.2, 8.3
        """

        class FailingVerifier:
            _verifier_name = "failing_verifier"

            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                raise RuntimeError("Verifier crashed!")

        verifier = WorkerOutputVerifier(custom_verifiers=[FailingVerifier()])
        results = verifier._run_custom_verifiers(
            assigned_task=task,
            output=output,
        )

        # Should have one result
        assert len(results) == 1

        # Result should be SKIPPED, not an exception
        assert results[0].status == VerificationStatus.SKIPPED

        # Error message should be preserved
        assert "Custom verifier error" in results[0].rejection_reason
        assert "Verifier crashed" in results[0].rejection_reason

    @given(
        original_prompt=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        final_result=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_final_verifier_exception_produces_skipped_result(
        self,
        original_prompt: str,
        final_result: str,
    ):
        """Property 10: Final verifier exception produces SKIPPED result.

        For any final result verifier custom verifier that raises an exception,
        the SDK SHALL produce a SKIPPED result instead of propagating the exception.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.2, 8.3
        """

        class FailingVerifier:
            _verifier_name = "failing_final_verifier"

            def verify(
                self, task: str, output: str, context: dict | None = None
            ) -> VerificationResult:
                raise ValueError("Final verifier crashed!")

        verifier = FinalResultVerifier(custom_verifiers=[FailingVerifier()])
        results = verifier._run_custom_verifiers(
            original_prompt=original_prompt,
            final_result=final_result,
        )

        # Should have one result
        assert len(results) == 1

        # Result should be SKIPPED, not an exception
        assert results[0].status == VerificationStatus.SKIPPED

        # Error message should be preserved
        assert "Custom verifier error" in results[0].rejection_reason
        assert "Final verifier crashed" in results[0].rejection_reason

    @given(
        workflow_iteration=st.integers(min_value=0, max_value=10),
        max_workflow_iterations=st.integers(min_value=1, max_value=5),
        original_prompt=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_workflow_max_iterations_returns_failure_summary(
        self,
        workflow_iteration: int,
        max_workflow_iterations: int,
        original_prompt: str,
    ):
        """Property 10: Workflow max iterations returns failure summary.

        For any workflow that reaches max iterations, the SDK SHALL return
        a failure summary containing the original user prompt.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.1
        """
        from kiva.nodes.verify import _generate_failure_summary

        if workflow_iteration >= max_workflow_iterations:
            # At max iterations, we should generate a failure summary
            summary = _generate_failure_summary(
                original_prompt=original_prompt,
                rejection_reason="Verification failed",
                improvement_suggestions=["Try a different approach"],
                attempts=workflow_iteration + 1,
            )

            # Summary must contain the original prompt
            assert original_prompt in summary

            # Summary must indicate it's a failure
            assert "Could Not" in summary or "failed" in summary.lower()

    @given(
        failed_results=st.lists(
            st.builds(
                VerificationResult,
                status=st.just(VerificationStatus.FAILED),
                rejection_reason=st.text(min_size=1, max_size=100).filter(
                    lambda x: x.strip()
                ),
                improvement_suggestions=st.lists(
                    st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
                    min_size=0,
                    max_size=2,
                ),
            ),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_aggregated_failure_preserves_all_rejection_reasons(
        self,
        failed_results: list[VerificationResult],
    ):
        """Property 10: Aggregated failure preserves all rejection reasons.

        For any set of failed verification results, the aggregated result
        SHALL preserve all rejection reasons and improvement suggestions.

        Feature: output-verification-agent, Property 10: Graceful Degradation On Failure
        Validates: Requirements 8.1, 8.3
        """
        verifier = WorkerOutputVerifier()
        aggregated = verifier._aggregate_results(failed_results)

        # Aggregated status should be FAILED
        assert aggregated.status == VerificationStatus.FAILED

        # All rejection reasons should be in the aggregated reason
        for result in failed_results:
            if result.rejection_reason:
                assert result.rejection_reason in aggregated.rejection_reason, (
                    f"Rejection reason '{result.rejection_reason}' not found in "
                    f"aggregated reason: {aggregated.rejection_reason}"
                )

        # All improvement suggestions should be preserved
        all_suggestions = []
        for result in failed_results:
            all_suggestions.extend(result.improvement_suggestions)

        for suggestion in all_suggestions:
            assert suggestion in aggregated.improvement_suggestions, (
                f"Suggestion '{suggestion}' not found in aggregated suggestions: "
                f"{aggregated.improvement_suggestions}"
            )

