"""Tests for verification models and utilities.

This module contains unit tests and property-based tests for the verification
data models defined in kiva.verification.
"""

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from kiva.verification import (
    AgentMessage,
    RetryContext,
    VerificationResult,
    VerificationStatus,
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
