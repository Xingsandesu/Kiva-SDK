"""Verification models and utilities for the Kiva SDK.

This module defines the Pydantic data models used for output verification,
including verification results, retry context, and inter-agent messages.

The verification system implements a dual-layer architecture:
1. Worker Agent output verification - validates against assigned tasks
2. Final result verification - validates against user's original prompt
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """Verification status enumeration.

    Attributes:
        PASSED: Verification succeeded, output meets requirements.
        FAILED: Verification failed, output does not meet requirements.
        SKIPPED: Verification was skipped (e.g., verifier error).
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VerificationResult(BaseModel):
    """Verification result model.

    Contains the outcome of a verification operation, including status,
    rejection reasons, and improvement suggestions.

    Attributes:
        status: The verification status (passed/failed/skipped).
        rejection_reason: Reason for rejection if verification failed.
        improvement_suggestions: List of suggestions for improving the output.
        field_errors: Dictionary of field-level Pydantic validation errors.
        validator_name: Name of the validator that produced this result.
        confidence: Confidence score of the verification (0.0 to 1.0).

    Example:
        >>> result = VerificationResult(
        ...     status=VerificationStatus.FAILED,
        ...     rejection_reason="Output does not address the task",
        ...     improvement_suggestions=["Include more specific details"],
        ... )
    """

    status: VerificationStatus
    rejection_reason: str | None = None
    improvement_suggestions: list[str] = Field(default_factory=list)
    field_errors: dict[str, str] = Field(default_factory=dict)
    validator_name: str = "default"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RetryContext(BaseModel):
    """Retry context model.

    Encapsulates all context information needed for retry operations,
    ensuring Worker Agents have complete history information.

    Attributes:
        iteration: Current retry iteration number.
        max_iterations: Maximum allowed iterations.
        previous_outputs: List of outputs from previous attempts.
        previous_rejections: List of verification results from previous attempts
            (VerificationResult objects).
        task_history: History of task assignments.
        original_task: The original task assigned to the worker.

    Example:
        >>> context = RetryContext(
        ...     iteration=2,
        ...     max_iterations=3,
        ...     original_task="Analyze the data",
        ...     previous_outputs=["First attempt output"],
        ...     previous_rejections=[
        ...         VerificationResult(status=VerificationStatus.FAILED)
        ...     ],
        ... )
    """

    iteration: int
    max_iterations: int
    previous_outputs: list[str] = Field(default_factory=list)
    previous_rejections: list[VerificationResult] = Field(default_factory=list)
    task_history: list[dict[str, Any]] = Field(default_factory=list)
    original_task: str


class AgentMessage(BaseModel):
    """Agent inter-communication message model.

    Base model for all inter-agent communication, ensuring data integrity
    and type safety through Pydantic validation.

    Attributes:
        sender_id: Identifier of the sending agent.
        receiver_id: Identifier of the receiving agent.
        content: Message content (can be any type).
        message_type: Type/category of the message.
        timestamp: Unix timestamp when the message was created.

    Example:
        >>> message = AgentMessage(
        ...     sender_id="agent_1",
        ...     receiver_id="agent_2",
        ...     content={"result": "analysis complete"},
        ...     message_type="task_result",
        ...     timestamp=1234567890.0,
        ... )
    """

    sender_id: str
    receiver_id: str
    content: Any
    message_type: str
    timestamp: float
