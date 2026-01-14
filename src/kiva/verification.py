"""Verification models and utilities for the Kiva SDK.

This module defines the Pydantic data models used for output verification,
including verification results, retry context, and inter-agent messages.

The verification system verifies Worker Agent outputs against assigned tasks.
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError


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


class VerificationStateCode(str, Enum):
    """Verification lifecycle state codes."""

    INITIALIZING = "initializing"
    PREPROCESSING = "preprocessing"
    VERIFYING = "verifying"
    RETRY_WAITING = "retry_waiting"
    RETRY_RUNNING = "retry_running"
    FAILURE_HANDLING = "failure_handling"
    ROLLBACK = "rollback"
    COMMITTING = "committing"
    COMPLETED = "completed"


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


class MessageValidationResult(BaseModel):
    """Result of inter-agent message validation.

    Contains the outcome of validating an agent message against the
    AgentMessage Pydantic schema.

    Attributes:
        is_valid: Whether the message is valid.
        field_errors: Dictionary of field-level validation errors.
        rejection_reason: Human-readable rejection reason if invalid.
        expected_format: Description of the expected message format.

    Example:
        >>> result = MessageValidationResult(
        ...     is_valid=False,
        ...     field_errors={"sender_id": "Field required"},
        ...     rejection_reason="Message validation failed: 1 error(s)",
        ...     expected_format="AgentMessage with sender_id, receiver_id, ...",
        ... )
    """

    is_valid: bool
    field_errors: dict[str, str] = Field(default_factory=dict)
    rejection_reason: str | None = None
    expected_format: str | None = None


def validate_agent_message(message_data: dict[str, Any]) -> MessageValidationResult:
    """Validate a message against the AgentMessage Pydantic schema.

    This function validates inter-agent communication messages to ensure
    data integrity throughout the workflow. If validation fails, it returns
    detailed field-level error information.

    Args:
        message_data: Dictionary containing the message data to validate.

    Returns:
        MessageValidationResult with validation status and error details.

    Example:
        >>> result = validate_agent_message({
        ...     "sender_id": "agent_1",
        ...     "receiver_id": "agent_2",
        ...     "content": "Hello",
        ...     "message_type": "greeting",
        ...     "timestamp": 1234567890.0,
        ... })
        >>> assert result.is_valid is True

        >>> result = validate_agent_message({"sender_id": "agent_1"})
        >>> assert result.is_valid is False
        >>> assert "receiver_id" in result.field_errors
    """
    expected_format = (
        "AgentMessage with required fields: sender_id (str), receiver_id (str), "
        "content (any), message_type (str), timestamp (float)"
    )

    try:
        AgentMessage.model_validate(message_data)
        return MessageValidationResult(
            is_valid=True,
            expected_format=expected_format,
        )
    except ValidationError as e:
        field_errors: dict[str, str] = {}
        for error in e.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            field_errors[field_path] = error["msg"]

        err_count = len(e.errors())
        return MessageValidationResult(
            is_valid=False,
            field_errors=field_errors,
            rejection_reason=f"Message validation failed: {err_count} error(s)",
            expected_format=expected_format,
        )


class InterAgentMessageValidator:
    """Validator for inter-agent communication messages.

    Provides validation of messages sent between agents, ensuring they
    conform to the AgentMessage Pydantic schema. When validation fails,
    it triggers a rejection with context explaining the validation error.

    This class implements Requirements 4.2, 4.3, 4.4 for inter-agent
    communication validation.

    Example:
        >>> validator = InterAgentMessageValidator()
        >>> result = validator.validate({
        ...     "sender_id": "agent_1",
        ...     "receiver_id": "agent_2",
        ...     "content": {"data": "test"},
        ...     "message_type": "task_result",
        ...     "timestamp": 1234567890.0,
        ... })
        >>> assert result.is_valid is True
    """

    def validate(self, message_data: dict[str, Any]) -> MessageValidationResult:
        """Validate a message against the AgentMessage schema.

        Args:
            message_data: Dictionary containing the message data.

        Returns:
            MessageValidationResult with validation status and error details.
        """
        return validate_agent_message(message_data)

    def validate_and_create(
        self, message_data: dict[str, Any]
    ) -> tuple[AgentMessage | None, MessageValidationResult]:
        """Validate and create an AgentMessage if valid.

        Args:
            message_data: Dictionary containing the message data.

        Returns:
            Tuple of (AgentMessage or None, MessageValidationResult).
            If validation passes, returns the created message and a valid result.
            If validation fails, returns None and the error result.
        """
        result = self.validate(message_data)
        if result.is_valid:
            message = AgentMessage.model_validate(message_data)
            return message, result
        return None, result


@runtime_checkable
class Verifier(Protocol):
    """Verifier protocol for custom verification rules.

    Uses Protocol for structural subtyping, allowing any function or class
    that matches the signature to be used as a verifier.

    Example:
        >>> class MyVerifier:
        ...     def verify(
        ...         self, task: str, output: str, context: dict | None = None
        ...     ) -> VerificationResult:
        ...         if len(output) < 10:
        ...             return VerificationResult(
        ...                 status=VerificationStatus.FAILED,
        ...                 rejection_reason="Output too short",
        ...             )
        ...         return VerificationResult(status=VerificationStatus.PASSED)
    """

    @abstractmethod
    def verify(
        self,
        task: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify the output against the task.

        Args:
            task: The task that was assigned to the worker.
            output: The output produced by the worker.
            context: Optional additional context for verification.

        Returns:
            VerificationResult with status and details.
        """
        ...


class LLMVerificationResult(BaseModel):
    """Structured output model for LLM-based verification.

    Used with ChatOpenAI.with_structured_output() to get consistent
    verification results from the LLM.
    """

    passed: bool = Field(description="Whether the output adequately addresses the task")
    rejection_reason: str | None = Field(
        default=None,
        description="Reason for rejection if verification failed",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving the output",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the verification",
    )


class WorkerOutputVerifier:
    """Worker Agent output verifier.

    Verifies whether a Worker Agent's output adequately completes the
    specific task assigned to it. The verification target is the
    task_assignment.task, NOT the user's original prompt.

    This verifier performs three types of validation:
    1. LLM-based semantic verification - checks if output addresses the task
    2. Pydantic schema validation - validates output structure if schema provided
    3. Custom verifier execution - runs any registered custom verifiers

    Attributes:
        model_name: The LLM model to use for verification.
        api_key: API key for the LLM provider.
        base_url: Base URL for the LLM API.
        custom_verifiers: List of custom verifier instances.

    Example:
        >>> verifier = WorkerOutputVerifier(model_name="gpt-4o")
        >>> result = await verifier.verify(
        ...     assigned_task="Summarize the document",
        ...     output="This document discusses...",
        ... )
        >>> print(result.status)
        VerificationStatus.PASSED
    """

    VERIFICATION_SYSTEM_PROMPT = """You are a verification agent responsible for \
checking if a worker's output adequately completes the assigned task.

Your job is to verify:
1. The output directly addresses the assigned task
2. The output contains sufficient reasoning or evidence
3. The output is complete and not truncated or partial

Be strict but fair. If the output reasonably addresses the task, mark it as passed.
If there are significant gaps or the output doesn't address the task, mark it as \
failed and provide specific improvement suggestions."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        custom_verifiers: list[Verifier] | None = None,
    ):
        """Initialize the WorkerOutputVerifier.

        Args:
            model_name: The LLM model to use for verification.
            api_key: API key for the LLM provider.
            base_url: Base URL for the LLM API.
            custom_verifiers: List of custom verifier instances.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.custom_verifiers = custom_verifiers or []

    def _create_model(self) -> ChatOpenAI:
        """Create a ChatOpenAI instance for verification."""
        kwargs: dict[str, Any] = {"model": self.model_name}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    def _validate_pydantic_schema(
        self,
        output: str,
        schema: type[BaseModel],
    ) -> VerificationResult:
        """Validate output against a Pydantic schema.

        Args:
            output: The output string to validate (expected to be JSON).
            schema: The Pydantic model class to validate against.

        Returns:
            VerificationResult with field-level errors if validation fails.
        """
        import json

        try:
            # Try to parse as JSON first
            try:
                data = json.loads(output)
            except json.JSONDecodeError as e:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    rejection_reason=f"Output is not valid JSON: {e}",
                    field_errors={"_json": str(e)},
                    validator_name="pydantic_schema",
                )

            # Validate against schema
            schema.model_validate(data)
            return VerificationResult(
                status=VerificationStatus.PASSED,
                validator_name="pydantic_schema",
            )

        except ValidationError as e:
            field_errors = {}
            suggestions = []
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                field_errors[field_path] = error["msg"]
                suggestions.append(f"Fix field '{field_path}': {error['msg']}")

            err_count = len(e.errors())
            return VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason=f"Pydantic validation failed: {err_count} error(s)",
                improvement_suggestions=suggestions,
                field_errors=field_errors,
                validator_name="pydantic_schema",
            )

    async def _verify_with_llm(
        self,
        assigned_task: str,
        output: str,
    ) -> VerificationResult:
        """Verify output using LLM-based semantic analysis.

        Args:
            assigned_task: The task that was assigned to the worker.
            output: The output produced by the worker.

        Returns:
            VerificationResult from LLM analysis.
        """
        model = self._create_model()
        structured_model = model.with_structured_output(LLMVerificationResult)

        messages = [
            SystemMessage(content=self.VERIFICATION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""Please verify if the following output adequately \
completes the assigned task.

ASSIGNED TASK:
{assigned_task}

WORKER OUTPUT:
{output}

Analyze whether the output addresses the task requirements and provide your \
verification result."""
            ),
        ]

        try:
            result = await structured_model.ainvoke(messages)
            status = (
                VerificationStatus.PASSED
                if result.passed
                else VerificationStatus.FAILED
            )
            return VerificationResult(
                status=status,
                rejection_reason=result.rejection_reason,
                improvement_suggestions=result.improvement_suggestions,
                confidence=result.confidence,
                validator_name="llm_verifier",
            )
        except Exception as e:
            # If LLM verification fails, skip with warning
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                rejection_reason=f"LLM verification failed: {e}",
                validator_name="llm_verifier",
            )

    def _run_custom_verifiers(
        self,
        assigned_task: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> list[VerificationResult]:
        """Run all custom verifiers.

        Args:
            assigned_task: The task that was assigned to the worker.
            output: The output produced by the worker.
            context: Optional additional context.

        Returns:
            List of VerificationResults from custom verifiers.
        """
        results = []
        for verifier in self.custom_verifiers:
            try:
                result = verifier.verify(assigned_task, output, context)
                results.append(result)
            except Exception as e:
                # If a custom verifier fails, add a skipped result
                results.append(
                    VerificationResult(
                        status=VerificationStatus.SKIPPED,
                        rejection_reason=f"Custom verifier error: {e}",
                        validator_name=getattr(verifier, "_verifier_name", "custom"),
                    )
                )
        return results

    def _aggregate_results(
        self,
        results: list[VerificationResult],
    ) -> VerificationResult:
        """Aggregate multiple verification results into a single result.

        The aggregation logic:
        - If any result is FAILED, the aggregate is FAILED
        - If all results are PASSED or SKIPPED, the aggregate is PASSED
        - Rejection reasons and suggestions are combined

        Args:
            results: List of VerificationResults to aggregate.

        Returns:
            Single aggregated VerificationResult.
        """
        if not results:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                validator_name="aggregate",
            )

        # Check for any failures
        failed_results = [r for r in results if r.status == VerificationStatus.FAILED]

        if failed_results:
            # Combine all rejection reasons and suggestions
            rejection_reasons = [
                r.rejection_reason for r in failed_results if r.rejection_reason
            ]
            all_suggestions = []
            all_field_errors = {}

            for r in failed_results:
                all_suggestions.extend(r.improvement_suggestions)
                all_field_errors.update(r.field_errors)

            # Calculate average confidence from non-skipped results
            non_skipped = [r for r in results if r.status != VerificationStatus.SKIPPED]
            avg_confidence = (
                sum(r.confidence for r in non_skipped) / len(non_skipped)
                if non_skipped
                else 1.0
            )

            return VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason=(
                    "; ".join(rejection_reasons) if rejection_reasons else None
                ),
                improvement_suggestions=all_suggestions,
                field_errors=all_field_errors,
                confidence=avg_confidence,
                validator_name="aggregate",
            )

        # All passed or skipped
        non_skipped = [r for r in results if r.status != VerificationStatus.SKIPPED]
        avg_confidence = (
            sum(r.confidence for r in non_skipped) / len(non_skipped)
            if non_skipped
            else 1.0
        )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            confidence=avg_confidence,
            validator_name="aggregate",
        )

    async def verify(
        self,
        assigned_task: str,
        output: str,
        schema: type[BaseModel] | None = None,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify Worker output against the assigned task.

        Performs verification in the following order:
        1. Pydantic schema validation (if schema provided)
        2. LLM-based semantic verification
        3. Custom verifier execution

        All results are aggregated into a single VerificationResult.

        Args:
            assigned_task: The specific task assigned to the Worker Agent.
                This is NOT the user's original prompt, but the task from
                task_assignment.task.
            output: The output produced by the Worker Agent.
            schema: Optional Pydantic model class for structure validation.
            context: Optional additional context for custom verifiers.

        Returns:
            Aggregated VerificationResult with status and details.

        Example:
            >>> verifier = WorkerOutputVerifier()
            >>> result = await verifier.verify(
            ...     assigned_task="Calculate the sum of 2 and 3",
            ...     output="The sum of 2 and 3 is 5.",
            ... )
            >>> assert result.status == VerificationStatus.PASSED
        """
        results: list[VerificationResult] = []

        # 1. Pydantic schema validation (if provided)
        if schema is not None:
            schema_result = self._validate_pydantic_schema(output, schema)
            results.append(schema_result)
            # If schema validation fails, we still continue with other validations
            # but the aggregate will be FAILED

        # 2. LLM-based semantic verification
        llm_result = await self._verify_with_llm(assigned_task, output)
        results.append(llm_result)

        # 3. Custom verifier execution
        custom_results = self._run_custom_verifiers(assigned_task, output, context)
        results.extend(custom_results)

        # Aggregate all results
        return self._aggregate_results(results)
