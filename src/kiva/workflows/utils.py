"""Shared utility functions for workflow implementations.

This module provides common helper functions used across different workflow
types (router, supervisor, parliament) to avoid code duplication and ensure
consistent behavior. Includes support for agent instance management and
inter-agent message validation.

Functions:
    get_agent_by_id: Locate an agent instance by its identifier.
    generate_invocation_id: Create unique IDs for agent invocations.
    generate_instance_id: Create unique IDs for agent instances.
    emit_event: Send streaming events with proper error handling.
    extract_content: Parse agent response content from various formats.
    execute_single_agent: Run a single agent with event emission.
    execute_agent_instance: Run an agent instance with isolated context.
    make_error_result: Create standardized error result dictionaries.
    create_instance_context: Create isolated context for an agent instance.
    create_agent_message: Create and validate an inter-agent message.
    validate_and_send_message: Validate a message before sending between agents.
"""

import logging
import time
import uuid
from typing import Any

from langgraph.config import get_stream_writer

from kiva.exceptions import wrap_agent_error
from kiva.verification import (
    AgentMessage,
    InterAgentMessageValidator,
    MessageValidationResult,
)

logger = logging.getLogger(__name__)


def get_agent_by_id(
    agents: list, agent_id: str, fallback_first: bool = False
) -> Any | None:
    """Find an agent by its ID from the agents list.

    Searches for an agent matching the given ID using either the agent's
    name attribute or a positional index format (e.g., "agent_0").

    Args:
        agents: List of agent instances to search through.
        agent_id: The identifier to match against agent names or indices.
        fallback_first: If True, return the first agent when no match found.

    Returns:
        The matching agent instance, or None if not found.

    Examples:
        >>> agent = get_agent_by_id(agents, "researcher")
        >>> agent = get_agent_by_id(agents, "agent_0")
        >>> agent = get_agent_by_id(agents, "unknown", fallback_first=True)
    """
    for i, agent in enumerate(agents):
        if (getattr(agent, "name", None) or f"agent_{i}") == agent_id:
            return agent

    if agent_id.startswith("agent_") and agent_id[6:].isdigit():
        idx = int(agent_id[6:])
        if 0 <= idx < len(agents):
            return agents[idx]

    if fallback_first and agents:
        return agents[0]

    return None


def generate_invocation_id(execution_id: str, agent_id: str) -> str:
    """Generate a unique invocation ID for an agent call.

    Creates a composite identifier combining the execution context, agent
    identity, and a random component for uniqueness.

    Args:
        execution_id: The parent execution's identifier (first 8 chars used).
        agent_id: The agent's identifier.

    Returns:
        A unique string in format "{execution_prefix}-{agent_id}-{random_hex}".
    """
    return f"{execution_id[:8]}-{agent_id}-{uuid.uuid4().hex[:8]}"


def generate_instance_id(execution_id: str, agent_id: str, instance_num: int) -> str:
    """Generate a unique instance ID for a parallel agent instance.

    Args:
        execution_id: The parent execution's identifier.
        agent_id: The agent definition's identifier.
        instance_num: The instance number (0-indexed).

    Returns:
        A unique string in format "{execution_prefix}-{agent_id}-i{num}-{random}".
    """
    return f"{execution_id[:8]}-{agent_id}-i{instance_num}-{uuid.uuid4().hex[:6]}"


def create_instance_context(
    instance_id: str,
    agent_id: str,
    task: str,
    base_context: dict | None = None,
) -> dict:
    """Create an isolated context for an agent instance.

    Each instance gets its own scratchpad/memory that is independent
    from other instances of the same agent.

    Args:
        instance_id: Unique identifier for this instance.
        agent_id: The agent definition ID.
        task: The specific task for this instance.
        base_context: Optional base context to extend.

    Returns:
        Dictionary containing the instance's isolated context.
    """
    return {
        "instance_id": instance_id,
        "agent_id": agent_id,
        "task": task,
        "scratchpad": [],
        "memory": {},
        "created_at": time.time(),
        **(base_context or {}),
    }


def emit_event(event: dict) -> None:
    """Emit a stream event to the LangGraph stream writer.

    Attempts to send an event through the LangGraph streaming infrastructure.
    Failures are logged at debug level to avoid disrupting workflow execution.

    Args:
        event: Dictionary containing event data (type, timestamps, etc.).
    """
    try:
        get_stream_writer()(event)
    except Exception as e:
        logger.debug(
            "Failed to emit stream event: %s (event type: %s)", e, event.get("type")
        )


def extract_content(result: Any) -> str:
    """Extract content from an agent's response result.

    Handles various response formats returned by agents, extracting the
    actual content string from message-based responses.

    Args:
        result: The raw result from an agent invocation.

    Returns:
        The extracted content as a string.
    """
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        return getattr(result["messages"][-1], "content", str(result["messages"][-1]))
    return str(result)


async def execute_single_agent(
    agent: Any,
    agent_id: str,
    task: str,
    execution_id: str = "",
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """Execute a single agent and return the result with event emission.

    Runs an agent with the given task, emitting start/end events for
    observability and handling errors with proper wrapping.

    Args:
        agent: The agent instance to invoke (must have ainvoke method).
        agent_id: Identifier for the agent being executed.
        task: The task/prompt to send to the agent.
        execution_id: Parent execution ID for correlation.
        recursion_limit: Optional max internal steps for agent execution.

    Returns:
        Dictionary with agent_id, invocation_id, result, and optional error fields.
    """
    invocation_id = generate_invocation_id(execution_id, agent_id)

    try:
        emit_event(
            {
                "type": "agent_start",
                "agent_id": agent_id,
                "invocation_id": invocation_id,
                "execution_id": execution_id,
                "task": task,
                "timestamp": time.time(),
            }
        )

        config = {"recursion_limit": recursion_limit} if recursion_limit else None
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task}]},
                config=config,
            )
        except TypeError:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task}]}
            )
        content = extract_content(result)

        emit_event(
            {
                "type": "agent_end",
                "agent_id": agent_id,
                "invocation_id": invocation_id,
                "execution_id": execution_id,
                "result": content,
                "timestamp": time.time(),
            }
        )

        return {"agent_id": agent_id, "invocation_id": invocation_id, "result": content}

    except Exception as e:
        error = wrap_agent_error(e, agent_id, task)
        return {
            "agent_id": agent_id,
            "invocation_id": invocation_id,
            "result": None,
            "error": str(error),
            "original_error_type": type(e).__name__,
            "recovery_suggestion": error.recovery_suggestion,
        }


async def execute_agent_instance(
    agent: Any,
    instance_id: str,
    agent_id: str,
    task: str,
    context: dict,
    execution_id: str = "",
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """Execute an agent instance with isolated context.

    Similar to execute_single_agent but includes instance-specific context
    and tracking for parallel execution scenarios.

    Args:
        agent: The agent instance to invoke.
        instance_id: Unique identifier for this specific instance.
        agent_id: The agent definition ID.
        task: The task/prompt to send to the agent.
        context: Instance-specific context/scratchpad.
        execution_id: Parent execution ID for correlation.
        recursion_limit: Optional max internal steps for agent execution.

    Returns:
        Dictionary with instance_id, agent_id, result, context, and optional error.
    """
    try:
        emit_event(
            {
                "type": "instance_start",
                "instance_id": instance_id,
                "agent_id": agent_id,
                "execution_id": execution_id,
                "task": task,
                "timestamp": time.time(),
            }
        )

        # Include context in the task if available
        task_with_context = task
        if context.get("scratchpad"):
            task_with_context = f"{task}\n\nContext from previous steps:\n" + "\n".join(
                str(s) for s in context["scratchpad"]
            )

        config = {"recursion_limit": recursion_limit} if recursion_limit else None
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task_with_context}]},
                config=config,
            )
        except TypeError:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task_with_context}]}
            )
        content = extract_content(result)

        # Update context with result
        updated_context = {
            **context,
            "last_result": content,
            "completed_at": time.time(),
        }
        updated_context["scratchpad"].append({"task": task, "result": content})

        emit_event(
            {
                "type": "instance_end",
                "instance_id": instance_id,
                "agent_id": agent_id,
                "execution_id": execution_id,
                "result": content,
                "timestamp": time.time(),
            }
        )

        return {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "result": content,
            "context": updated_context,
        }

    except Exception as e:
        error = wrap_agent_error(e, agent_id, task)
        return {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "result": None,
            "error": str(error),
            "original_error_type": type(e).__name__,
            "recovery_suggestion": error.recovery_suggestion,
            "context": context,
        }


async def make_error_result(
    agent_id: str, invocation_id: str, error_msg: str = ""
) -> dict[str, Any]:
    """Create a standardized error result dictionary.

    Args:
        agent_id: The agent's identifier.
        invocation_id: The invocation ID for this failed attempt.
        error_msg: The error message. Defaults to a generic "not found" message.

    Returns:
        Dictionary with agent_id, invocation_id, result (None), and error fields.
    """
    if not error_msg:
        error_msg = f"Agent '{agent_id}' not found"
    return {
        "agent_id": agent_id,
        "invocation_id": invocation_id,
        "result": None,
        "error": error_msg,
    }


# Inter-agent message validation utilities
_message_validator = InterAgentMessageValidator()


def create_agent_message(
    sender_id: str,
    receiver_id: str,
    content: Any,
    message_type: str,
    timestamp: float | None = None,
) -> tuple[AgentMessage | None, MessageValidationResult]:
    """Create and validate an inter-agent message.

    Creates an AgentMessage after validating the data against the Pydantic
    schema. If validation fails, returns None with error details.

    Args:
        sender_id: Identifier of the sending agent.
        receiver_id: Identifier of the receiving agent.
        content: Message content (can be any type).
        message_type: Type/category of the message.
        timestamp: Unix timestamp. Defaults to current time if not provided.

    Returns:
        Tuple of (AgentMessage or None, MessageValidationResult).
        If validation passes, returns the created message and a valid result.
        If validation fails, returns None and the error result.

    Example:
        >>> message, result = create_agent_message(
        ...     sender_id="agent_1",
        ...     receiver_id="agent_2",
        ...     content={"data": "test"},
        ...     message_type="task_result",
        ... )
        >>> if result.is_valid:
        ...     print(f"Message created: {message.sender_id} -> {message.receiver_id}")

    Validates: Requirements 4.2, 4.3, 4.4
    """
    if timestamp is None:
        timestamp = time.time()

    message_data = {
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "content": content,
        "message_type": message_type,
        "timestamp": timestamp,
    }

    return _message_validator.validate_and_create(message_data)


def validate_and_send_message(
    sender_id: str,
    receiver_id: str,
    content: Any,
    message_type: str,
    timestamp: float | None = None,
) -> tuple[AgentMessage | None, MessageValidationResult]:
    """Validate a message before sending between agents.

    This function validates the message data against the AgentMessage schema
    before creating the message. If validation fails, it returns detailed
    field-level error information in the rejection context.

    Args:
        sender_id: Identifier of the sending agent.
        receiver_id: Identifier of the receiving agent.
        content: Message content (can be any type).
        message_type: Type/category of the message.
        timestamp: Unix timestamp. Defaults to current time if not provided.

    Returns:
        Tuple of (AgentMessage or None, MessageValidationResult).
        The MessageValidationResult contains:
        - is_valid: Whether the message passed validation
        - field_errors: Dict of field names to error messages
        - rejection_reason: Human-readable rejection reason
        - expected_format: Description of expected message format

    Example:
        >>> message, result = validate_and_send_message(
        ...     sender_id="agent_1",
        ...     receiver_id="agent_2",
        ...     content="Hello",
        ...     message_type="greeting",
        ... )
        >>> if not result.is_valid:
        ...     print(f"Validation failed: {result.rejection_reason}")
        ...     for field, error in result.field_errors.items():
        ...         print(f"  {field}: {error}")
    """
    return create_agent_message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        content=content,
        message_type=message_type,
        timestamp=timestamp,
    )


def validate_message_data(message_data: dict[str, Any]) -> MessageValidationResult:
    """Validate raw message data against the AgentMessage schema.

    This is a lower-level function that validates a dictionary of message
    data without creating an AgentMessage object.

    Args:
        message_data: Dictionary containing the message data to validate.

    Returns:
        MessageValidationResult with validation status and error details.

    Example:
        >>> result = validate_message_data({
        ...     "sender_id": "agent_1",
        ...     "receiver_id": "agent_2",
        ...     "content": "test",
        ...     "message_type": "data",
        ...     "timestamp": 1234567890.0,
        ... })
        >>> assert result.is_valid is True

    Validates: Requirements 4.2, 4.3, 4.4
    """
    return _message_validator.validate(message_data)
