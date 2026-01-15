"""Shared utility functions for workflow implementations.

This module provides common helper functions used across different workflow
types (router, supervisor, parliament) to avoid code duplication and ensure
consistent behavior. Includes support for agent instance management.

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
"""

import logging
import time
import uuid
from typing import Any

from langgraph.config import get_stream_writer

from kiva.exceptions import wrap_agent_error

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
    agent: Any, agent_id: str, task: str, execution_id: str = ""
) -> dict[str, Any]:
    """Execute a single agent and return the result with event emission.

    Runs an agent with the given task, emitting start/end events for
    observability and handling errors with proper wrapping.

    Args:
        agent: The agent instance to invoke (must have ainvoke method).
        agent_id: Identifier for the agent being executed.
        task: The task/prompt to send to the agent.
        execution_id: Parent execution ID for correlation.

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

        result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
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
    worker_max_iterations: int = 100,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Execute an agent instance with isolated context.

    Similar to execute_single_agent but includes instance-specific context
    and tracking for parallel execution scenarios.
    Supports real-time streaming of worker agent state updates.
    Implements retry logic with exponential backoff for resilience.

    Args:
        agent: The agent instance to invoke.
        instance_id: Unique identifier for this specific instance.
        agent_id: The agent definition ID.
        task: The task/prompt to send to the agent.
        context: Instance-specific context/scratchpad.
        execution_id: Parent execution ID for correlation.
        worker_max_iterations: Maximum iterations for the worker agent.
        max_retries: Maximum number of retries on failure.

    Returns:
        Dictionary with instance_id, agent_id, result, context, and optional error.
    """
    import asyncio

    # Try block for the entire execution including retries
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                emit_event(
                    {
                        "type": "instance_retry",
                        "instance_id": instance_id,
                        "agent_id": agent_id,
                        "execution_id": execution_id,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "error": str(last_error),
                        "timestamp": time.time(),
                    }
                )
                # Exponential backoff
                await asyncio.sleep(min(2**attempt, 10))

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
                task_with_context = (
                    f"{task}\n\nContext from previous steps:\n"
                    + "\n".join(str(s) for s in context["scratchpad"])
                )

            input_data = {"messages": [{"role": "user", "content": task_with_context}]}
            config = {"recursion_limit": worker_max_iterations}

            result = None

            # Try to stream updates if supported
            if hasattr(agent, "astream"):
                last_state = None
                async for state_chunk in agent.astream(
                    input_data, config=config, stream_mode="values"
                ):
                    last_state = state_chunk

                    # Extract latest message to emit update event
                    messages = state_chunk.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        # Avoid emitting the initial user message if possible,
                        # or just emit everything. The consumer can filter.

                        emit_event({
                            "type": "worker_state_update",
                            "instance_id": instance_id,
                            "agent_id": agent_id,
                            "execution_id": execution_id,
                            "data": {
                                "role": getattr(last_msg, "type", "unknown"),
                                "content": getattr(last_msg, "content", str(last_msg)),
                                "tool_calls": getattr(last_msg, "tool_calls", []),
                                "additional_kwargs": getattr(
                                    last_msg, "additional_kwargs", {}
                                ),
                            },
                            "timestamp": time.time()
                        })
                result = last_state
            else:
                result = await agent.ainvoke(input_data, config=config)

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
            last_error = e
            # Only continue loop if we have retries left
            if attempt < max_retries:
                logger.warning(
                    f"Agent {agent_id} failed attempt {attempt+1}/{max_retries+1}: {e}"
                )
                continue

            # If we're here, we've exhausted retries
            error = wrap_agent_error(e, agent_id, task)

            # Emit error event with details for console=False support
            emit_event({
                "type": "instance_error",
                "instance_id": instance_id,
                "agent_id": agent_id,
                "execution_id": execution_id,
                "error": str(error),
                "timestamp": time.time()
            })

            return {
                "instance_id": instance_id,
                "agent_id": agent_id,
                "result": None,
                "error": str(error),
                "original_error_type": type(e).__name__,
                "recovery_suggestion": error.recovery_suggestion,
                "context": context,
            }

    # Should be unreachable given the return in except block, but for type safety:
    return {
        "instance_id": instance_id,
        "agent_id": agent_id,
        "result": None,
        "error": "Unknown error (retries exhausted)",
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
