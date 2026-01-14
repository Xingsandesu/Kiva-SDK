"""Event definitions for the Kiva SDK.

This module defines the StreamEvent dataclass used for real-time event
streaming during orchestration execution.

Event Types:
    Basic Events:
        - token: Streaming token from LLM response
        - workflow_selected: Workflow type and complexity determined
        - final_result: Final synthesized result available
        - error: Error occurred during execution

    Single-Agent Events:
        - parallel_start: Parallel agent execution initiated
        - agent_start: Individual agent execution started
        - agent_end: Individual agent execution completed
        - parallel_complete: All parallel agents finished

    Parallel Instance Events:
        - instance_spawn: New agent instance created and starting
        - instance_start: Agent instance beginning task execution
        - instance_end: Agent instance completed task
        - instance_complete: Agent instance finished (success or error)
        - instance_result: Result from an agent instance execution
        - parallel_instances_start: Batch of instances starting execution
        - parallel_instances_complete: Batch of instances finished execution

    Worker Verification Events:
        - worker_verification_start: Worker output verification initiated
        - worker_verification_passed: All workers passed verification
        - worker_verification_failed: One or more workers failed verification
        - worker_verification_max_reached: Max worker verification iterations exceeded

    Retry Events:
        - retry_triggered: Worker retry execution initiated
        - retry_completed: Worker retry execution completed
        - retry_skipped: Retry skipped (no retry context available)
"""

from dataclasses import asdict, dataclass
from typing import Any

# Worker Verification Event Types
WORKER_VERIFICATION_START = "worker_verification_start"
WORKER_VERIFICATION_PASSED = "worker_verification_passed"
WORKER_VERIFICATION_FAILED = "worker_verification_failed"
WORKER_VERIFICATION_MAX_REACHED = "worker_verification_max_reached"
WORKER_VERIFICATION_ERROR = "worker_verification_error"

# Retry Event Types
RETRY_TRIGGERED = "retry_triggered"
RETRY_COMPLETED = "retry_completed"
RETRY_SKIPPED = "retry_skipped"

# Verification lifecycle/state machine event types
VERIFICATION_STATE_CHANGED = "verification_state_changed"

# All verification-related event types for easy reference
WORKER_VERIFICATION_EVENT_TYPES = [
    WORKER_VERIFICATION_START,
    WORKER_VERIFICATION_PASSED,
    WORKER_VERIFICATION_FAILED,
    WORKER_VERIFICATION_MAX_REACHED,
    WORKER_VERIFICATION_ERROR,
]

RETRY_EVENT_TYPES = [
    RETRY_TRIGGERED,
    RETRY_COMPLETED,
    RETRY_SKIPPED,
]

VERIFICATION_EVENT_TYPES = (
    WORKER_VERIFICATION_EVENT_TYPES + RETRY_EVENT_TYPES + [VERIFICATION_STATE_CHANGED]
)


def is_verification_event(event_type: str) -> bool:
    """Check if an event type is a verification-related event.

    Args:
        event_type: The event type string to check.

    Returns:
        True if the event type is verification-related, False otherwise.
    """
    return event_type in VERIFICATION_EVENT_TYPES


def is_worker_verification_event(event_type: str) -> bool:
    """Check if an event type is a worker verification event.

    Args:
        event_type: The event type string to check.

    Returns:
        True if the event type is a worker verification event, False otherwise.
    """
    return event_type in WORKER_VERIFICATION_EVENT_TYPES


def is_retry_event(event_type: str) -> bool:
    """Check if an event type is a retry event.

    Args:
        event_type: The event type string to check.

    Returns:
        True if the event type is a retry event, False otherwise.
    """
    return event_type in RETRY_EVENT_TYPES


@dataclass
class StreamEvent:
    """Streaming event emitted during orchestration execution.

    Attributes:
        type: Event type identifier (e.g., "token", "agent_start", "final_result").
        data: Event payload containing type-specific information.
        timestamp: Unix timestamp when the event was created.
        agent_id: Identifier of the agent associated with this event, if applicable.

    Example:
        >>> event = StreamEvent(
        ...     type="agent_end",
        ...     data={"result": "Task completed"},
        ...     timestamp=1234567890.0,
        ...     agent_id="calculator",
        ... )
        >>> event.to_dict()
        {'type': 'agent_end', 'data': {...}, ...}
    """

    type: str
    data: dict[str, Any]
    timestamp: float
    agent_id: str | None = None

    def to_dict(self) -> dict:
        """Convert the event to a dictionary representation."""
        return asdict(self)
