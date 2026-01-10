"""Event definitions for the Kiva SDK.

This module defines the StreamEvent dataclass used for real-time event
streaming during orchestration execution.

Event Types:
    - token: Streaming token from LLM response
    - workflow_selected: Workflow type and complexity determined
    - parallel_start: Parallel agent execution initiated
    - agent_start: Individual agent execution started
    - agent_end: Individual agent execution completed
    - parallel_complete: All parallel agents finished
    - final_result: Final synthesized result available
    - error: Error occurred during execution
"""

from dataclasses import asdict, dataclass
from typing import Any


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
