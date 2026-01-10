"""Workflow implementations for the Kiva SDK.

This package provides three workflow patterns for orchestrating multi-agent systems:

Workflows:
    router_workflow: Routes tasks to a single agent (simple tasks).
    supervisor_workflow: Coordinates parallel agent execution (medium complexity).
    parliament_workflow: Iterative conflict resolution (complex reasoning).

Utilities:
    The utils module provides shared helper functions used across workflows,
    including agent lookup, event emission, and result extraction.
"""

from kiva.workflows.parliament import (
    parliament_workflow,
    should_continue_parliament,
)
from kiva.workflows.router import router_workflow
from kiva.workflows.supervisor import supervisor_workflow
from kiva.workflows.utils import (
    emit_event,
    execute_single_agent,
    extract_content,
    generate_invocation_id,
    get_agent_by_id,
    make_error_result,
)

__all__ = [
    # Workflows
    "router_workflow",
    "supervisor_workflow",
    "parliament_workflow",
    "should_continue_parliament",
    # Utilities
    "get_agent_by_id",
    "generate_invocation_id",
    "emit_event",
    "extract_content",
    "execute_single_agent",
    "make_error_result",
]
