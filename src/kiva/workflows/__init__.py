"""Workflow implementations for the Kiva SDK.

This package provides three workflow patterns for orchestrating multi-agent systems,
with support for agent instance spawning and parallel execution:

Workflows:
    router_workflow: Routes tasks to a single agent (simple tasks).
    supervisor_workflow: Coordinates parallel agent/instance execution
        (medium complexity).
    parliament_workflow: Iterative conflict resolution (complex reasoning).

Executor:
    execute_instance_node: Executes individual agent instances spawned via Send.
    execute_instances_batch: Batch execution of multiple instances.

Utilities:
    The utils module provides shared helper functions used across workflows,
    including agent lookup, event emission, instance management, and result extraction.
"""

from kiva.workflows.executor import execute_instance_node, execute_instances_batch
from kiva.workflows.parliament import (
    parliament_workflow,
    should_continue_parliament,
)
from kiva.workflows.router import router_workflow
from kiva.workflows.supervisor import supervisor_workflow
from kiva.workflows.utils import (
    create_instance_context,
    emit_event,
    execute_agent_instance,
    execute_single_agent,
    extract_content,
    generate_instance_id,
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
    # Executor
    "execute_instance_node",
    "execute_instances_batch",
    # Utilities
    "get_agent_by_id",
    "generate_invocation_id",
    "generate_instance_id",
    "create_instance_context",
    "emit_event",
    "extract_content",
    "execute_single_agent",
    "execute_agent_instance",
    "make_error_result",
]
