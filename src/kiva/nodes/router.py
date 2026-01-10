"""Workflow routing logic for the Kiva SDK.

This module provides the routing function that directs execution to the
appropriate workflow based on the analysis results.
"""

from typing import Literal

from kiva.state import OrchestratorState

WorkflowType = Literal["router_workflow", "supervisor_workflow", "parliament_workflow"]

WORKFLOW_MAP: dict[str, WorkflowType] = {
    "router": "router_workflow",
    "supervisor": "supervisor_workflow",
    "parliament": "parliament_workflow",
}


def route_to_workflow(state: OrchestratorState) -> WorkflowType:
    """Route to the appropriate workflow node based on state.

    Checks for workflow_override first, then falls back to the workflow
    determined by the analyze_and_plan node.

    Args:
        state: The current orchestrator state.

    Returns:
        The workflow node name to route to.
    """
    workflow = state.get("workflow_override") or state.get("workflow", "router")
    return WORKFLOW_MAP.get(workflow, "router_workflow")
