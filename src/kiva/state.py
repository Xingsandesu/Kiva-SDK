"""State definitions for the Kiva SDK.

This module defines the OrchestratorState TypedDict used throughout the
orchestration graph to maintain execution state.
"""

import operator
from typing import Annotated, TypedDict


class OrchestratorState(TypedDict):
    """State container for the orchestration graph.

    This TypedDict defines all state fields passed between graph nodes during
    orchestration execution. Fields marked with Annotated[..., operator.add]
    are accumulated across node executions.

    Attributes:
        messages: Accumulated LLM messages from all nodes.
        prompt: Original user input/task description.
        complexity: Assessed task complexity ("simple", "medium", "complex").
        workflow: Selected workflow type ("router", "supervisor", "parliament").
        agents: List of available worker agent instances.
        task_assignments: List of {agent_id, task} assignments from planning.
        agent_results: Accumulated results from agent executions.
        final_result: Synthesized final response, or None if not yet complete.
        execution_id: Unique identifier for this execution.
        conflicts: Detected conflicts between agent responses (parliament workflow).
        iteration: Current iteration number (parliament workflow).
        model_name: Model identifier for the lead agent.
        api_key: API authentication key.
        base_url: API endpoint URL.
        workflow_override: Force a specific workflow, bypassing analysis.
        max_iterations: Maximum iterations for parliament workflow.
        max_parallel_agents: Maximum concurrent agent executions.
    """

    messages: Annotated[list, operator.add]
    prompt: str
    complexity: str
    workflow: str
    agents: list
    task_assignments: list[dict]
    agent_results: Annotated[list[dict], operator.add]
    final_result: str | None
    execution_id: str
    conflicts: list[dict]
    iteration: int
    model_name: str
    api_key: str | None
    base_url: str | None
    workflow_override: str | None
    max_iterations: int
    max_parallel_agents: int
