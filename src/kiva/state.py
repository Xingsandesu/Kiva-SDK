"""State definitions for the Kiva SDK.

This module defines the state schemas used throughout the orchestration graph,
including the main OrchestratorState and the AgentInstanceState for parallel
agent execution with isolated contexts.
"""

import operator
from typing import Annotated, TypedDict


class AgentInstanceState(TypedDict):
    """State for individual agent instance execution.

    Each agent instance runs with its own isolated context, enabling
    parallel execution of the same agent definition with different tasks.

    Attributes:
        instance_id: Unique identifier for this agent instance.
        agent_id: The agent definition ID (can have multiple instances).
        task: The specific subtask assigned to this instance.
        context: Instance-specific context/memory (scratchpad).
        execution_id: Parent execution identifier for correlation.
        model_name: Model identifier for the agent.
        api_key: API authentication key.
        base_url: API endpoint URL.
    """

    instance_id: str
    agent_id: str
    task: str
    context: dict
    execution_id: str
    model_name: str
    api_key: str | None
    base_url: str | None


class TaskAssignment(TypedDict, total=False):
    """Task assignment for an agent.

    Attributes:
        agent_id: The agent definition to use.
        task: The specific task description.
        instances: Number of parallel instances to spawn (default: 1).
        instance_context: Optional context to pass to each instance.
    """

    agent_id: str
    task: str
    instances: int
    instance_context: dict


class PlanningResult(TypedDict, total=False):
    """Result from the planning/intent detection phase.

    Attributes:
        complexity: Assessed task complexity.
        workflow: Selected workflow type.
        reasoning: Explanation of the planning decision.
        task_assignments: List of task assignments with instance info.
        parallel_strategy: How to parallelize ("none", "fan_out", "map_reduce").
        total_instances: Total number of agent instances to spawn.
    """

    complexity: str
    workflow: str
    reasoning: str
    task_assignments: list[TaskAssignment]
    parallel_strategy: str
    total_instances: int


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
        task_assignments: List of task assignments with instance configuration.
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
        parallel_strategy: Parallelization strategy from planner.
        total_instances: Total agent instances planned for execution.
        instance_contexts: Accumulated contexts from completed instances.
    """

    messages: Annotated[list, operator.add]
    prompt: str
    complexity: str
    workflow: str
    agents: list
    task_assignments: list[TaskAssignment]
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
    parallel_strategy: str
    total_instances: int
    instance_contexts: Annotated[list[dict], operator.add]
