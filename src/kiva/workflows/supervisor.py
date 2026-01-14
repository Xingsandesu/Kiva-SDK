"""Supervisor Workflow - coordinates multiple agents with instance support.

This workflow manages parallel execution of multiple agents, each handling
a portion of the overall task. Now supports spawning multiple instances
of the same agent for parallel subtask processing.

Use cases:
    - Multi-faceted research tasks
    - Parallel data processing
    - Tasks requiring diverse expertise without conflict resolution
    - Map-reduce style operations with multiple instances
"""

import time
from typing import Any

from kiva.state import OrchestratorState
from kiva.workflows.executor import execute_instances_batch
from kiva.workflows.utils import (
    emit_event,
    execute_single_agent,
    generate_invocation_id,
    get_agent_by_id,
    make_error_result,
)


async def supervisor_workflow(state: OrchestratorState) -> dict[str, Any]:
    """Execute multiple worker agents/instances under supervisor coordination.

    Distributes task assignments across available agents, supporting both
    single-instance and multi-instance execution patterns. When instances > 1
    for an assignment, spawns multiple parallel instances of the same agent.

    Args:
        state: The orchestrator state containing:
            - task_assignments: List of {agent_id, task, instances} dictionaries
            - agents: List of available agent instances
            - max_parallel_agents: Maximum concurrent executions (default: 5)
            - execution_id: Parent execution identifier
            - prompt: Fallback prompt if task not specified
            - parallel_strategy: "none", "fan_out", or "map_reduce"

    Returns:
        Dictionary with 'agent_results' containing results from all agents/instances.
    """
    task_assignments = state.get("task_assignments", [])
    agents = state.get("agents", [])
    max_parallel = state.get("max_parallel_agents", 5)
    execution_id = state.get("execution_id", "")
    parallel_strategy = state.get("parallel_strategy", "none")

    if not task_assignments:
        return {
            "agent_results": [
                {
                    "agent_id": "unknown",
                    "result": None,
                    "error": "No task assignments provided",
                }
            ]
        }

    if not agents:
        return {
            "agent_results": [
                {"agent_id": "unknown", "result": None, "error": "No agents available"}
            ]
        }

    # Check if we need multi-instance execution
    needs_instances = parallel_strategy in ("fan_out", "map_reduce") or any(
        a.get("instances", 1) > 1 for a in task_assignments
    )

    if needs_instances:
        return await _execute_with_instances(
            task_assignments, agents, max_parallel, execution_id, state
        )
    else:
        return await _execute_single_agents(
            task_assignments, agents, max_parallel, execution_id, state
        )


async def _execute_with_instances(
    task_assignments: list[dict],
    agents: list,
    max_parallel: int,
    execution_id: str,
    state: OrchestratorState,
) -> dict[str, Any]:
    """Execute with multi-instance support.

    Expands task assignments into individual instances and executes them
    in parallel batches.
    """
    # Expand assignments into instances
    instances = []
    for assignment in task_assignments:
        agent_id = assignment.get("agent_id", "agent_0")
        task = assignment.get("task", state.get("prompt", ""))
        num_instances = min(assignment.get("instances", 1), max_parallel)
        base_context = assignment.get("instance_context", {})

        for i in range(num_instances):
            instances.append(
                {
                    "agent_id": agent_id,
                    "task": task,
                    "context": {**base_context, "instance_num": i},
                }
            )

            if len(instances) >= max_parallel:
                break

        if len(instances) >= max_parallel:
            break

    # Emit parallel start event
    emit_event(
        {
            "type": "parallel_instances_start",
            "instance_count": len(instances),
            "agent_ids": list({inst["agent_id"] for inst in instances}),
            "execution_id": execution_id,
            "timestamp": time.time(),
        }
    )

    # Execute all instances
    results = await execute_instances_batch(
        instances,
        agents,
        execution_id,
        worker_recursion_limit=state.get("worker_recursion_limit", 25),
    )

    # Convert instance results to agent_results format
    agent_results = []
    instance_contexts = []
    for result in results:
        agent_results.append(
            {
                "agent_id": result.get("agent_id"),
                "instance_id": result.get("instance_id"),
                "invocation_id": result.get(
                    "instance_id"
                ),  # Use instance_id as invocation_id
                "result": result.get("result"),
                "error": result.get("error"),
            }
        )
        if ctx := result.get("context"):
            instance_contexts.append(ctx)

    emit_event(
        {
            "type": "parallel_instances_complete",
            "results": [
                {
                    "agent_id": r.get("agent_id"),
                    "instance_id": r.get("instance_id"),
                    "success": r.get("error") is None,
                }
                for r in agent_results
            ],
            "execution_id": execution_id,
            "timestamp": time.time(),
        }
    )

    return {
        "agent_results": agent_results,
        "instance_contexts": instance_contexts,
    }


async def _execute_single_agents(
    task_assignments: list[dict],
    agents: list,
    max_parallel: int,
    execution_id: str,
    state: OrchestratorState,
) -> dict[str, Any]:
    """Execute single-instance agents (original behavior)."""
    import asyncio

    agent_ids = [
        a.get("agent_id", f"agent_{i}") for i, a in enumerate(task_assignments)
    ]
    emit_event(
        {
            "type": "parallel_start",
            "agent_ids": agent_ids,
            "execution_id": execution_id,
            "timestamp": time.time(),
        }
    )

    tasks = []
    for i, assignment in enumerate(task_assignments[:max_parallel]):
        agent_id = assignment.get("agent_id", f"agent_{i}")
        task = assignment.get("task", state.get("prompt", ""))
        agent = get_agent_by_id(agents, agent_id)

        if agent is None:
            invocation_id = generate_invocation_id(execution_id, agent_id)
            tasks.append(make_error_result(agent_id, invocation_id))
        else:
            recursion_limit = getattr(agent, "kiva_recursion_limit", None) or state.get(
                "worker_recursion_limit", 25
            )
            tasks.append(
                execute_single_agent(
                    agent,
                    agent_id,
                    task,
                    execution_id,
                    recursion_limit=recursion_limit,
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    agent_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            agent_id = (
                task_assignments[i].get("agent_id", f"agent_{i}")
                if i < len(task_assignments)
                else f"agent_{i}"
            )
            invocation_id = generate_invocation_id(execution_id, agent_id)
            agent_results.append(
                {
                    "agent_id": agent_id,
                    "invocation_id": invocation_id,
                    "result": None,
                    "error": str(result),
                }
            )
        else:
            agent_results.append(result)

    emit_event(
        {
            "type": "parallel_complete",
            "results": [
                {"agent_id": r.get("agent_id"), "success": r.get("error") is None}
                for r in agent_results
            ],
            "execution_id": execution_id,
            "timestamp": time.time(),
        }
    )

    return {"agent_results": agent_results}
