"""Supervisor Workflow - coordinates multiple agents in parallel.

This workflow manages parallel execution of multiple agents, each handling
a portion of the overall task. Suitable for tasks that can be decomposed
into independent subtasks without requiring iterative refinement.

Use cases:
    - Multi-faceted research tasks
    - Parallel data processing
    - Tasks requiring diverse expertise without conflict resolution
"""

import asyncio
import time
from typing import Any

from kiva.state import OrchestratorState
from kiva.workflows.utils import (
    emit_event,
    execute_single_agent,
    generate_invocation_id,
    get_agent_by_id,
    make_error_result,
)


async def supervisor_workflow(state: OrchestratorState) -> dict[str, Any]:
    """Execute multiple worker agents in parallel under supervisor coordination.

    Distributes task assignments across available agents, executing them
    concurrently up to the configured maximum parallelism limit.

    Args:
        state: The orchestrator state containing:
            - task_assignments: List of {agent_id, task} dictionaries
            - agents: List of available agent instances
            - max_parallel_agents: Maximum concurrent executions (default: 5)
            - execution_id: Parent execution identifier
            - prompt: Fallback prompt if task not specified

    Returns:
        Dictionary with 'agent_results' containing results from all agents.
    """
    task_assignments = state.get("task_assignments", [])
    agents = state.get("agents", [])
    max_parallel = state.get("max_parallel_agents", 5)
    execution_id = state.get("execution_id", "")

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
            tasks.append(execute_single_agent(agent, agent_id, task, execution_id))

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
