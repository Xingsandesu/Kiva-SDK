"""Router Workflow - routes to a single worker agent for simple tasks.

This workflow is designed for straightforward tasks that can be handled by
a single agent. It selects the most appropriate agent from the available
pool and delegates the entire task to that agent.

Use cases:
    - Simple Q&A tasks
    - Single-domain queries
    - Tasks with clear, unambiguous requirements
"""

import time
from typing import Any

from kiva.state import OrchestratorState
from kiva.workflows.utils import (
    emit_event,
    extract_content,
    generate_invocation_id,
    get_agent_by_id,
)


async def router_workflow(state: OrchestratorState) -> dict[str, Any]:
    """Route and execute a task using a single worker agent.

    Selects the first assigned agent and delegates the complete task to it.
    Emits streaming events for observability and handles errors gracefully.

    Args:
        state: The orchestrator state containing task assignments, agents,
            execution ID, and prompt.

    Returns:
        Dictionary with 'agent_results' containing a single result entry.
    """
    task_assignments = state.get("task_assignments", [])
    agents = state.get("agents", [])
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

    assignment = task_assignments[0]
    agent_id = assignment.get("agent_id", "agent_0")
    task = assignment.get("task", state.get("prompt", ""))
    invocation_id = generate_invocation_id(execution_id, agent_id)

    # Router uses fallback_first=True to ensure a valid agent is selected
    agent = get_agent_by_id(agents, agent_id, fallback_first=True)
    if agent is None:
        return {
            "agent_results": [
                {
                    "agent_id": agent_id,
                    "invocation_id": invocation_id,
                    "result": None,
                    "error": f"Agent '{agent_id}' not found",
                }
            ]
        }

    try:
        recursion_limit = getattr(agent, "kiva_recursion_limit", None) or state.get(
            "worker_recursion_limit", 25
        )
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

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task}]},
                config={"recursion_limit": recursion_limit},
            )
        except TypeError:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task}]}
            )
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

        return {
            "agent_results": [
                {
                    "agent_id": agent_id,
                    "invocation_id": invocation_id,
                    "result": content,
                }
            ]
        }

    except Exception as e:
        from kiva.exceptions import wrap_agent_error

        error = wrap_agent_error(e, agent_id, task)
        return {
            "agent_results": [
                {
                    "agent_id": agent_id,
                    "invocation_id": invocation_id,
                    "result": None,
                    "error": str(error),
                    "original_error_type": type(e).__name__,
                    "recovery_suggestion": error.recovery_suggestion,
                }
            ]
        }
