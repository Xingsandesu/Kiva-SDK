"""Agent instance executor for parallel execution.

This module provides the execution node for individual agent instances
spawned via the Send API. Each instance runs with isolated context.
"""

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from kiva.state import AgentInstanceState
from kiva.workflows.utils import (
    emit_event,
    execute_agent_instance,
    get_agent_by_id,
)


async def execute_instance_node(
    state: AgentInstanceState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute a single agent instance with isolated context.

    This node is invoked via Send from the routing logic. Each instance
    has its own context/scratchpad and executes independently.

    Args:
        state: The AgentInstanceState containing instance-specific data.
        config: Optional LangGraph config with parent state access.

    Returns:
        Dictionary with agent_results and instance_contexts to merge
        back into the parent OrchestratorState.
    """
    instance_id = state.get("instance_id", "")
    agent_id = state.get("agent_id", "")
    task = state.get("task", "")
    context = state.get("context", {})
    execution_id = state.get("execution_id", "")

    # Get agents from configurable or use empty list
    # Note: In Send-based execution, we need to pass agents via config
    agents = []
    worker_recursion_limit = 100
    if config and "configurable" in config:
        agents = config["configurable"].get("agents", [])
        worker_recursion_limit = config["configurable"].get(
            "worker_recursion_limit", worker_recursion_limit
        )

    emit_event(
        {
            "type": "instance_spawn",
            "instance_id": instance_id,
            "agent_id": agent_id,
            "execution_id": execution_id,
            "task": task,
            "timestamp": time.time(),
        }
    )

    agent = get_agent_by_id(agents, agent_id)
    if agent is None:
        error_result = {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "result": None,
            "error": f"Agent '{agent_id}' not found",
            "context": context,
        }
        return {
            "agent_results": [error_result],
            "instance_contexts": [context],
        }

    recursion_limit = (
        getattr(agent, "kiva_recursion_limit", None) or worker_recursion_limit
    )
    result = await execute_agent_instance(
        agent=agent,
        instance_id=instance_id,
        agent_id=agent_id,
        task=task,
        context=context,
        execution_id=execution_id,
        recursion_limit=recursion_limit,
    )

    emit_event(
        {
            "type": "instance_complete",
            "instance_id": instance_id,
            "agent_id": agent_id,
            "execution_id": execution_id,
            "success": result.get("error") is None,
            "timestamp": time.time(),
        }
    )

    return {
        "agent_results": [result],
        "instance_contexts": [result.get("context", context)],
    }


async def execute_instances_batch(
    instances: list[dict],
    agents: list,
    execution_id: str,
    *,
    worker_recursion_limit: int = 100,
) -> list[dict[str, Any]]:
    """Execute multiple agent instances in parallel.

    This is a utility function for workflows that need to execute
    multiple instances without using the Send API directly.

    Args:
        instances: List of instance configurations with agent_id, task, context.
        agents: List of available agent definitions.
        execution_id: Parent execution ID for correlation.
        worker_recursion_limit: Optional max internal steps for agent execution.

    Returns:
        List of result dictionaries from all instances.
    """
    import asyncio

    from kiva.workflows.utils import (
        create_instance_context,
        generate_instance_id,
    )

    tasks = []
    for i, inst in enumerate(instances):
        agent_id = inst.get("agent_id", f"agent_{i}")
        task = inst.get("task", "")
        base_context = inst.get("context", {})

        instance_id = generate_instance_id(execution_id, agent_id, i)
        context = create_instance_context(instance_id, agent_id, task, base_context)

        agent = get_agent_by_id(agents, agent_id)
        if agent is None:
            # Create error result for missing agent
            async def make_error(aid=agent_id, iid=instance_id, ctx=context):
                return {
                    "instance_id": iid,
                    "agent_id": aid,
                    "result": None,
                    "error": f"Agent '{aid}' not found",
                    "context": ctx,
                }

            tasks.append(make_error())
        else:
            recursion_limit = (
                getattr(agent, "kiva_recursion_limit", None) or worker_recursion_limit
            )
            tasks.append(
                execute_agent_instance(
                    agent=agent,
                    instance_id=instance_id,
                    agent_id=agent_id,
                    task=task,
                    context=context,
                    execution_id=execution_id,
                    recursion_limit=recursion_limit,
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            agent_id = instances[i].get("agent_id", f"agent_{i}")
            processed_results.append(
                {
                    "instance_id": f"error-{i}",
                    "agent_id": agent_id,
                    "result": None,
                    "error": str(result),
                }
            )
        else:
            processed_results.append(result)

    return processed_results
