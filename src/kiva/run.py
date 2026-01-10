"""SDK entry point - the run() function.

This module provides the main entry point for running multi-agent orchestration.
The run() function is an async generator that yields StreamEvent objects for
real-time monitoring of execution progress.

Example:
    >>> async for event in run(prompt="...", agents=[...]):
    ...     if event.type == "final_result":
    ...         print(event.data["result"])
"""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from kiva.events import StreamEvent
from kiva.exceptions import ConfigurationError
from kiva.graph import build_orchestrator_graph


async def run(
    prompt: str,
    agents: list[Any],
    *,
    model_name: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    workflow_override: str | None = None,
    max_iterations: int = 10,
    max_parallel_agents: int = 5,
) -> AsyncIterator[StreamEvent]:
    """Run multi-agent orchestration and yield streaming events.

    This is the main entry point for the Kiva SDK. It analyzes the task,
    selects an appropriate workflow, executes agents, and synthesizes results.

    Args:
        prompt: User input or task description.
        agents: List of worker agents. Each agent must have an `ainvoke` method.
        model_name: Model identifier for the lead agent. Defaults to "gpt-4o".
        api_key: API authentication key. Optional.
        base_url: API endpoint URL. Optional.
        workflow_override: Force a specific workflow ("router", "supervisor",
            "parliament"). If None, workflow is selected automatically.
        max_iterations: Maximum iterations for parliament workflow. Defaults to 10.
        max_parallel_agents: Maximum concurrent agent executions. Defaults to 5.

    Yields:
        StreamEvent objects representing execution progress.

    Raises:
        ConfigurationError: If agents list is empty or agents lack ainvoke method.

    Example:
        >>> from kiva import run, create_agent, ChatOpenAI, tool
        >>>
        >>> @tool
        ... def search(query: str) -> str:
        ...     '''Search for information.'''
        ...     return f"Results for {query}"
        >>>
        >>> model = ChatOpenAI(model="gpt-4o", api_key="...")
        >>> agent = create_agent(model=model, tools=[search])
        >>> agent.name = "searcher"
        >>> agent.description = "Searches for information"
        >>>
        >>> async for event in run("Search for Python", agents=[agent]):
        ...     print(event.type)
    """
    if not agents:
        raise ConfigurationError("agents list cannot be empty")

    for i, agent in enumerate(agents):
        if not hasattr(agent, "ainvoke"):
            raise ConfigurationError(
                f"Agent at index {i} must have ainvoke method. "
                f"Please use create_agent() to create agents."
            )

    execution_id = str(uuid.uuid4())
    graph = build_orchestrator_graph()

    initial_state = {
        "prompt": prompt,
        "agents": agents,
        "messages": [],
        "agent_results": [],
        "execution_id": execution_id,
        "conflicts": [],
        "iteration": 0,
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "workflow_override": workflow_override,
        "max_iterations": max_iterations,
        "max_parallel_agents": max_parallel_agents,
        "complexity": "",
        "workflow": "",
        "task_assignments": [],
        "final_result": None,
    }

    async for chunk in graph.astream(
        initial_state, stream_mode=["messages", "updates", "custom"]
    ):
        async for event in _process_stream_chunk(chunk, execution_id):
            yield event


async def _process_stream_chunk(
    chunk: tuple[str, Any], execution_id: str
) -> AsyncIterator[StreamEvent]:
    """Process a stream chunk and yield StreamEvent objects.

    Args:
        chunk: Tuple of (mode, data) from the graph stream.
        execution_id: Parent execution identifier for correlation.

    Yields:
        StreamEvent objects based on the chunk content.
    """
    mode, data = chunk

    if mode == "messages":
        msg_chunk, metadata = data
        if content := getattr(msg_chunk, "content", ""):
            yield StreamEvent(
                type="token",
                data={"content": content, "execution_id": execution_id},
                timestamp=time.time(),
                agent_id=metadata.get("langgraph_node"),
            )

    elif mode == "updates" and isinstance(data, dict):
        for node_name, node_data in data.items():
            if isinstance(node_data, dict):
                async for event in _process_node_update(
                    node_name, node_data, execution_id
                ):
                    yield event

    elif mode == "custom" and isinstance(data, dict):
        event_type = data.get("type", "custom")

        # Normalize agent_end events to match updates mode format
        if event_type == "agent_end":
            result_data = {
                "agent_id": data.get("agent_id"),
                "invocation_id": data.get("invocation_id"),
                "result": data.get("result"),
            }
            event_data = {"result": result_data, "execution_id": execution_id}
        else:
            event_data = {**data, "execution_id": execution_id}

        yield StreamEvent(
            type=event_type,
            data=event_data,
            timestamp=data.get("timestamp", time.time()),
            agent_id=data.get("agent_id"),
        )


async def _process_node_update(
    node_name: str, node_data: dict[str, Any], execution_id: str
) -> AsyncIterator[StreamEvent]:
    """Process a node update and yield appropriate StreamEvent objects.

    Args:
        node_name: Name of the graph node that produced the update.
        node_data: Data dictionary from the node.
        execution_id: Parent execution identifier.

    Yields:
        StreamEvent objects based on the node update.
    """
    if node_name == "analyze_and_plan":
        if workflow := node_data.get("workflow"):
            yield StreamEvent(
                type="workflow_selected",
                data={
                    "workflow": workflow,
                    "complexity": node_data.get("complexity"),
                    "execution_id": execution_id,
                    "task_assignments": node_data.get("task_assignments", []),
                },
                timestamp=time.time(),
            )

    elif node_name in ("router_workflow", "supervisor_workflow", "parliament_workflow"):
        # Events are emitted via emit_event() during workflow execution
        pass

    elif node_name == "synthesize_results":
        final_result = node_data.get("final_result", "")
        partial_info = node_data.get("partial_result_info", {})

        if final_result is not None:
            citations = node_data.get("citations") or _extract_citations_from_result(
                final_result or ""
            )
            event_data = {
                "result": final_result,
                "citations": citations,
                "execution_id": execution_id,
            }

            if partial_info:
                event_data.update(
                    {
                        "partial_result": partial_info.get("is_partial", False),
                        "success_count": partial_info.get("success_count", 0),
                        "failure_count": partial_info.get("failure_count", 0),
                    }
                )
                if partial_info.get("failed"):
                    event_data["failed_agents"] = [
                        f["agent_id"] for f in partial_info["failed"]
                    ]

            yield StreamEvent(
                type="final_result", data=event_data, timestamp=time.time()
            )


def _extract_citations_from_result(result: str) -> list[dict[str, str]]:
    """Extract citations from the final result text.

    Args:
        result: The final result string.

    Returns:
        List of citation dictionaries.
    """
    from kiva.nodes.synthesize import extract_citations

    return extract_citations(result)
