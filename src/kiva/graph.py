"""Graph construction for the Kiva SDK.

This module builds the LangGraph state graph that orchestrates multi-agent
workflows. The graph structure enables automatic workflow selection,
dynamic agent instance spawning via Send API, and parallel execution.

Graph Structure:
    START -> analyze_and_plan -> [conditional routing with Send]
                                    |
                    +---------------+---------------+
                    |               |               |
                    v               v               v
            router_workflow  supervisor_workflow  parliament_workflow
                    |               |               |
                    |          [fan_out via Send]  |
                    |               |               |
                    |          agent_instance(s)   |
                    |               |               |
                    +---------------+---------------+
                                    |
                                    v
                            synthesize_results -> END

The Send API enables dynamic fan-out where the planner can spawn N instances
of the same agent definition, each with isolated context.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from kiva.nodes import analyze_and_plan, route_to_workflow, synthesize_results
from kiva.state import AgentInstanceState, OrchestratorState
from kiva.workflows import (
    parliament_workflow,
    router_workflow,
    should_continue_parliament,
    supervisor_workflow,
)
from kiva.workflows.executor import execute_instance_node


def _route_with_instances(state: OrchestratorState) -> list[Send] | str:
    """Route to workflow with optional instance fan-out via Send.

    This function implements dynamic routing based on the planning result.
    For parallel strategies (fan_out, map_reduce), it uses Send to spawn
    multiple agent instances that execute in parallel.

    Args:
        state: The current orchestrator state.

    Returns:
        Either a workflow node name string, or a list of Send objects
        for parallel instance execution.
    """
    parallel_strategy = state.get("parallel_strategy", "none")
    workflow = state.get("workflow_override") or state.get("workflow", "router")

    # For simple routing without parallelization, delegate to workflow
    if parallel_strategy == "none":
        return route_to_workflow(state)

    # Router workflow always handles its own execution
    if workflow == "router":
        return route_to_workflow(state)

    # For fan_out or map_reduce, use Send to spawn instances
    task_assignments = state.get("task_assignments", [])
    if not task_assignments:
        return route_to_workflow(state)

    sends = []
    execution_id = state.get("execution_id", "")

    from kiva.workflows.utils import (
        create_instance_context,
        generate_instance_id,
    )

    for assignment in task_assignments:
        agent_id = assignment.get("agent_id", "agent_0")
        task = assignment.get("task", state.get("prompt", ""))
        instances = assignment.get("instances", 1)
        base_context = assignment.get("instance_context", {})

        # Create Send for each instance (even if agent doesn't exist)
        # The execute_instance node will handle missing agents gracefully
        for i in range(instances):
            instance_id = generate_instance_id(execution_id, agent_id, i)
            context = create_instance_context(instance_id, agent_id, task, base_context)

            instance_state = AgentInstanceState(
                instance_id=instance_id,
                agent_id=agent_id,
                task=task,
                context=context,
                execution_id=execution_id,
                model_name=state.get("model_name", "gpt-4o"),
                api_key=state.get("api_key"),
                base_url=state.get("base_url"),
                worker_max_iterations=state.get("worker_max_iterations", 100),
                max_retries=state.get("max_retries", 3),
            )
            sends.append(Send("execute_instance", instance_state))

    # If no sends created, fall back to workflow routing
    if not sends:
        return route_to_workflow(state)

    return sends


def _collect_instance_results(state: OrchestratorState) -> str:
    """Determine next step after instance execution.

    Args:
        state: Current state with accumulated instance results.

    Returns:
        Next node name (synthesize_results).
    """
    return "synthesize_results"


def build_orchestrator_graph() -> StateGraph:
    """Build and compile the orchestrator state graph.

    Creates a LangGraph StateGraph with nodes for task analysis, workflow
    execution, instance execution, and result synthesis. The graph uses
    conditional edges and Send API to route to the appropriate workflow
    and spawn parallel agent instances.

    Returns:
        A compiled StateGraph ready for execution.

    Example:
        >>> graph = build_orchestrator_graph()
        >>> async for chunk in graph.astream(initial_state):
        ...     process(chunk)
    """
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("analyze_and_plan", analyze_and_plan)
    graph.add_node("router_workflow", router_workflow)
    graph.add_node("supervisor_workflow", supervisor_workflow)
    graph.add_node("parliament_workflow", parliament_workflow)
    graph.add_node("execute_instance", execute_instance_node)
    graph.add_node("synthesize_results", synthesize_results)

    # Add edges
    graph.add_edge(START, "analyze_and_plan")

    # Conditional routing with Send support for parallel instances
    graph.add_conditional_edges(
        "analyze_and_plan",
        _route_with_instances,
        {
            "router_workflow": "router_workflow",
            "supervisor_workflow": "supervisor_workflow",
            "parliament_workflow": "parliament_workflow",
            # execute_instance is handled via Send, not direct routing
        },
    )

    # Instance execution flows to synthesis
    graph.add_edge("execute_instance", "synthesize_results")

    # Workflow edges
    graph.add_edge("router_workflow", "synthesize_results")
    graph.add_edge("supervisor_workflow", "synthesize_results")
    graph.add_conditional_edges(
        "parliament_workflow",
        should_continue_parliament,
        {
            "synthesize": "synthesize_results",
            "parliament_workflow": "parliament_workflow",
        },
    )
    graph.add_edge("synthesize_results", END)

    return graph.compile()


def get_graph_nodes() -> list[str]:
    """Get the list of node names in the orchestrator graph.

    Returns:
        List of node name strings.
    """
    return [
        "analyze_and_plan",
        "router_workflow",
        "supervisor_workflow",
        "parliament_workflow",
        "execute_instance",
        "synthesize_results",
    ]


def get_graph_edges() -> list[tuple[str, str]]:
    """Get the list of static edges in the orchestrator graph.

    Note: This does not include conditional edges or Send-based edges.

    Returns:
        List of (source, target) edge tuples.
    """
    return [
        ("__start__", "analyze_and_plan"),
        ("router_workflow", "synthesize_results"),
        ("supervisor_workflow", "synthesize_results"),
        ("execute_instance", "synthesize_results"),
        ("synthesize_results", "__end__"),
    ]
