"""Graph construction for the Kiva SDK.

This module builds the LangGraph state graph that orchestrates multi-agent
workflows. The graph structure enables automatic workflow selection and
execution based on task complexity.

Graph Structure:
    START -> analyze_and_plan -> [conditional routing]
                                    |
                    +---------------+---------------+
                    |               |               |
                    v               v               v
            router_workflow  supervisor_workflow  parliament_workflow
                    |               |               |
                    +---------------+---------------+
                                    |
                                    v
                            synthesize_results -> END
"""

from langgraph.graph import END, START, StateGraph

from kiva.nodes import analyze_and_plan, route_to_workflow, synthesize_results
from kiva.state import OrchestratorState
from kiva.workflows import (
    parliament_workflow,
    router_workflow,
    should_continue_parliament,
    supervisor_workflow,
)


def build_orchestrator_graph() -> StateGraph:
    """Build and compile the orchestrator state graph.

    Creates a LangGraph StateGraph with nodes for task analysis, workflow
    execution, and result synthesis. The graph uses conditional edges to
    route to the appropriate workflow based on task complexity.

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
    graph.add_node("synthesize_results", synthesize_results)

    # Add edges
    graph.add_edge(START, "analyze_and_plan")
    graph.add_conditional_edges(
        "analyze_and_plan",
        route_to_workflow,
        {
            "router_workflow": "router_workflow",
            "supervisor_workflow": "supervisor_workflow",
            "parliament_workflow": "parliament_workflow",
        },
    )
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
        "synthesize_results",
    ]


def get_graph_edges() -> list[tuple[str, str]]:
    """Get the list of static edges in the orchestrator graph.

    Note: This does not include conditional edges.

    Returns:
        List of (source, target) edge tuples.
    """
    return [
        ("__start__", "analyze_and_plan"),
        ("router_workflow", "synthesize_results"),
        ("supervisor_workflow", "synthesize_results"),
        ("synthesize_results", "__end__"),
    ]
