"""Graph construction for the Kiva SDK.

This module builds the LangGraph state graph that orchestrates multi-agent
workflows. The graph structure enables automatic workflow selection,
dynamic agent instance spawning via Send API, parallel execution, and
dual-layer verification.

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
                          verify_worker_output
                                   |
                       +-----------+-----------+
                       |                       |
                   [passed]               [failed]
                       |                       |
                       v                       v
               synthesize_results         worker_retry
                       |
                       v
                      END

The Send API enables dynamic fan-out where the planner can spawn N instances
of the same agent definition, each with isolated context.

Verification Architecture:
    1. Worker Output Verification: Validates each Worker Agent's output against
       its assigned task (NOT the user's original prompt).
    2. Retry Mechanism:
       - Worker level: Retry failed workers with rejection context
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from kiva.nodes import (
    analyze_and_plan,
    route_to_workflow,
    synthesize_results,
    verify_worker_output,
    worker_retry,
)
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
        Next node name (verify_worker_output).
    """
    return "verify_worker_output"


def build_orchestrator_graph() -> StateGraph:
    """Build and compile the orchestrator state graph.

    Creates a LangGraph StateGraph with nodes for task analysis, workflow
    execution, instance execution, result synthesis, and verification. The
    graph uses conditional edges and Send API to route to the appropriate
    workflow and spawn parallel agent instances.

    The graph implements a dual-layer verification architecture:
    1. Worker output verification: Verifies each Worker Agent's output against
       its assigned task after workflow execution.
    2. Final result verification: Verifies the synthesized result against the
       user's original prompt after synthesis.

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
    # Add verification nodes
    graph.add_node("verify_worker_output", verify_worker_output)
    graph.add_node("worker_retry", worker_retry)

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

    # Instance execution flows to worker verification
    graph.add_edge("execute_instance", "verify_worker_output")

    # Workflow edges - all workflows flow to worker verification
    graph.add_edge("router_workflow", "verify_worker_output")
    graph.add_edge("supervisor_workflow", "verify_worker_output")
    graph.add_conditional_edges(
        "parliament_workflow",
        should_continue_parliament,
        {
            "synthesize": "verify_worker_output",
            "parliament_workflow": "parliament_workflow",
        },
    )

    # Worker verification uses Command to route to synthesize or retry
    # (no explicit edge needed - Command handles routing)

    # Worker retry flows back to worker verification
    graph.add_edge("worker_retry", "verify_worker_output")

    # Synthesis flows to END (final verification removed)
    graph.add_edge("synthesize_results", END)

    # Final verification uses Command to route to END or restart workflow
    # (no explicit edge needed - Command handles routing)

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
        "verify_worker_output",
        "worker_retry",
    ]


def get_graph_edges() -> list[tuple[str, str]]:
    """Get the list of static edges in the orchestrator graph.

    Note: This does not include conditional edges or Send-based edges.
    Also excludes edges controlled by Command (verify_worker_output and
    verify_final_result use Command for dynamic routing).

    Returns:
        List of (source, target) edge tuples.
    """
    return [
        ("__start__", "analyze_and_plan"),
        ("router_workflow", "verify_worker_output"),
        ("supervisor_workflow", "verify_worker_output"),
        ("execute_instance", "verify_worker_output"),
        ("worker_retry", "verify_worker_output"),
        ("synthesize_results", "__end__"),
    ]
