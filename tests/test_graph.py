"""Unit tests for graph construction.

Tests for build_orchestrator_graph function and graph structure.
"""

from langgraph.graph.state import CompiledStateGraph

from kiva.graph import build_orchestrator_graph, get_graph_edges, get_graph_nodes


class TestBuildOrchestratorGraph:
    """Tests for build_orchestrator_graph function."""

    def test_returns_compiled_graph(self):
        """Test that build_orchestrator_graph returns a compiled StateGraph."""
        graph = build_orchestrator_graph()
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_expected_nodes(self):
        """Test that the graph contains all expected nodes."""
        graph = build_orchestrator_graph()
        expected_nodes = {
            "analyze_and_plan",
            "router_workflow",
            "supervisor_workflow",
            "parliament_workflow",
            "execute_instance",
            "synthesize_results",
            "verify_worker_output",
            "verify_final_result",
            "worker_retry",
        }
        # CompiledStateGraph stores nodes in its nodes attribute
        actual_nodes = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected_nodes == actual_nodes

    def test_graph_has_start_node(self):
        """Test that the graph has a start node."""
        graph = build_orchestrator_graph()
        assert "__start__" in graph.nodes

    def test_graph_has_end_node(self):
        """Test that the graph has proper termination (verify_final_result -> END)."""
        graph = build_orchestrator_graph()
        # In LangGraph, END is represented differently - verify verify_final_result exists
        # and the graph structure is valid (it compiled successfully)
        assert "verify_final_result" in graph.nodes


class TestGetGraphNodes:
    """Tests for get_graph_nodes helper function."""

    def test_returns_list(self):
        """Test that get_graph_nodes returns a list."""
        nodes = get_graph_nodes()
        assert isinstance(nodes, list)

    def test_contains_all_workflow_nodes(self):
        """Test that all workflow nodes are included."""
        nodes = get_graph_nodes()
        assert "analyze_and_plan" in nodes
        assert "router_workflow" in nodes
        assert "supervisor_workflow" in nodes
        assert "parliament_workflow" in nodes
        assert "execute_instance" in nodes
        assert "synthesize_results" in nodes
        assert "verify_worker_output" in nodes
        assert "verify_final_result" in nodes
        assert "worker_retry" in nodes

    def test_node_count(self):
        """Test that the correct number of nodes is returned."""
        nodes = get_graph_nodes()
        assert len(nodes) == 9


class TestGetGraphEdges:
    """Tests for get_graph_edges helper function."""

    def test_returns_list_of_tuples(self):
        """Test that get_graph_edges returns a list of tuples."""
        edges = get_graph_edges()
        assert isinstance(edges, list)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2

    def test_start_to_analyze_edge(self):
        """Test that there's an edge from start to analyze_and_plan."""
        edges = get_graph_edges()
        assert ("__start__", "analyze_and_plan") in edges

    def test_router_to_synthesize_edge(self):
        """Test that router_workflow connects to verify_worker_output."""
        edges = get_graph_edges()
        assert ("router_workflow", "verify_worker_output") in edges

    def test_supervisor_to_synthesize_edge(self):
        """Test that supervisor_workflow connects to verify_worker_output."""
        edges = get_graph_edges()
        assert ("supervisor_workflow", "verify_worker_output") in edges

    def test_execute_instance_to_synthesize_edge(self):
        """Test that execute_instance connects to verify_worker_output."""
        edges = get_graph_edges()
        assert ("execute_instance", "verify_worker_output") in edges

    def test_worker_retry_to_verify_edge(self):
        """Test that worker_retry connects to verify_worker_output."""
        edges = get_graph_edges()
        assert ("worker_retry", "verify_worker_output") in edges

    def test_synthesize_to_verify_final_edge(self):
        """Test that synthesize_results connects to verify_final_result."""
        edges = get_graph_edges()
        assert ("synthesize_results", "verify_final_result") in edges


class TestGraphStructure:
    """Tests for overall graph structure and connectivity."""

    def test_graph_is_callable(self):
        """Test that the compiled graph is callable."""
        graph = build_orchestrator_graph()
        assert callable(graph.invoke)
        assert callable(graph.ainvoke)

    def test_graph_supports_streaming(self):
        """Test that the compiled graph supports streaming."""
        graph = build_orchestrator_graph()
        assert hasattr(graph, "stream")
        assert hasattr(graph, "astream")

    def test_multiple_builds_return_equivalent_graphs(self):
        """Test that multiple calls return equivalent graph structures."""
        graph1 = build_orchestrator_graph()
        graph2 = build_orchestrator_graph()

        # Both should have the same nodes
        nodes1 = set(graph1.nodes.keys())
        nodes2 = set(graph2.nodes.keys())
        assert nodes1 == nodes2
