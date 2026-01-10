"""Unit tests for graph nodes."""

from kiva.nodes.analyze import _get_agent_descriptions, _parse_json_response
from kiva.nodes.router import route_to_workflow
from kiva.nodes.synthesize import extract_citations


class TestRouteToWorkflow:
    """Tests for route_to_workflow conditional router."""

    def test_route_to_router_workflow(self):
        """Test routing to router workflow."""
        state = {"workflow": "router"}
        result = route_to_workflow(state)
        assert result == "router_workflow"

    def test_route_to_supervisor_workflow(self):
        """Test routing to supervisor workflow."""
        state = {"workflow": "supervisor"}
        result = route_to_workflow(state)
        assert result == "supervisor_workflow"

    def test_route_to_parliament_workflow(self):
        """Test routing to parliament workflow."""
        state = {"workflow": "parliament"}
        result = route_to_workflow(state)
        assert result == "parliament_workflow"

    def test_workflow_override_takes_priority(self):
        """Test that workflow_override takes priority over workflow."""
        state = {"workflow": "router", "workflow_override": "supervisor"}
        result = route_to_workflow(state)
        assert result == "supervisor_workflow"

    def test_workflow_override_parliament(self):
        """Test workflow_override with parliament."""
        state = {"workflow": "router", "workflow_override": "parliament"}
        result = route_to_workflow(state)
        assert result == "parliament_workflow"

    def test_default_to_router_when_missing(self):
        """Test default to router workflow when workflow is missing."""
        state = {}
        result = route_to_workflow(state)
        assert result == "router_workflow"

    def test_default_to_router_for_unknown_workflow(self):
        """Test default to router workflow for unknown workflow value."""
        state = {"workflow": "unknown_workflow"}
        result = route_to_workflow(state)
        assert result == "router_workflow"


class TestExtractCitations:
    """Tests for extract_citations function."""

    def test_extract_bracket_citations(self):
        """Test extracting [agent_id] style citations."""
        text = "According to [agent_1], the answer is 42. [agent_2] confirms this."
        citations = extract_citations(text)
        sources = [c["source"] for c in citations]
        assert "agent_1" in sources
        assert "agent_2" in sources

    def test_extract_according_to_citations(self):
        """Test extracting 'According to X' style citations."""
        text = "According to SearchAgent, the data shows growth. Based on AnalysisAgent, this is significant."
        citations = extract_citations(text)
        sources = [c["source"] for c in citations]
        assert "SearchAgent" in sources
        assert "AnalysisAgent" in sources

    def test_no_duplicate_citations(self):
        """Test that duplicate citations are not included."""
        text = "[agent_1] says yes. [agent_1] also says maybe."
        citations = extract_citations(text)
        sources = [c["source"] for c in citations]
        assert sources.count("agent_1") == 1

    def test_empty_text_returns_empty_list(self):
        """Test that empty text returns empty citations list."""
        citations = extract_citations("")
        assert citations == []

    def test_filter_common_brackets(self):
        """Test that common non-citation brackets are filtered."""
        text = "[note] This is a note. [warning] Be careful."
        citations = extract_citations(text)
        sources = [c["source"] for c in citations]
        assert "note" not in sources
        assert "warning" not in sources


class TestParseJsonResponse:
    """Tests for _parse_json_response helper function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        content = '{"complexity": "medium", "workflow": "supervisor", "reasoning": "test", "task_assignments": []}'
        result = _parse_json_response(content)
        assert result["complexity"] == "medium"
        assert result["workflow"] == "supervisor"

    def test_parse_json_in_markdown_code_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        content = """```json
{"complexity": "complex", "workflow": "parliament", "reasoning": "test", "task_assignments": []}
```"""
        result = _parse_json_response(content)
        assert result["complexity"] == "complex"
        assert result["workflow"] == "parliament"

    def test_parse_json_in_plain_code_block(self):
        """Test parsing JSON wrapped in plain code block."""
        content = """```
{"complexity": "simple", "workflow": "router", "reasoning": "test", "task_assignments": []}
```"""
        result = _parse_json_response(content)
        assert result["complexity"] == "simple"
        assert result["workflow"] == "router"

    def test_invalid_json_returns_defaults(self):
        """Test that invalid JSON returns default values."""
        content = "This is not valid JSON"
        result = _parse_json_response(content)
        assert result["complexity"] == "simple"
        assert result["workflow"] == "router"
        assert "task_assignments" in result


class TestGetAgentDescriptions:
    """Tests for _get_agent_descriptions helper function."""

    def test_empty_agents_list(self):
        """Test with empty agents list."""
        result = _get_agent_descriptions([])
        assert result == "No agents available"

    def test_agents_with_name_and_description(self):
        """Test agents with name and description attributes."""

        class MockAgent:
            def __init__(self, name, description):
                self.name = name
                self.description = description

        agents = [
            MockAgent("search_agent", "Searches the web"),
            MockAgent("calc_agent", "Performs calculations"),
        ]
        result = _get_agent_descriptions(agents)
        assert "search_agent" in result
        assert "Searches the web" in result
        assert "calc_agent" in result
        assert "Performs calculations" in result

    def test_agents_without_attributes(self):
        """Test agents without name/description attributes."""

        class MockAgent:
            pass

        agents = [MockAgent(), MockAgent()]
        result = _get_agent_descriptions(agents)
        assert "agent_0" in result
        assert "agent_1" in result
        assert "No description available" in result
