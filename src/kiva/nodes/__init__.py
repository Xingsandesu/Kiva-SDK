"""Graph nodes for the Kiva SDK.

This package provides the core graph nodes used in the orchestration workflow:

Nodes:
    analyze_and_plan: Lead agent analyzes task and selects workflow.
    route_to_workflow: Routes to the appropriate workflow based on analysis.
    synthesize_results: Combines agent outputs into a final response.

Functions:
    extract_citations: Extracts source citations from result text.
"""

from kiva.nodes.analyze import analyze_and_plan
from kiva.nodes.router import route_to_workflow
from kiva.nodes.synthesize import extract_citations, synthesize_results

__all__ = [
    "analyze_and_plan",
    "route_to_workflow",
    "synthesize_results",
    "extract_citations",
]
