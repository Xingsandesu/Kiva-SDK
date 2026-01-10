"""Analyze and plan node for the Lead Agent.

This module implements the task analysis and workflow selection logic.
The Lead Agent examines the user's request, assesses complexity, and
determines the optimal workflow pattern.
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kiva.state import OrchestratorState

ANALYZE_SYSTEM_PROMPT = """You are a task coordinator. Analyze user requests, \
assess complexity, and select the best workflow.

Complexity assessment:
- simple: Single domain, direct Q&A, requires one expert
- medium: Multiple experts collaborating, relatively independent tasks
- complex: Requires reasoning, verification, possible conflicts, needs iteration

Workflow selection:
- router: Simple tasks, route to single most appropriate agent
- supervisor: Medium complexity, parallel calls to multiple agents
- parliament: Complex reasoning, iterative validation and conflict resolution

Available Worker Agents:
{agent_descriptions}

Output JSON format:
{{
    "complexity": "simple|medium|complex",
    "workflow": "router|supervisor|parliament",
    "reasoning": "Your analysis",
    "task_assignments": [
        {{"agent_id": "agent_name", "task": "Specific task description"}}
    ]
}}
"""


def _get_agent_descriptions(agents: list) -> str:
    """Extract descriptions from agents for the system prompt.

    Args:
        agents: List of agent instances.

    Returns:
        Formatted string of agent names and descriptions.
    """
    if not agents:
        return "No agents available"
    return "\n".join(
        f"- {getattr(a, 'name', None) or f'agent_{i}'}: "
        f"{getattr(a, 'description', 'No description available')}"
        for i, a in enumerate(agents)
    )


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed dictionary, or default values if parsing fails.
    """
    if match := re.search(r"```(?:json)?\s*([\s\S]*?)```", content):
        content = match.group(1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "complexity": "simple",
            "workflow": "router",
            "reasoning": "Failed to parse LLM response, defaulting to router workflow",
            "task_assignments": [],
        }


async def analyze_and_plan(state: OrchestratorState) -> dict[str, Any]:
    """Lead Agent analyzes user intent and selects the appropriate workflow.

    Examines the user's prompt, assesses task complexity, and determines
    which workflow pattern (router, supervisor, or parliament) is most
    suitable. Also generates task assignments for worker agents.

    Args:
        state: The current orchestrator state containing prompt and agents.

    Returns:
        Dictionary with complexity, workflow, task_assignments, and messages.
    """
    from langchain_openai import ChatOpenAI

    model_kwargs = {"model": state.get("model_name", "gpt-4o")}
    if api_key := state.get("api_key"):
        model_kwargs["api_key"] = api_key
    if base_url := state.get("base_url"):
        model_kwargs["base_url"] = base_url

    model = ChatOpenAI(**model_kwargs)
    agents = state.get("agents", [])

    messages = [
        SystemMessage(
            content=ANALYZE_SYSTEM_PROMPT.format(
                agent_descriptions=_get_agent_descriptions(agents)
            )
        ),
        HumanMessage(content=state["prompt"]),
    ]

    response: AIMessage = await model.ainvoke(messages)
    result = _parse_json_response(response.content)

    complexity = result.get("complexity", "simple")
    if complexity not in ("simple", "medium", "complex"):
        complexity = "simple"

    # workflow_override takes priority over LLM analysis
    workflow = state.get("workflow_override") or result.get("workflow", "router")
    if workflow not in ("router", "supervisor", "parliament"):
        workflow = "router"

    task_assignments = result.get("task_assignments", [])
    if not task_assignments and agents:
        agent_id = getattr(agents[0], "name", None) or "agent_0"
        task_assignments = [{"agent_id": agent_id, "task": state["prompt"]}]

    return {
        "complexity": complexity,
        "workflow": workflow,
        "task_assignments": task_assignments,
        "messages": [response],
    }
