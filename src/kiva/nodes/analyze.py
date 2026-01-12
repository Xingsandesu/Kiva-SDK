"""Analyze and plan node for the Lead Agent.

This module implements the task analysis, intent detection, and workflow
selection logic. The Lead Agent examines the user's request, assesses
complexity, determines the optimal workflow pattern, and decides on
parallelization strategy including agent instance counts.
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kiva.state import OrchestratorState, PlanningResult, TaskAssignment

ANALYZE_SYSTEM_PROMPT = """You are a task coordinator with advanced planning capabilities.
Analyze user requests, assess complexity, select the best workflow, and determine
parallelization strategy.

## Complexity Assessment
- simple: Single domain, direct Q&A, requires one expert
- medium: Multiple experts collaborating, relatively independent tasks
- complex: Requires reasoning, verification, possible conflicts, needs iteration

## Workflow Selection
- router: Simple tasks, route to single most appropriate agent
- supervisor: Medium complexity, parallel calls to multiple agents
- parliament: Complex reasoning, iterative validation and conflict resolution

## Parallelization Strategy
You can spawn MULTIPLE INSTANCES of the same agent for parallel subtask execution.

- none: No parallelization needed, single agent instance
- fan_out: Spawn N instances of one or more agents for independent subtasks
- map_reduce: Split task into subtasks, process in parallel, then aggregate

For each task_assignment, you can specify:
- agent_id: Which agent definition to use
- task: The specific subtask
- instances: How many parallel instances to spawn (default: 1)

Example scenarios for multi-instance:
1. "Search for info about 5 different topics" → spawn 5 instances of search_agent
2. "Analyze these 3 documents" → spawn 3 instances of analyzer_agent
3. "Get weather for NYC, LA, and Chicago" → spawn 3 instances of weather_agent

## Available Worker Agents
{agent_descriptions}

## Output JSON Format
{{
    "complexity": "simple|medium|complex",
    "workflow": "router|supervisor|parliament",
    "parallel_strategy": "none|fan_out|map_reduce",
    "reasoning": "Your analysis including why this parallelization strategy",
    "task_assignments": [
        {{
            "agent_id": "agent_name",
            "task": "Specific task description",
            "instances": 1
        }}
    ],
    "total_instances": <total number of instances across all assignments>
}}

IMPORTANT:
- Use instances > 1 only when the task can be meaningfully split into
  independent subtasks
- Each instance will have its own isolated context/scratchpad
- Consider max_parallel_agents limit when deciding instance counts
- For map_reduce, ensure subtasks can be aggregated meaningfully
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


def _parse_json_response(content: str) -> PlanningResult:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed PlanningResult dictionary.
    """
    if match := re.search(r"```(?:json)?\s*([\s\S]*?)```", content):
        content = match.group(1).strip()
    try:
        parsed = json.loads(content)
        return PlanningResult(
            complexity=parsed.get("complexity", "simple"),
            workflow=parsed.get("workflow", "router"),
            reasoning=parsed.get("reasoning", ""),
            task_assignments=parsed.get("task_assignments", []),
            parallel_strategy=parsed.get("parallel_strategy", "none"),
            total_instances=parsed.get("total_instances", 1),
        )
    except json.JSONDecodeError:
        return PlanningResult(
            complexity="simple",
            workflow="router",
            reasoning="Failed to parse LLM response, defaulting to router workflow",
            task_assignments=[],
            parallel_strategy="none",
            total_instances=1,
        )


def _normalize_task_assignments(
    assignments: list[dict], agents: list, prompt: str, max_parallel: int
) -> tuple[list[TaskAssignment], int]:
    """Normalize and validate task assignments.

    Args:
        assignments: Raw task assignments from LLM.
        agents: Available agent instances.
        prompt: Original user prompt as fallback.
        max_parallel: Maximum parallel instances allowed.

    Returns:
        Tuple of (normalized assignments, total instance count).
    """
    if not assignments and agents:
        agent_id = getattr(agents[0], "name", None) or "agent_0"
        return [TaskAssignment(agent_id=agent_id, task=prompt, instances=1)], 1

    normalized = []
    total_instances = 0

    for assignment in assignments:
        instances = min(assignment.get("instances", 1), max_parallel - total_instances)
        instances = max(1, instances)  # At least 1 instance

        if total_instances + instances > max_parallel:
            instances = max(1, max_parallel - total_instances)

        normalized.append(
            TaskAssignment(
                agent_id=assignment.get("agent_id", "agent_0"),
                task=assignment.get("task", prompt),
                instances=instances,
                instance_context=assignment.get("instance_context", {}),
            )
        )
        total_instances += instances

        if total_instances >= max_parallel:
            break

    return normalized, total_instances


async def analyze_and_plan(state: OrchestratorState) -> dict[str, Any]:
    """Lead Agent analyzes user intent and plans execution strategy.

    Examines the user's prompt, assesses task complexity, determines
    which workflow pattern (router, supervisor, or parliament) is most
    suitable, and decides on parallelization strategy including how many
    instances of each agent to spawn.

    Args:
        state: The current orchestrator state containing prompt and agents.

    Returns:
        Dictionary with complexity, workflow, task_assignments, parallel_strategy,
        total_instances, and messages.
    """
    from langchain_openai import ChatOpenAI

    model_kwargs = {"model": state.get("model_name", "gpt-4o")}
    if api_key := state.get("api_key"):
        model_kwargs["api_key"] = api_key
    if base_url := state.get("base_url"):
        model_kwargs["base_url"] = base_url

    model = ChatOpenAI(**model_kwargs)
    agents = state.get("agents", [])
    max_parallel = state.get("max_parallel_agents", 5)

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

    parallel_strategy = result.get("parallel_strategy", "none")
    if parallel_strategy not in ("none", "fan_out", "map_reduce"):
        parallel_strategy = "none"

    task_assignments, total_instances = _normalize_task_assignments(
        result.get("task_assignments", []),
        agents,
        state["prompt"],
        max_parallel,
    )

    return {
        "complexity": complexity,
        "workflow": workflow,
        "task_assignments": task_assignments,
        "parallel_strategy": parallel_strategy,
        "total_instances": total_instances,
        "messages": [response],
    }
