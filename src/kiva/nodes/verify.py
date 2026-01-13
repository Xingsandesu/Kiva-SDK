"""Verification graph nodes for the Kiva SDK.

This module implements the verification nodes for the dual-layer verification
architecture:
1. verify_worker_output - Verifies Worker Agent outputs against assigned tasks
2. verify_final_result - Verifies final results against user's original prompt
3. worker_retry - Handles retry logic for failed Worker verifications

The nodes use LangGraph Command for conditional routing based on verification
results.
"""

import time
from typing import Any, Literal

from langgraph.types import Command

from kiva.state import OrchestratorState
from kiva.verification import (
    RetryContext,
    VerificationResult,
    VerificationStatus,
    WorkerOutputVerifier,
)
from kiva.workflows.utils import emit_event


async def verify_worker_output(
    state: OrchestratorState,
) -> Command[Literal["synthesize_results", "worker_retry"]]:
    """Worker output verification node.

    Verifies each Worker Agent's output against its assigned task (NOT the
    user's original prompt). Uses LangGraph Command for conditional routing.

    Verification target: task_assignment.task (the specific task assigned to
    each Worker), NOT state.prompt (user's original request).

    Args:
        state: The current orchestrator state containing agent results and
            task assignments.

    Returns:
        Command routing to either:
        - "synthesize_results" if all verifications pass or max iterations reached
        - "worker_retry" if verification fails and retries remain

    Emits events:
        - worker_verification_start: When verification begins
        - worker_verification_passed: When all workers pass verification
        - worker_verification_failed: When one or more workers fail
        - worker_verification_max_reached: When max iterations exceeded
    """
    agent_results = state.get("agent_results", [])
    task_assignments = state.get("task_assignments", [])
    iteration = state.get("verification_iteration", 0)
    max_iterations = state.get("max_verification_iterations", 3)

    emit_event({
        "type": "worker_verification_start",
        "iteration": iteration,
        "agent_count": len(agent_results),
        "timestamp": time.time(),
    })

    # Create Worker output verifier
    verifier = WorkerOutputVerifier(
        model_name=state.get("model_name", "gpt-4o"),
        api_key=state.get("api_key"),
        base_url=state.get("base_url"),
        custom_verifiers=state.get("custom_verifiers", []),
    )

    results: list[VerificationResult] = []
    failed_agents: list[dict[str, Any]] = []

    for result in agent_results:
        # Skip results with errors
        if result.get("error"):
            continue

        # Get the assigned task for this Worker
        agent_id = result.get("agent_id", "")
        assigned_task = _get_assigned_task(
            agent_id, task_assignments, state.get("prompt", "")
        )

        # Verify Worker output against assigned task (NOT original prompt)
        verification = await verifier.verify(
            assigned_task=assigned_task,
            output=result.get("result", ""),
            schema=state.get("output_schema"),
        )
        results.append(verification)

        if verification.status != VerificationStatus.PASSED:
            failed_agents.append({
                "agent_id": agent_id,
                "assigned_task": assigned_task,
                "verification": verification,
                "previous_output": result.get("result", ""),
            })

    # Check if all passed
    all_passed = all(r.status == VerificationStatus.PASSED for r in results)

    if all_passed:
        emit_event({
            "type": "worker_verification_passed",
            "iteration": iteration,
            "results": [r.model_dump() for r in results],
            "timestamp": time.time(),
        })
        return Command(
            update={"verification_results": [r.model_dump() for r in results]},
            goto="synthesize_results",
        )

    # Check if max iterations reached
    if iteration >= max_iterations:
        emit_event({
            "type": "worker_verification_max_reached",
            "iteration": iteration,
            "failed_agents": [f["agent_id"] for f in failed_agents],
            "timestamp": time.time(),
        })
        return Command(
            update={
                "verification_results": [r.model_dump() for r in results],
                "verification_warning": (
                    "Max iterations reached for worker verification"
                ),
            },
            goto="synthesize_results",
        )

    # Build retry context for failed agents
    emit_event({
        "type": "worker_verification_failed",
        "iteration": iteration,
        "failed_agents": [f["agent_id"] for f in failed_agents],
        "timestamp": time.time(),
    })

    retry_context = _build_retry_context(
        iteration=iteration,
        max_iterations=max_iterations,
        failed_agents=failed_agents,
        agent_results=agent_results,
        task_assignments=task_assignments,
        original_prompt=state.get("prompt", ""),
    )

    return Command(
        update={
            "verification_results": [r.model_dump() for r in results],
            "retry_context": retry_context.model_dump(),
            "verification_iteration": iteration + 1,
        },
        goto="worker_retry",
    )


def _get_assigned_task(
    agent_id: str,
    task_assignments: list[dict[str, Any]],
    fallback_prompt: str,
) -> str:
    """Get the assigned task for a specific agent.

    Args:
        agent_id: The agent's identifier.
        task_assignments: List of task assignments from planning.
        fallback_prompt: Fallback to use if no assignment found.

    Returns:
        The assigned task string for the agent.
    """
    for assignment in task_assignments:
        if assignment.get("agent_id") == agent_id:
            return assignment.get("task", fallback_prompt)
    return fallback_prompt


def _build_retry_context(
    iteration: int,
    max_iterations: int,
    failed_agents: list[dict[str, Any]],
    agent_results: list[dict[str, Any]],
    task_assignments: list[dict[str, Any]],
    original_prompt: str,
) -> RetryContext:
    """Build retry context for failed Worker verifications.

    Creates a complete RetryContext containing:
    - The assigned task (NOT original prompt)
    - All previous outputs from failed agents
    - All previous rejection reasons
    - Complete task history

    Args:
        iteration: Current iteration number.
        max_iterations: Maximum allowed iterations.
        failed_agents: List of failed agent info with verification results.
        agent_results: All agent results from current execution.
        task_assignments: Task assignments from planning.
        original_prompt: User's original prompt (for reference only).

    Returns:
        RetryContext with complete history for retry operation.
    """
    # Collect previous outputs from all agents
    previous_outputs = [
        r.get("result", "") for r in agent_results if r.get("result")
    ]

    # Collect previous rejections from failed agents
    previous_rejections = [
        f["verification"] for f in failed_agents
        if isinstance(f.get("verification"), VerificationResult)
    ]

    # Get the primary failed agent's assigned task
    # (for single-agent retry, this is the task to retry)
    primary_task = original_prompt
    if failed_agents:
        primary_task = failed_agents[0].get("assigned_task", original_prompt)

    return RetryContext(
        iteration=iteration + 1,
        max_iterations=max_iterations,
        previous_outputs=previous_outputs,
        previous_rejections=previous_rejections,
        task_history=task_assignments,
        original_task=primary_task,
    )


def build_retry_prompt(retry_context: RetryContext) -> str:
    """Build a retry prompt from the retry context.

    Creates a prompt that includes:
    - The original assigned task
    - Previous rejection reasons
    - Improvement suggestions
    - Instructions to try a different approach

    Args:
        retry_context: The retry context with history.

    Returns:
        A formatted retry prompt string.
    """
    prompt_parts = [
        f"Original Task: {retry_context.original_task}",
        "",
        "Your previous attempts were rejected. Please try a DIFFERENT approach.",
        "",
        "Previous Rejection Reasons:",
    ]

    for i, rejection in enumerate(retry_context.previous_rejections):
        reason = rejection.rejection_reason or "Unknown reason"
        prompt_parts.append(f"  Attempt {i + 1}: {reason}")
        if rejection.improvement_suggestions:
            suggestions = ", ".join(rejection.improvement_suggestions)
            prompt_parts.append(f"    Suggestions: {suggestions}")

    prompt_parts.extend([
        "",
        "IMPORTANT: Do NOT repeat similar approaches. Try something fundamentally "
        "different.",
        "",
        f"Now, please complete the task: {retry_context.original_task}",
    ])

    return "\n".join(prompt_parts)
