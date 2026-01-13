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
    FinalResultVerifier,
    RetryContext,
    VerificationResult,
    VerificationStatus,
    WorkerOutputVerifier,
)
from kiva.workflows.utils import emit_event

# Sentinel for END state in LangGraph
END = "__end__"


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


async def verify_final_result(
    state: OrchestratorState,
) -> Command[Literal["__end__", "analyze_and_plan"]]:
    """Final result verification node.

    Verifies whether the synthesized final result adequately satisfies the
    user's original prompt. The verification target is state.prompt (the
    user's original input), NOT the individual task assignments.

    Failure handling strategy:
    - If verification fails and max iterations not reached: restart entire
      workflow from analyze_and_plan
    - If max iterations reached: return failure summary explaining the reason
      and the user's original input

    Args:
        state: The current orchestrator state containing final_result and
            original prompt.

    Returns:
        Command routing to either:
        - END if verification passes or max iterations reached
        - "analyze_and_plan" if verification fails and retries remain

    Emits events:
        - final_verification_start: When verification begins
        - final_verification_passed: When final result passes verification
        - final_verification_failed: When final result fails verification
        - final_verification_max_reached: When max workflow iterations exceeded
    """
    final_result = state.get("final_result", "")
    original_prompt = state.get("prompt", "")
    agent_results = state.get("agent_results", [])
    workflow_iteration = state.get("workflow_iteration", 0)
    max_workflow_iterations = state.get("max_workflow_iterations", 2)

    emit_event({
        "type": "final_verification_start",
        "iteration": workflow_iteration,
        "timestamp": time.time(),
    })

    # Create final result verifier
    verifier = FinalResultVerifier(
        model_name=state.get("model_name", "gpt-4o"),
        api_key=state.get("api_key"),
        base_url=state.get("base_url"),
        custom_verifiers=state.get("custom_verifiers", []),
    )

    # Verify final result against user's original prompt
    verification = await verifier.verify(
        original_prompt=original_prompt,
        final_result=final_result or "",
        worker_results=agent_results,
    )

    if verification.status == VerificationStatus.PASSED:
        emit_event({
            "type": "final_verification_passed",
            "iteration": workflow_iteration,
            "timestamp": time.time(),
        })
        return Command(
            update={"final_verification_result": verification.model_dump()},
            goto=END,
        )

    # Check if max iterations reached
    if workflow_iteration >= max_workflow_iterations:
        # Generate failure summary explaining the reason and user's original input
        failure_summary = _generate_failure_summary(
            original_prompt=original_prompt,
            rejection_reason=verification.rejection_reason,
            improvement_suggestions=verification.improvement_suggestions,
            attempts=workflow_iteration + 1,
        )

        emit_event({
            "type": "final_verification_max_reached",
            "iteration": workflow_iteration,
            "reason": verification.rejection_reason,
            "failure_summary": failure_summary,
            "timestamp": time.time(),
        })

        return Command(
            update={
                "final_result": failure_summary,
                "final_verification_result": verification.model_dump(),
                "final_verification_warning": "Max workflow iterations reached",
            },
            goto=END,
        )

    # Restart entire workflow from analyze_and_plan
    emit_event({
        "type": "final_verification_failed",
        "iteration": workflow_iteration,
        "reason": verification.rejection_reason,
        "action": "restart_workflow",
        "timestamp": time.time(),
    })

    # Build previous attempts history
    previous_attempts = state.get("previous_workflow_attempts", [])
    previous_attempts = list(previous_attempts)  # Make a copy
    previous_attempts.append({
        "iteration": workflow_iteration,
        "final_result": final_result,
        "rejection_reason": verification.rejection_reason,
        "agent_results": agent_results,
    })

    # Build retry instruction for analyze_and_plan
    retry_instruction = _build_workflow_retry_instruction(
        original_prompt=original_prompt,
        rejection_reason=verification.rejection_reason,
        previous_attempts=previous_attempts,
    )

    return Command(
        update={
            "final_verification_result": verification.model_dump(),
            "workflow_iteration": workflow_iteration + 1,
            "previous_workflow_attempts": previous_attempts,
            # Reset workflow-related state for restart
            "agent_results": [],
            "verification_results": [],
            "verification_iteration": 0,
            "final_result": None,
            # Add retry instruction to guide different approach
            "retry_instruction": retry_instruction,
        },
        goto="analyze_and_plan",
    )


def _generate_failure_summary(
    original_prompt: str,
    rejection_reason: str | None,
    improvement_suggestions: list[str],
    attempts: int,
) -> str:
    """Generate a failure summary when max iterations reached.

    Creates a concise summary explaining why the task could not be completed,
    including the original user request and suggestions for improvement.

    Args:
        original_prompt: The user's original request.
        rejection_reason: The reason for the final rejection.
        improvement_suggestions: List of suggestions for improvement.
        attempts: Total number of attempts made.

    Returns:
        A formatted failure summary string.
    """
    summary_parts = [
        "## Task Could Not Fully Meet Requirements",
        "",
        f"**Original User Request:** {original_prompt}",
        "",
        f"**Attempts Made:** {attempts}",
        "",
        f"**Rejection Reason:** {rejection_reason or 'Unknown reason'}",
    ]

    if improvement_suggestions:
        summary_parts.extend([
            "",
            "**Improvement Suggestions:**",
            *[f"- {s}" for s in improvement_suggestions],
        ])

    summary_parts.extend([
        "",
        "---",
        "*The system has made its best effort to complete the task but could not "
        "fully satisfy all requirements.*",
        "*Suggestion: Please try simplifying your request or providing more "
        "specific guidance.*",
    ])

    return "\n".join(summary_parts)


def _build_workflow_retry_instruction(
    original_prompt: str,
    rejection_reason: str | None,
    previous_attempts: list[dict[str, Any]],
) -> str:
    """Build workflow retry instruction for analyze_and_plan.

    Creates an instruction that guides the system to try a fundamentally
    different approach when restarting the workflow.

    Args:
        original_prompt: The user's original request.
        rejection_reason: The reason for the current rejection.
        previous_attempts: History of previous workflow attempts.

    Returns:
        A formatted retry instruction string.
    """
    instruction_parts = [
        "IMPORTANT: This is a RETRY of the entire workflow.",
        "",
        f"Original user request: {original_prompt}",
        "",
        "Previous attempt(s) failed verification:",
    ]

    for i, attempt in enumerate(previous_attempts):
        reason = attempt.get("rejection_reason", "Unknown reason")
        instruction_parts.append(f"  Attempt {i + 1}: {reason}")

    instruction_parts.extend([
        "",
        "Please try a FUNDAMENTALLY DIFFERENT approach:",
        "- Consider different agent assignments",
        "- Try alternative task decomposition",
        "- Use different reasoning strategies",
        "",
        "DO NOT repeat the same approach that failed before.",
    ])

    return "\n".join(instruction_parts)
