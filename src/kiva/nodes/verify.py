"""Verification graph nodes for the Kiva SDK.

This module implements the verification nodes for Worker output verification:
1. verify_worker_output - Verifies Worker Agent outputs against assigned tasks
2. worker_retry - Handles retry logic for failed Worker verifications

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
    VerificationStateCode,
    VerificationStatus,
    WorkerOutputVerifier,
)
from kiva.workflows.utils import (
    emit_event,
    execute_single_agent,
    get_agent_by_id,
)


async def verify_worker_output(
    state: OrchestratorState,
) -> Command[Literal["synthesize_results", "worker_retry"]]:
    """Worker output verification node.

    Verifies each Worker Agent's output against its assigned task (NOT the
    user's original prompt). Uses LangGraph Command for conditional routing.

    Verification target: task_assignment.task (the specific task assigned to
    each Worker), NOT state.prompt (user's original request).

    Graceful degradation:
    - If the verifier itself fails with an exception, verification is bypassed
      and the worker output is returned with a warning flag.
    - Error events are emitted with detailed failure information.

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
        - worker_verification_error: When verifier itself fails (graceful degradation)
    """
    agent_results = state.get("agent_results", [])
    task_assignments = state.get("task_assignments", [])
    iteration = state.get("verification_iteration", 0)
    max_iterations = state.get("max_verification_iterations", 3)

    base_timeline = state.get("verification_timeline", []) or []
    timeline: list[dict[str, Any]] = list(base_timeline)
    new_timeline_entries: list[dict[str, Any]] = []

    def emit_state_change(
        code: VerificationStateCode,
        *,
        scope: str = "worker",
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "scope": scope,
            "state": code.value,
            "timestamp": time.time(),
            "iteration": iteration,
            "max_iterations": max_iterations,
            "message": message,
            "details": details or {},
        }
        new_timeline_entries.append(entry)
        timeline.append(entry)
        emit_event(
            {
                "type": "verification_state_changed",
                "scope": scope,
                "state": code.value,
                "verification_status": {
                    "execution_id": state.get("execution_id", ""),
                    "scope": scope,
                    "state": code.value,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "timestamp": entry["timestamp"],
                    "message": message,
                    "details": entry["details"],
                    "timeline": timeline,
                },
                "timestamp": entry["timestamp"],
            }
        )

    emit_state_change(
        VerificationStateCode.INITIALIZING,
        message="worker verification initialized",
        details={"agent_count": len(agent_results)},
    )
    emit_state_change(
        VerificationStateCode.PREPROCESSING,
        message="worker verification preprocessing",
        details={"task_assignments": len(task_assignments)},
    )

    emit_event(
        {
            "type": "worker_verification_start",
            "iteration": iteration,
            "agent_count": len(agent_results),
            "max_iterations": max_iterations,
            "timestamp": time.time(),
        }
    )

    # Create Worker output verifier with graceful degradation on failure
    try:
        emit_state_change(
            VerificationStateCode.VERIFYING,
            message="creating worker verifier",
        )
        verifier = WorkerOutputVerifier(
            model_name=state.get("model_name", "gpt-4o"),
            api_key=state.get("api_key"),
            base_url=state.get("base_url"),
            custom_verifiers=state.get("custom_verifiers", []),
        )
    except Exception as e:
        # Graceful degradation: if verifier creation fails, bypass verification
        emit_state_change(
            VerificationStateCode.FAILURE_HANDLING,
            message="worker verifier creation failed",
            details={"error": str(e), "action": "bypass_verification"},
        )
        emit_state_change(
            VerificationStateCode.COMMITTING,
            message="committing worker verification (bypassed)",
        )
        emit_state_change(
            VerificationStateCode.COMPLETED,
            message="worker verification completed (bypassed)",
        )
        emit_event(
            {
                "type": "worker_verification_error",
                "error": f"Failed to create verifier: {e}",
                "action": "bypass_verification",
                "timestamp": time.time(),
            }
        )
        return Command(
            update={
                "verification_results": [],
                "verification_warning": (
                    f"Verification bypassed due to verifier error: {e}"
                ),
                "verification_state": {
                    "scope": "worker",
                    "state": VerificationStateCode.COMPLETED.value,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "warning": f"Verification bypassed due to verifier error: {e}",
                },
                "verification_timeline": new_timeline_entries,
            },
            goto="synthesize_results",
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
        # with graceful degradation on verification failure
        try:
            emit_state_change(
                VerificationStateCode.VERIFYING,
                message="verifying worker output",
                details={"agent_id": agent_id},
            )
            verification = await verifier.verify(
                assigned_task=assigned_task,
                output=result.get("result", ""),
                schema=state.get("output_schema"),
            )
        except Exception as e:
            # Graceful degradation: if verification fails, create SKIPPED result
            emit_state_change(
                VerificationStateCode.FAILURE_HANDLING,
                message="worker verification error",
                details={
                    "agent_id": agent_id,
                    "error": str(e),
                    "action": "skip_verification",
                },
            )
            emit_event(
                {
                    "type": "worker_verification_error",
                    "agent_id": agent_id,
                    "error": str(e),
                    "action": "skip_verification",
                    "timestamp": time.time(),
                }
            )
            verification = VerificationResult(
                status=VerificationStatus.SKIPPED,
                rejection_reason=f"Verification failed with error: {e}",
                validator_name="graceful_degradation",
            )

        results.append(verification)

        if verification.status == VerificationStatus.FAILED:
            failed_agents.append(
                {
                    "agent_id": agent_id,
                    "assigned_task": assigned_task,
                    "verification": verification,
                    "previous_output": result.get("result", ""),
                }
            )

    # Check if all passed (PASSED or SKIPPED are both acceptable for graceful
    # degradation)
    all_passed = all(
        r.status in (VerificationStatus.PASSED, VerificationStatus.SKIPPED)
        for r in results
    )

    # Check if any were skipped (for warning purposes)
    any_skipped = any(r.status == VerificationStatus.SKIPPED for r in results)

    if all_passed:
        emit_state_change(
            VerificationStateCode.COMMITTING,
            message="committing worker verification (passed)",
            details={"skipped": any_skipped},
        )
        event_data = {
            "type": "worker_verification_passed",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "results": [r.model_dump() for r in results],
            "timestamp": time.time(),
        }
        if any_skipped:
            event_data["warning"] = "Some verifications were skipped due to errors"
        emit_event(event_data)

        update_data: dict[str, Any] = {
            "verification_results": [r.model_dump() for r in results],
            "verification_state": {
                "scope": "worker",
                "state": VerificationStateCode.COMPLETED.value,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "passed": True,
                "skipped": any_skipped,
            },
            "verification_timeline": new_timeline_entries,
        }
        if any_skipped:
            update_data["verification_warning"] = (
                "Some verifications were skipped due to errors"
            )

        emit_state_change(
            VerificationStateCode.COMPLETED,
            message="worker verification completed (passed)",
            details={"skipped": any_skipped},
        )
        return Command(
            update=update_data,
            goto="synthesize_results",
        )

    # Check if max iterations reached
    if iteration >= max_iterations:
        emit_state_change(
            VerificationStateCode.FAILURE_HANDLING,
            message="worker verification max iterations reached",
            details={"failed_agents": [f["agent_id"] for f in failed_agents]},
        )
        emit_state_change(
            VerificationStateCode.ROLLBACK,
            message="rolling back to best-effort synthesis",
        )
        emit_state_change(
            VerificationStateCode.COMMITTING,
            message="committing worker verification (max reached)",
        )
        emit_state_change(
            VerificationStateCode.COMPLETED,
            message="worker verification completed (max reached)",
        )
        emit_event(
            {
                "type": "worker_verification_max_reached",
                "iteration": iteration,
                "max_iterations": max_iterations,
                "failed_agents": [f["agent_id"] for f in failed_agents],
                "timestamp": time.time(),
            }
        )
        return Command(
            update={
                "verification_results": [r.model_dump() for r in results],
                "verification_warning": (
                    "Max iterations reached for worker verification"
                ),
                "verification_state": {
                    "scope": "worker",
                    "state": VerificationStateCode.COMPLETED.value,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "passed": False,
                    "action": "best_effort_synthesis",
                    "failed_agents": [f["agent_id"] for f in failed_agents],
                },
                "verification_timeline": new_timeline_entries,
            },
            goto="synthesize_results",
        )

    # Build retry context for failed agents
    emit_state_change(
        VerificationStateCode.FAILURE_HANDLING,
        message="worker verification failed",
        details={"failed_agents": [f["agent_id"] for f in failed_agents]},
    )
    emit_event(
        {
            "type": "worker_verification_failed",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "failed_agents": [f["agent_id"] for f in failed_agents],
            "results": [r.model_dump() for r in results],
            "timestamp": time.time(),
        }
    )

    emit_state_change(
        VerificationStateCode.RETRY_WAITING,
        message="building retry context",
        details={"failed_agents": [f["agent_id"] for f in failed_agents]},
    )
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
            "verification_state": {
                "scope": "worker",
                "state": VerificationStateCode.RETRY_WAITING.value,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "failed_agents": [f["agent_id"] for f in failed_agents],
            },
            "verification_timeline": new_timeline_entries,
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
    previous_outputs = [r.get("result", "") for r in agent_results if r.get("result")]

    # Collect previous rejections from failed agents
    previous_rejections = [
        f["verification"]
        for f in failed_agents
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

    prompt_parts.extend(
        [
            "",
            "IMPORTANT: Do NOT repeat similar approaches. Try something fundamentally "
            "different.",
            "",
            f"Now, please complete the task: {retry_context.original_task}",
        ]
    )

    return "\n".join(prompt_parts)


async def worker_retry(
    state: OrchestratorState,
) -> dict[str, Any]:
    """Worker retry node.

    Handles retry logic for failed Worker verifications. This node is
    responsible for:
    1. Building a retry prompt with the assigned task and rejection context
    2. Re-executing the failed Worker Agent(s) with the retry prompt

    The retry prompt includes:
    - The original assigned task (NOT the user's original prompt)
    - Previous rejection reasons
    - Improvement suggestions
    - Instructions to try a different approach

    Args:
        state: The current orchestrator state containing retry_context,
            agents, and task_assignments.

    Returns:
        Dictionary with updated agent_results from the retry execution.

    Emits events:
        - retry_triggered: When retry execution begins
        - retry_completed: When retry execution completes

    """
    retry_context_dict = state.get("retry_context")
    if not retry_context_dict:
        base_timeline = state.get("verification_timeline", []) or []
        timeline: list[dict[str, Any]] = list(base_timeline)
        new_timeline_entries: list[dict[str, Any]] = []

        entry = {
            "scope": "retry",
            "state": VerificationStateCode.INITIALIZING.value,
            "timestamp": time.time(),
            "iteration": state.get("verification_iteration", 0),
            "max_iterations": state.get("max_verification_iterations", 3),
            "message": "retry skipped (no context)",
            "details": {"reason": "No retry context available"},
        }
        new_timeline_entries.append(entry)
        timeline.append(entry)
        emit_event(
            {
                "type": "verification_state_changed",
                "scope": "retry",
                "state": VerificationStateCode.ROLLBACK.value,
                "verification_status": {
                    "execution_id": state.get("execution_id", ""),
                    "scope": "retry",
                    "state": VerificationStateCode.ROLLBACK.value,
                    "iteration": entry["iteration"],
                    "max_iterations": entry["max_iterations"],
                    "timestamp": entry["timestamp"],
                    "message": entry["message"],
                    "details": entry["details"],
                    "timeline": timeline,
                },
                "timestamp": entry["timestamp"],
            }
        )
        emit_event(
            {
                "type": "retry_skipped",
                "reason": "No retry context available",
                "timestamp": time.time(),
            }
        )
        return {"agent_results": []}

    # Convert dict back to RetryContext if needed
    if isinstance(retry_context_dict, dict):
        retry_context = RetryContext(**retry_context_dict)
    else:
        retry_context = retry_context_dict

    # Build retry prompt with assigned task and rejection context
    retry_prompt = build_retry_prompt(retry_context)

    base_timeline = state.get("verification_timeline", []) or []
    timeline: list[dict[str, Any]] = list(base_timeline)
    new_timeline_entries: list[dict[str, Any]] = []

    def emit_state_change(
        code: VerificationStateCode,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "scope": "retry",
            "state": code.value,
            "timestamp": time.time(),
            "iteration": retry_context.iteration,
            "max_iterations": retry_context.max_iterations,
            "message": message,
            "details": details or {},
        }
        new_timeline_entries.append(entry)
        timeline.append(entry)
        emit_event(
            {
                "type": "verification_state_changed",
                "scope": "retry",
                "state": code.value,
                "verification_status": {
                    "execution_id": state.get("execution_id", ""),
                    "scope": "retry",
                    "state": code.value,
                    "iteration": retry_context.iteration,
                    "max_iterations": retry_context.max_iterations,
                    "timestamp": entry["timestamp"],
                    "message": message,
                    "details": entry["details"],
                    "timeline": timeline,
                },
                "timestamp": entry["timestamp"],
            }
        )

    emit_state_change(
        VerificationStateCode.RETRY_WAITING,
        message="retry prompt built",
    )

    # Get agents and task assignments from state
    agents = state.get("agents", [])
    task_assignments = state.get("task_assignments", [])
    execution_id = state.get("execution_id", "")

    emit_event(
        {
            "type": "retry_triggered",
            "iteration": retry_context.iteration,
            "retry_prompt": retry_prompt[:200],  # Truncate to avoid overly long events
            "agents": [a.get("agent_id", "") for a in task_assignments],
            "timestamp": time.time(),
        }
    )

    results: list[dict[str, Any]] = []

    # Re-execute agents with the retry prompt
    emit_state_change(
        VerificationStateCode.RETRY_RUNNING,
        message="retry executing agents",
        details={"agent_count": len(task_assignments)},
    )
    for assignment in task_assignments:
        agent_id = assignment.get("agent_id", "")
        agent = get_agent_by_id(agents, agent_id)

        if agent:
            recursion_limit = getattr(agent, "kiva_recursion_limit", None) or state.get(
                "worker_recursion_limit", 25
            )
            result = await execute_single_agent(
                agent=agent,
                agent_id=agent_id,
                task=retry_prompt,
                execution_id=execution_id,
                recursion_limit=recursion_limit,
            )
            results.append(result)
        else:
            # Agent not found, add error result
            results.append(
                {
                    "agent_id": agent_id,
                    "result": None,
                    "error": f"Agent '{agent_id}' not found for retry",
                }
            )

    emit_event(
        {
            "type": "retry_completed",
            "iteration": retry_context.iteration,
            "results_count": len(results),
            "agents": [a.get("agent_id", "") for a in task_assignments],
            "timestamp": time.time(),
        }
    )

    emit_state_change(
        VerificationStateCode.COMMITTING,
        message="committing retry results",
    )
    emit_state_change(
        VerificationStateCode.COMPLETED,
        message="retry completed",
        details={"results_count": len(results)},
    )

    return {"agent_results": results}
