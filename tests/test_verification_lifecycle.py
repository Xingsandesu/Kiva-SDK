import pytest

from kiva.nodes.verify import verify_worker_output
from kiva.verification import VerificationResult, VerificationStatus


@pytest.mark.asyncio
async def test_verification_state_changed_emitted_on_worker_pass(monkeypatch):
    emitted: list[dict] = []

    def fake_emit(event: dict) -> None:
        emitted.append(event)

    class FakeWorkerVerifier:
        def __init__(self, *args, **kwargs):
            pass

        async def verify(self, *args, **kwargs) -> VerificationResult:
            return VerificationResult(status=VerificationStatus.PASSED)

    monkeypatch.setattr("kiva.nodes.verify.emit_event", fake_emit)
    monkeypatch.setattr("kiva.nodes.verify.WorkerOutputVerifier", FakeWorkerVerifier)

    state = {
        "execution_id": "exec_1",
        "agent_results": [{"agent_id": "a1", "result": "ok"}],
        "task_assignments": [{"agent_id": "a1", "task": "do"}],
        "verification_iteration": 0,
        "max_verification_iterations": 3,
        "prompt": "p",
        "model_name": "m",
        "api_key": None,
        "base_url": None,
        "custom_verifiers": [],
        "output_schema": None,
        "verification_timeline": [],
    }

    cmd = await verify_worker_output(state)

    assert cmd.goto == "synthesize_results"
    states = [
        e.get("verification_status", {}).get("state")
        for e in emitted
        if e.get("type") == "verification_state_changed"
    ]
    assert "initializing" in states
    assert "preprocessing" in states
    assert "committing" in states
    assert "completed" in states

    status_events = [
        e for e in emitted if e.get("type") == "verification_state_changed"
    ]
    assert status_events
    last = status_events[-1]["verification_status"]
    assert last["execution_id"] == "exec_1"
    assert last["scope"] == "worker"
    assert isinstance(last["timeline"], list)


@pytest.mark.asyncio
async def test_verification_state_changed_emitted_on_worker_retry(monkeypatch):
    emitted: list[dict] = []

    def fake_emit(event: dict) -> None:
        emitted.append(event)

    class FakeWorkerVerifier:
        def __init__(self, *args, **kwargs):
            pass

        async def verify(self, *args, **kwargs) -> VerificationResult:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                rejection_reason="bad",
                improvement_suggestions=["fix"],
            )

    monkeypatch.setattr("kiva.nodes.verify.emit_event", fake_emit)
    monkeypatch.setattr("kiva.nodes.verify.WorkerOutputVerifier", FakeWorkerVerifier)

    state = {
        "execution_id": "exec_2",
        "agent_results": [{"agent_id": "a1", "result": "nope"}],
        "task_assignments": [{"agent_id": "a1", "task": "do"}],
        "verification_iteration": 0,
        "max_verification_iterations": 3,
        "prompt": "p",
        "model_name": "m",
        "api_key": None,
        "base_url": None,
        "custom_verifiers": [],
        "output_schema": None,
        "verification_timeline": [],
    }

    cmd = await verify_worker_output(state)

    assert cmd.goto == "worker_retry"
    states = [
        e.get("verification_status", {}).get("state")
        for e in emitted
        if e.get("type") == "verification_state_changed"
    ]
    assert "failure_handling" in states
    assert "retry_waiting" in states

