# Output Verification (Worker-Level)

Kiva performs worker output verification after agents run. If verification fails, Kiva can retry workers with structured rejection reasons and improvement suggestions. If verification infrastructure fails, Kiva degrades gracefully (SKIPPED results) and continues.

## What Is Verified?

- Target: the assigned worker task (`task_assignment.task`), not the original user prompt
- Inputs: worker output text (and optional JSON schema)
- Outputs: a `VerificationResult` per worker and an aggregated decision

## Configuration

### Retry Limits

- Global default: `Kiva(max_iterations=...)`
- Per-agent override: `@kiva.agent(..., max_iterations=...)`
- Per-run override: `kiva.run(..., max_iterations=...)`

Priority order (highest to lowest):
- `kiva.run(max_iterations=...)`
- `@kiva.agent(..., max_iterations=...)`
- `Kiva(max_iterations=...)`

Example:

```python
from kiva import Kiva

kiva = Kiva(base_url="...", api_key="...", model="gpt-4o", max_iterations=3)

@kiva.agent("writer", "Writes content", max_iterations=5)
def writer(topic: str) -> str:
    return f"Draft about {topic}"

result = kiva.run("Write something", max_iterations=4)
```

## Custom Verifiers

Use `@kiva.verifier` to add project-specific validation logic.

- Signature:
  - `task: str`
  - `output: str`
  - `context: dict | None`
  - returns `VerificationResult`

Example:

```python
from kiva import Kiva, VerificationResult, VerificationStatus

kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")

@kiva.verifier("length_check", priority=10)
def length_check(task: str, output: str, context: dict | None = None) -> VerificationResult:
    if len(output) < 50:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason="Output too short",
            improvement_suggestions=["Provide a more detailed answer"],
            validator_name="length_check",
        )
    return VerificationResult(status=VerificationStatus.PASSED, validator_name="length_check")
```

## Pydantic Schema Validation

If you provide an output schema, Kiva can validate worker outputs against it (expects JSON output).

Example schema:

```python
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str
```

## Retry Context

When verification fails, Kiva builds a `RetryContext` containing:
- `iteration` / `max_iterations`
- `original_task`
- `previous_outputs`
- `previous_rejections` (`VerificationResult` list)
- `task_history`

Workers can use this context to try a different approach on retry.

## Events

Verification and retry events are emitted into the same run() stream used by downstream integrators (see `docs/run-streaming-api.md` for payload fields).

- Worker verification:
  - `worker_verification_start`
  - `worker_verification_passed`
  - `worker_verification_failed`
  - `worker_verification_max_reached`
  - `worker_verification_error`
- Retry:
  - `retry_triggered`
  - `retry_completed`
  - `retry_skipped`
- Lifecycle snapshot:
  - `verification_state_changed`

## Lifecycle State Codes

`verification_state_changed` includes a `verification_status` object with:
- `execution_id`, `scope`, `state`, `iteration`, `max_iterations`, `timestamp`
- `message`, `details`
- `timeline` (full timeline snapshot list)

Common state codes:
- `initializing`
- `preprocessing`
- `verifying`
- `retry_waiting`
- `retry_running`
- `failure_handling`
- `rollback`
- `committing`
- `completed`

