# Kiva run() Streaming API (console=false Integration Guide)

This guide documents the event stream produced by `Kiva.run(..., console=False)`, the event types and fields, state management recommendations, and usage examples. Downstream systems can consume events 1:1 to build their own state machines.

## Usage

Example (see `examples/01_high_level.py` lines 66–67):

```python
result = kiva.run("...", console=False)
for event in result:
    handle(event)  # consume all events and build your state machine
print("FINAL:", result.result())  # final_result
```

With `console=False`:
- `run()` returns an iterable of StreamEvent items
- Each event has `type`, `data`, `timestamp`, `agent_id`
- Consume all events, not only `token`

## Event Structure

Each event is a `StreamEvent`:
- `type: str` event type
- `data: dict` payload
- `timestamp: float` event time
- `agent_id: str | None` associated agent if any

Reference implementation: `src/kiva/events.py`

## Event Types and Fields

Events come from three sources:
- LLM streaming messages (`run._process_stream_chunk: messages`)
- Graph node updates (`run._process_node_update: updates`)
- Workflow-level emitted events (`emit_event: custom`)

Below is the full catalog of possible events and their `data` fields.

### Basic Events

- `token`
  - `data`: `{ "content": str, "execution_id": str }`
  - Notes: LLM streaming token; `agent_id` is current node (e.g. `router_workflow`)

- `workflow_selected`
  - `data`: `{ "workflow": str, "complexity": str|None, "task_assignments": list, "parallel_strategy": str, "total_instances": int, "execution_id": str }`
  - Notes: Planning complete; chosen workflow and parallel strategy

- `final_result`
  - `data`:
    - required: `{ "result": str, "citations": list, "execution_id": str }`
    - optional: `{ "partial_result": bool, "success_count": int, "failure_count": int, "failed_agents": list[str] }`
  - Notes: Synthesized result; may include partial statistics and citations

### Single-Agent Events

- `agent_start`
  - `data`: `{ "agent_id": str, "invocation_id": str, "execution_id": str, "task": str, "timestamp": float }`
  - Notes: Single agent execution start (router workflow)

- `agent_end`
  - `data`: `{ "agent_id": str, "invocation_id": str, "execution_id": str, "result": str|None, "timestamp": float }`
  - Notes: Single agent execution end; `result` has the output if success

### Parallel-Agent Events (Supervisor/Parliament)

- `parallel_start`
  - `data`: `{ "agent_ids": list[str], "iteration": int, "execution_id": str, "timestamp": float, "phase": "conflict_resolution"|None }`
  - Notes: Parallel agent execution (or conflict resolution) started

- `parallel_complete`
  - `data`: `{ "results": [ { "agent_id": str, "success": bool } ... ], "conflicts_found": int|None, "conflicts_remaining": int|None, "iteration": int, "execution_id": str, "timestamp": float }`
  - Notes: Parallel execution complete with success/conflict stats

### Instance-Level Events (Send API)

- `instance_spawn` / `instance_start` / `instance_end` / `instance_complete`
  - `data`: `{ "instance_id": str, "agent_id": str, "execution_id": str, "task": str|None, "result": str|None, "success": bool|None }`
  - Notes: Lifecycle events for parallel instances of the same agent

- `instance_result`
  - `data`: `{ "instance_id": str, "agent_id": str, "result": str|None, "error": str|None, "execution_id": str }`
  - Notes: Instance execution result (success or error)

- `parallel_instances_start` / `parallel_instances_complete`
  - `data`:
    - start: `{ "instance_count": int, "agent_ids": list[str], "execution_id": str }`
    - complete: `{ "results": [ { "agent_id": str, "instance_id": str, "success": bool } ... ], "execution_id": str, "timestamp": float }`

- `worker_retry_results`
  - `data`: `{ "results": list[dict], "execution_id": str }`
  - Notes: Result updates from retry node (observe retry outputs)

### Worker Verification Events

- `worker_verification_start`
  - `data`: `{ "iteration": int, "agent_count": int, "execution_id": str }`

- `worker_verification_passed`
  - `data`: `{ "iteration": int, "results": list[VerificationResult], "execution_id": str }`

- `worker_verification_failed`
  - `data`: `{ "iteration": int, "failed_agents": list[str], "execution_id": str }`

- `worker_verification_max_reached`
  - `data`: `{ "iteration": int, "failed_agents": list[str], "execution_id": str }`

- `worker_verification_error`
  - `data`: `{ "error": str, "action": str, "agent_id": str|None, "execution_id": str }`

- `verification_results_updated`
  - `data`: `{ "results": list[VerificationResult], "iteration": int, "warning": str|None, "execution_id": str }`

- `verification_state_changed`
  - `data`: `{ "scope": "worker"|"retry", "state": str, "verification_status": dict, "timestamp": float, "execution_id": str }`

### Retry Events

- `retry_triggered`
  - `data`: `{ "iteration": int, "retry_prompt": str, "agents": list[str], "execution_id": str }`

- `retry_completed`
  - `data`: `{ "iteration": int, "results_count": int, "agents": list[str], "execution_id": str }`

- `retry_skipped`
  - `data`: `{ "reason": str, "execution_id": str }`

### Other

- `error`
  - `data`: `{ ... , "execution_id": str }` (fields depend on the failure source)

## State Management Recommendations

Downstream systems should build a state machine and maintain:
- Execution-level: `execution_id` across all events
- Planning/Routing: `workflow_selected` (task_assignments, parallel strategy, total_instances)
- Agent-level: `agent_start`/`agent_end`/`parallel_*`
- Instance-level: `instance_*`/`parallel_instances_*`/`instance_result`
- Verification: `worker_verification_*`/`verification_results_updated`
- Retry: `retry_*`
- Final: `final_result` (citations, partial stats)
- Token: `token` (for rendering or statistics)

Typical fields:
- `execution_id` for correlation
- `agent_id` or `instance_id` for mapping
- `iteration` for verification/parallel loops
- `timestamp` for ordering and performance insights

Recommendations:
- Use `token` for incremental render; rely on `final_result.result` for final output
- Drive UI state with verification/retry events
- Surface `verification_results_updated.warning` when present

## Behavior with console=false

- Returns an event stream (iterator) without Rich console UI
- Events are the same as console mode; only UI rendering differs
- Must consume all events, especially:
  - `workflow_selected` (initial structure)
  - `agent_*`/`parallel_*`/`instance_*` (execution)
  - `worker_verification_*`/`verification_results_updated` (verification)
  - `retry_*` (retry flow)
  - `final_result` (completion)

## Notes

- Do not print the stream object itself inside the loop; print the `event` or structured data
- `_KivaRunStream.result()` returns final result after iteration
- `__str__` does not consume the stream; prefer explicit `event` and `result()`

## Code References

- Entry and event stream construction: `src/kiva/run.py`
- Workflows and event emission points:
  - Router: `src/kiva/workflows/router.py`
  - Supervisor: `src/kiva/workflows/supervisor.py`
  - Parliament: `src/kiva/workflows/parliament.py`
  - Executor and utilities: `src/kiva/workflows/executor.py`, `src/kiva/workflows/utils.py`
