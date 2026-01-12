# End-to-End (E2E) Tests

This directory contains comprehensive end-to-end tests for the Kiva SDK using a real LLM API endpoint.

## Test Configuration

All tests use environment variables for API configuration:

```bash
export KIVA_API_BASE="http://your-api-endpoint/v1"
export KIVA_API_KEY="your-api-key"
export KIVA_MODEL="your-model-name"
```

Default values (if not set):
- `KIVA_API_BASE`: `http://localhost:8000/v1`
- `KIVA_API_KEY`: `` (empty)
- `KIVA_MODEL`: `gpt-4o`

## Test Files

### Core Workflow Tests

#### `test_e2e_router_workflow.py`
Tests the router workflow for simple single-agent tasks.

**Scenarios:**
- Simple weather query
- Simple calculation
- Workflow override
- Search query
- Event sequence ordering

**Run:** `uv run --dev pytest tests/e2e/test_e2e_router_workflow.py -v`

#### `test_e2e_supervisor_workflow.py`
Tests the supervisor workflow for parallel multi-agent tasks.

**Scenarios:**
- Multi-agent task execution
- Workflow override
- Three agents coordination
- Parallel event emission
- Max parallel agents limit

**Run:** `uv run --dev pytest tests/e2e/test_e2e_supervisor_workflow.py -v`

#### `test_e2e_parliament_workflow.py`
Tests the parliament workflow for iterative conflict resolution.

**Scenarios:**
- Basic execution
- Max iterations respected
- Three agents deliberation
- Conflict detection
- Single iteration (no conflict)

**Run:** `uv run --dev pytest tests/e2e/test_e2e_parliament_workflow.py -v`

### API Level Tests

#### `test_e2e_high_level_api.py`
Tests the high-level Kiva client API.

**Scenarios:**
- Single function agent
- Class agent with multiple tools
- Multiple agents
- Add agent method
- Method chaining
- Async run
- Temperature setting
- Empty response handling

**Run:** `uv run --dev pytest tests/e2e/test_e2e_high_level_api.py -v`

#### `test_e2e_low_level_api.py`
Tests the low-level run() async generator API.

**Scenarios:**
- Returns async iterator
- Yields StreamEvent objects
- Event types
- Token streaming
- Workflow selected event
- Final result event
- Execution ID consistency
- Multiple agents
- Event timestamps
- Agent events

**Run:** `uv run --dev pytest tests/e2e/test_e2e_low_level_api.py -v`

#### `test_e2e_agent_router.py`
Tests the AgentRouter for modular organization.

**Scenarios:**
- Basic router usage
- Multiple agents in router
- Multiple routers
- Nested routers
- Class agent in router
- Prefix override
- Mixed direct and router agents
- Method chaining

**Run:** `uv run --dev pytest tests/e2e/test_e2e_agent_router.py -v`

### Advanced Features

#### `test_e2e_parallel_instances.py`
Tests parallel agent instance spawning.

**Scenarios:**
- Basic parallel instances
- Instance events
- With supervisor workflow
- Max limit enforcement
- Fan-out strategy
- Result aggregation
- Isolated context
- Error handling

**Run:** `uv run --dev pytest tests/e2e/test_e2e_parallel_instances.py -v`

#### `test_e2e_console_output.py`
Tests the rich console visualization.

**Scenarios:**
- Basic execution
- Multi-agent display
- Custom refresh rate
- Parallel instances display
- Three agents display
- Long result handling
- Special characters

**Run:** `uv run --dev pytest tests/e2e/test_e2e_console_output.py -v`

### Error Handling & Edge Cases

#### `test_e2e_error_handling.py`
Tests error handling and boundary conditions.

**Scenarios:**
- Empty agents list
- Invalid agent type
- Agent execution errors
- Empty prompt
- Very long prompt
- Special characters
- No agents registered
- Invalid workflow override
- Zero max iterations
- Negative max parallel agents
- Partial agent failure
- Unicode agent names

**Run:** `uv run --dev pytest tests/e2e/test_e2e_error_handling.py -v`

### Consistency & Validation

#### `test_e2e_consistency.py`
Tests API consistency across different interfaces.

**Scenarios:**
- run() and run_with_console() consistency
- Sync and async consistency
- Event structure consistency
- Execution ID format
- Final result structure
- Agent event consistency
- Decorator vs add_agent consistency

**Run:** `uv run --dev pytest tests/e2e/test_e2e_consistency.py -v`

#### `test_e2e_output_validation.py`
Validates output patterns match documentation.

**Scenarios:**
- Router workflow pattern
- Supervisor workflow pattern
- Parliament workflow pattern
- Parallel instances pattern
- Event structure consistency
- Execution ID consistency
- Timestamp ordering
- Final result presence
- Workflow selected ordering

**Run:** `uv run --dev pytest tests/e2e/test_e2e_output_validation.py -v`

### Documentation & Capture

#### `test_e2e_output_capture.py`
Captures real execution outputs for documentation.

**Scenarios:**
- Router workflow output
- Supervisor workflow output
- Parliament workflow output
- Parallel instances output
- Error handling output
- High-level API output

**Run:** `uv run --dev pytest tests/e2e/test_e2e_output_capture.py -v -s`

**Output:** Saves JSON files to `docs/outputs/`

## Running Tests

### Run All E2E Tests

```bash
uv run --dev pytest tests/e2e/ -v
```

### Run Specific Test File

```bash
uv run --dev pytest tests/e2e/test_e2e_router_workflow.py -v
```

### Run Specific Test

```bash
uv run --dev pytest tests/e2e/test_e2e_router_workflow.py::TestRouterWorkflowE2E::test_router_simple_weather_query -v
```

### Run with Output Capture

```bash
uv run --dev pytest tests/e2e/test_e2e_output_capture.py -v -s
```

### Run with Coverage

```bash
uv run --dev pytest tests/e2e/ --cov=src/kiva --cov-report=html
```

## Test Statistics

- **Total Test Files:** 10
- **Total Test Cases:** ~80+
- **Coverage Areas:**
  - 3 Workflow types (Router, Supervisor, Parliament)
  - 3 API levels (High, Mid, Low)
  - Parallel instances
  - Error handling
  - Console output
  - API consistency
  - Output validation

## Test Fixtures

All tests use shared fixtures from `conftest.py`:

- `api_config`: API configuration
- `create_model`: Model factory
- `weather_tool`, `calculate_tool`, `search_tool`, `translate_tool`: Common tools
- `create_weather_agent`, `create_calculator_agent`, etc.: Agent factories

## Expected Behavior

### Event Sequence

All workflows follow this general pattern:

1. `workflow_selected` - First event (after tokens)
2. Agent execution events (`agent_start`, `agent_end`, etc.)
3. `final_result` - Last event

### Event Types

Common event types across all workflows:

- `token`: Streaming LLM tokens
- `workflow_selected`: Workflow determination
- `agent_start`: Agent begins execution
- `agent_end`: Agent completes execution
- `final_result`: Final synthesized result
- `error`: Error occurred

Additional event types for specific workflows:

- **Supervisor:** `parallel_start`, `parallel_complete`
- **Parliament:** `parallel_start` (with iteration), `parallel_complete` (with conflicts)
- **Parallel Instances:** `instance_spawn`, `instance_start`, `instance_end`, `instance_complete`, `parallel_instances_start`, `parallel_instances_complete`

## Troubleshooting

### Tests Timeout

If tests timeout, check:
- API endpoint is accessible
- API key is valid
- Model is available
- Network connection is stable

### Tests Fail

If tests fail, check:
- API configuration in `conftest.py`
- Model responses may vary slightly
- Adjust assertions if needed for model-specific behavior

### Slow Tests

E2E tests are slower than unit tests because they:
- Make real API calls
- Wait for LLM responses
- Process streaming tokens

Expected runtime: 2-5 seconds per test

## Documentation

Test outputs are documented in:

- `docs/execution-outputs.md`: Detailed output patterns
- `docs/outputs/*.json`: Captured execution data

## Contributing

When adding new tests:

1. Follow the existing naming convention: `test_e2e_<feature>_<scenario>`
2. Use descriptive docstrings
3. Add assertions for key behaviors
4. Print meaningful output for debugging
5. Update this README with new test scenarios

## Notes

- Tests use real LLM API, so responses may vary slightly
- Tests are designed to be resilient to minor variations
- Focus on structural validation rather than exact content matching
- All tests should pass consistently with the configured model
