# Execution Output Patterns

This document shows real execution outputs from the Kiva SDK across different scenarios. These examples help developers understand what to expect when using the SDK.

## Table of Contents

1. [Router Workflow](#router-workflow)
2. [Supervisor Workflow](#supervisor-workflow)
3. [Parliament Workflow](#parliament-workflow)
4. [Parallel Instances](#parallel-instances)
5. [Error Handling](#error-handling)
6. [High-Level API](#high-level-api)

---

## Router Workflow

**Scenario**: Simple single-agent task  
**Prompt**: "What's the weather in Beijing?"  
**Agents**: 1 (weather_agent)  
**Expected Workflow**: router

### Event Sequence

```
[WORKFLOW_SELECTED]
  Workflow: router
  Complexity: simple
  Task Assignments: 1

[AGENT_START]
  Agent: weather_agent
  Task: Get the current weather in Beijing

[AGENT_END]
  Agent: weather_agent
  Result: The current weather in Beijing is sunny with a temperature of 25°C and a humidity of 45%

[FINAL_RESULT]
  Result: The current weather in Beijing is sunny with a temperature of 25°C and a humidity of 45%
```

### Event Types

- `workflow_selected`: Workflow determination complete
- `token`: Streaming LLM tokens (multiple events)
- `agent_start`: Agent begins execution
- `agent_end`: Agent completes execution
- `final_result`: Final synthesized result

### Total Events

Approximately 110-120 events (including token streaming)

### Code Example

```python
from kiva import run
import os

# Get configuration from environment
api_base = os.getenv("KIVA_API_BASE")
api_key = os.getenv("KIVA_API_KEY")
model = os.getenv("KIVA_MODEL")

async for event in run(
    prompt="What's the weather in Beijing?",
    agents=[weather_agent],
    model_name=model,
    api_key=api_key,
    base_url=api_base,
):
    if event.type == "final_result":
        print(event.data["result"])
```

---

## Supervisor Workflow

**Scenario**: Parallel multi-agent task  
**Prompt**: "What's the weather in Tokyo? Also calculate 25 * 4"  
**Agents**: 2 (weather_agent, calculator_agent)  
**Expected Workflow**: supervisor

### Event Sequence

```
[WORKFLOW_SELECTED]
  Workflow: supervisor
  Parallel Strategy: none

[PARALLEL_START]
  Agents: ['weather_agent', 'calculator_agent']

[AGENT_START]
  Agent: weather_agent

[AGENT_START]
  Agent: calculator_agent

[AGENT_END]
  Agent: calculator_agent
  Result: The result of 25 * 4 is **100**

[AGENT_END]
  Agent: weather_agent
  Result: The current weather in Tokyo is cloudy with a temperature of 22°C and a humidity level of 60%

[PARALLEL_COMPLETE]

[FINAL_RESULT]
  Result: The current weather in Tokyo is cloudy with a temperature of 22°C and a humidity level of 60% [weather_agent].
  The result of 25 * 4 is **100** [calculator_agent].
```

### Event Types

- `workflow_selected`: Workflow determination
- `parallel_start`: Parallel execution begins
- `agent_start`: Each agent starts (multiple)
- `agent_end`: Each agent completes (multiple)
- `parallel_complete`: All agents finished
- `token`: Streaming tokens
- `final_result`: Synthesized result with citations

### Total Events

Approximately 200-220 events

### Key Features

- **Parallel Execution**: Multiple agents run concurrently
- **Citations**: Final result includes source attribution `[agent_id]`
- **Result Synthesis**: Lead agent combines outputs from all agents

### Code Example

```python
from kiva import run
import os

api_base = os.getenv("KIVA_API_BASE")
api_key = os.getenv("KIVA_API_KEY")
model = os.getenv("KIVA_MODEL")

async for event in run(
    prompt="What's the weather in Tokyo? Also calculate 25 * 4",
    agents=[weather_agent, calculator_agent],
    workflow_override="supervisor",
    model_name=model,
    api_key=api_key,
    base_url=api_base,
):
    if event.type == "parallel_start":
        print(f"Starting parallel execution: {event.data['agent_ids']}")
    elif event.type == "final_result":
        print(event.data["result"])
```

---

## Parliament Workflow

**Scenario**: Iterative conflict resolution  
**Prompt**: "Should I go outside today? Check weather and give advice"  
**Agents**: 2 (weather_agent, search_agent)  
**Expected Workflow**: parliament  
**Max Iterations**: 3

### Event Sequence

```
[WORKFLOW_SELECTED]
  Workflow: parliament

[PARALLEL_START]
  Iteration: 0
  Agents: ['weather_agent']

[PARALLEL_COMPLETE]
  Conflicts Found: 0

[FINAL_RESULT]
  Result: I can help with that! However, I need to know your current location (city or region) 
  to provide accurate weather information. Could you please tell me where you are located?
```

### Event Types

- `workflow_selected`: Workflow determination
- `parallel_start`: Iteration begins (includes iteration number)
- `parallel_complete`: Iteration ends (includes conflict count)
- `token`: Streaming tokens
- `final_result`: Final result after conflict resolution

### Total Events

Approximately 140-160 events per iteration

### Key Features

- **Iterative Execution**: Multiple rounds if conflicts detected
- **Conflict Detection**: Identifies contradictions between agents
- **Max Iterations**: Stops after reaching limit
- **Conflict Resolution**: Agents reconsider responses in subsequent iterations

### Code Example

```python
from kiva import run
import os

api_base = os.getenv("KIVA_API_BASE")
api_key = os.getenv("KIVA_API_KEY")
model = os.getenv("KIVA_MODEL")

async for event in run(
    prompt="Should I go outside today? Check weather and give advice",
    agents=[weather_agent, search_agent],
    workflow_override="parliament",
    max_iterations=3,
    model_name=model,
    api_key=api_key,
    base_url=api_base,
):
    if event.type == "parallel_start":
        print(f"Iteration {event.data.get('iteration', 0)}")
    elif event.type == "parallel_complete":
        print(f"Conflicts: {event.data.get('conflicts_found', 0)}")
```

---

## Parallel Instances

**Scenario**: Multiple instances of same agent  
**Prompt**: "Get weather for Beijing, Tokyo, and London"  
**Agents**: 1 (weather_agent)  
**Expected**: Multiple instances spawned

### Event Sequence

```
[WORKFLOW_SELECTED]
  Workflow: router
  Parallel Strategy: fan_out
  Total Instances: 3

[INSTANCE_SPAWN]
  Instance: exec-abc-weather_agent-i0-xyz
  Agent: weather_agent
  Task: Get weather for Beijing

[INSTANCE_START]
  Instance: exec-abc-weather_agent-i0-xyz

[INSTANCE_END]
  Instance: exec-abc-weather_agent-i0-xyz
  Result: Beijing: Sunny, 25°C, humidity 45%

[INSTANCE_SPAWN]
  Instance: exec-abc-weather_agent-i1-xyz
  Agent: weather_agent
  Task: Get weather for Tokyo

[INSTANCE_START]
  Instance: exec-abc-weather_agent-i1-xyz

[INSTANCE_END]
  Instance: exec-abc-weather_agent-i1-xyz
  Result: Tokyo: Cloudy, 22°C, humidity 60%

[INSTANCE_SPAWN]
  Instance: exec-abc-weather_agent-i2-xyz
  Agent: weather_agent
  Task: Get weather for London

[INSTANCE_START]
  Instance: exec-abc-weather_agent-i2-xyz

[INSTANCE_END]
  Instance: exec-abc-weather_agent-i2-xyz
  Result: London: Rainy, 15°C, humidity 80%

[PARALLEL_INSTANCES_START]
  Instance Count: 3

[PARALLEL_INSTANCES_COMPLETE]

[FINAL_RESULT]
  Result: Here's the weather for the three cities:
  - Beijing: Sunny, 25°C, humidity 45%
  - Tokyo: Cloudy, 22°C, humidity 60%
  - London: Rainy, 15°C, humidity 80%
```

### Event Types

- `workflow_selected`: Includes parallel_strategy and total_instances
- `instance_spawn`: New instance created
- `instance_start`: Instance begins execution
- `instance_end`: Instance completes
- `instance_complete`: Instance finished (success/error)
- `instance_result`: Result from instance
- `parallel_instances_start`: Batch start
- `parallel_instances_complete`: Batch complete
- `final_result`: Aggregated results

### Total Events

Approximately 200-220 events (varies with instance count)

### Key Features

- **Instance Isolation**: Each instance has unique ID and context
- **Parallel Execution**: Instances run concurrently
- **Result Aggregation**: Lead agent combines all instance results
- **Scalability**: Controlled by `max_parallel_agents` parameter

### Code Example

```python
from kiva import run
import os

api_base = os.getenv("KIVA_API_BASE")
api_key = os.getenv("KIVA_API_KEY")
model = os.getenv("KIVA_MODEL")

async for event in run(
    prompt="Get weather for Beijing, Tokyo, and London",
    agents=[weather_agent],
    max_parallel_agents=5,
    model_name=model,
    api_key=api_key,
    base_url=api_base,
):
    if event.type == "instance_spawn":
        print(f"Spawned: {event.data['instance_id']}")
    elif event.type == "instance_end":
        print(f"Completed: {event.data['result']}")
```

---

## Error Handling

**Scenario**: Empty agents list  
**Prompt**: "Test prompt"  
**Agents**: 0 (empty list)  
**Expected**: ConfigurationError

### Error Output

```
[ERROR RAISED]
  Type: ConfigurationError
  Message: agents list cannot be empty
```

### Common Errors

#### 1. Empty Agents List

```python
# ❌ This will raise ConfigurationError
async for event in run(prompt="test", agents=[]):
    pass

# Error: agents list cannot be empty
```

#### 2. Invalid Agent Type

```python
# ❌ Agent without ainvoke method
class InvalidAgent:
    def invoke(self, data):
        return data

async for event in run(prompt="test", agents=[InvalidAgent()]):
    pass

# Error: Agent at index 0 must have ainvoke method. Please use create_agent() to create agents.
```

#### 3. Agent Execution Failure

When an agent fails during execution, the SDK handles it gracefully:

```
[AGENT_END]
  Agent: failing_agent
  Error: Agent execution failed: Tool execution error
  Recovery Suggestion: 
    1. Checking the tool's implementation for errors
    2. Verifying the tool's input parameters
    3. Adding error handling to the tool
```

### Error Handling Best Practices

1. **Always validate agents**: Use `create_agent()` from LangChain
2. **Handle partial failures**: Check `partial_result` flag in final_result
3. **Monitor error events**: Listen for `error` event type
4. **Implement recovery**: Use recovery suggestions from AgentError

---

## High-Level API

**Scenario**: Using Kiva client  
**Prompt**: "What's the weather in Paris?"  
**API**: `Kiva.run_async()`

### Output

```
[FINAL RESULT]
  Result: The current weather in Paris is sunny with a temperature of 25°C.
```

### Code Example

```python
from kiva import Kiva
import os

api_base = os.getenv("KIVA_API_BASE")
api_key = os.getenv("KIVA_API_KEY")
model = os.getenv("KIVA_MODEL")

kiva = Kiva(
    base_url=api_base,
    api_key=api_key,
    model=model,
)

@kiva.agent("weather", "Gets weather information")
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"{city}: Sunny, 25°C"

# Async usage
result = await kiva.run_async("What's the weather in Paris?", console=False)
print(result)

# Sync usage (not in async context)
result = kiva.run("What's the weather in Paris?", console=False)
print(result)
```

### Features

- **Simplified API**: No need to manage events manually
- **Decorator Pattern**: Easy agent registration
- **Console Output**: Optional rich terminal visualization
- **Sync/Async**: Both modes supported

---

## Event Structure Reference

All events follow this structure:

```python
{
    "type": str,           # Event type identifier
    "data": dict,          # Event-specific data
    "timestamp": float,    # Unix timestamp
    "agent_id": str | None # Associated agent (if applicable)
}
```

### Common Data Fields

- `execution_id`: Unique identifier for the execution
- `workflow`: Selected workflow type
- `complexity`: Task complexity assessment
- `result`: Agent or final result
- `error`: Error message (if failed)
- `agent_id`: Agent identifier
- `instance_id`: Instance identifier (for parallel instances)
- `task`: Task description
- `citations`: Source attributions

---

## Testing Output Patterns

To verify output patterns in your tests:

```python
import pytest
from kiva import run

@pytest.mark.asyncio
async def test_output_pattern():
    """Test that output follows expected pattern."""
    events = []
    
    async for event in run(
        prompt="Test prompt",
        agents=[test_agent],
        model_name="...",
        api_key="...",
        base_url="...",
    ):
        events.append(event)
    
    # Verify event sequence
    event_types = [e.type for e in events]
    assert "workflow_selected" in event_types
    assert "final_result" in event_types
    
    # Verify workflow_selected comes before final_result
    workflow_idx = event_types.index("workflow_selected")
    final_idx = event_types.index("final_result")
    assert workflow_idx < final_idx
    
    # Verify final result structure
    final_event = next(e for e in events if e.type == "final_result")
    assert "result" in final_event.data
    assert "execution_id" in final_event.data
```

---

## Captured Output Files

Detailed JSON outputs for each scenario are available in `docs/outputs/`:

- `router_workflow.json`: Router workflow execution
- `supervisor_workflow.json`: Supervisor workflow execution
- `parliament_workflow.json`: Parliament workflow execution
- `parallel_instances.json`: Parallel instance execution

These files contain complete event sequences with timestamps and full data payloads.

---

## Notes

- Event counts may vary slightly based on LLM response length
- Token streaming events are numerous but can be filtered
- Timestamps are Unix timestamps (seconds since epoch)
- Configure API endpoint via environment variables
- See `docs/e2e-testing-guide.md` for setup instructions
