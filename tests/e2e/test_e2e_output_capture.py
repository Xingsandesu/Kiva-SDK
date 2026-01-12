"""E2E test for capturing and documenting output patterns.

This test captures real execution outputs for documentation purposes.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from kiva import run, run_with_console, Kiva


class TestOutputCaptureE2E:
    """E2E tests that capture outputs for documentation."""

    @pytest.mark.asyncio
    async def test_capture_router_workflow_output(
        self, api_config, create_weather_agent
    ):
        """Capture router workflow output pattern."""
        agents = [create_weather_agent()]
        events = []
        
        print("\n" + "="*80)
        print("ROUTER WORKFLOW - Simple Single Agent Task")
        print("="*80)
        print(f"Prompt: What's the weather in Beijing?")
        print(f"Agents: 1 (weather_agent)")
        print(f"Expected Workflow: router")
        print("-"*80)

        async for event in run(
            prompt="What's the weather in Beijing?",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)
            # Print key events
            if event.type in ["workflow_selected", "agent_start", "agent_end", "final_result"]:
                print(f"\n[{event.type.upper()}]")
                if event.type == "workflow_selected":
                    print(f"  Workflow: {event.data.get('workflow')}")
                    print(f"  Complexity: {event.data.get('complexity')}")
                    print(f"  Task Assignments: {len(event.data.get('task_assignments', []))}")
                elif event.type == "agent_start":
                    print(f"  Agent: {event.data.get('agent_id', event.agent_id)}")
                    print(f"  Task: {event.data.get('task', '')[:60]}...")
                elif event.type == "agent_end":
                    result_data = event.data.get('result', {})
                    if isinstance(result_data, dict):
                        result = result_data.get('result', '')
                    else:
                        result = str(result_data)
                    print(f"  Agent: {event.data.get('agent_id', event.agent_id)}")
                    print(f"  Result: {result[:100]}...")
                elif event.type == "final_result":
                    print(f"  Result: {event.data.get('result', '')[:200]}...")

        print("\n" + "="*80)
        print(f"Total Events: {len(events)}")
        print(f"Event Types: {set(e.type for e in events)}")
        print("="*80 + "\n")

        # Save to file
        self._save_output("router_workflow", events)

    @pytest.mark.asyncio
    async def test_capture_supervisor_workflow_output(
        self, api_config, create_weather_agent, create_calculator_agent
    ):
        """Capture supervisor workflow output pattern."""
        agents = [create_weather_agent(), create_calculator_agent()]
        events = []
        
        print("\n" + "="*80)
        print("SUPERVISOR WORKFLOW - Parallel Multi-Agent Task")
        print("="*80)
        print(f"Prompt: What's the weather in Tokyo? Also calculate 25 * 4")
        print(f"Agents: 2 (weather_agent, calculator_agent)")
        print(f"Expected Workflow: supervisor")
        print("-"*80)

        async for event in run(
            prompt="What's the weather in Tokyo? Also calculate 25 * 4",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="supervisor",
        ):
            events.append(event)
            if event.type in ["workflow_selected", "parallel_start", "agent_start", "agent_end", "parallel_complete", "final_result"]:
                print(f"\n[{event.type.upper()}]")
                if event.type == "workflow_selected":
                    print(f"  Workflow: {event.data.get('workflow')}")
                    print(f"  Parallel Strategy: {event.data.get('parallel_strategy', 'none')}")
                elif event.type == "parallel_start":
                    print(f"  Agents: {event.data.get('agent_ids', [])}")
                elif event.type == "agent_start":
                    print(f"  Agent: {event.data.get('agent_id', event.agent_id)}")
                elif event.type == "agent_end":
                    result_data = event.data.get('result', {})
                    if isinstance(result_data, dict):
                        agent_id = result_data.get('agent_id', 'unknown')
                        result = result_data.get('result', '')
                    else:
                        agent_id = event.agent_id or 'unknown'
                        result = str(result_data)
                    print(f"  Agent: {agent_id}")
                    print(f"  Result: {result[:80]}...")
                elif event.type == "final_result":
                    print(f"  Result: {event.data.get('result', '')[:200]}...")

        print("\n" + "="*80)
        print(f"Total Events: {len(events)}")
        print("="*80 + "\n")

        self._save_output("supervisor_workflow", events)

    @pytest.mark.asyncio
    async def test_capture_parliament_workflow_output(
        self, api_config, create_weather_agent, create_search_agent
    ):
        """Capture parliament workflow output pattern."""
        agents = [create_weather_agent(), create_search_agent()]
        events = []
        
        print("\n" + "="*80)
        print("PARLIAMENT WORKFLOW - Iterative Conflict Resolution")
        print("="*80)
        print(f"Prompt: Should I go outside today? Check weather and give advice")
        print(f"Agents: 2 (weather_agent, search_agent)")
        print(f"Expected Workflow: parliament")
        print(f"Max Iterations: 3")
        print("-"*80)

        async for event in run(
            prompt="Should I go outside today? Check weather and give advice",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            workflow_override="parliament",
            max_iterations=3,
        ):
            events.append(event)
            if event.type in ["workflow_selected", "parallel_start", "parallel_complete", "final_result"]:
                print(f"\n[{event.type.upper()}]")
                if event.type == "workflow_selected":
                    print(f"  Workflow: {event.data.get('workflow')}")
                elif event.type == "parallel_start":
                    iteration = event.data.get('iteration', 0)
                    print(f"  Iteration: {iteration}")
                    print(f"  Agents: {event.data.get('agent_ids', [])}")
                elif event.type == "parallel_complete":
                    conflicts = event.data.get('conflicts_found', 0)
                    print(f"  Conflicts Found: {conflicts}")
                elif event.type == "final_result":
                    print(f"  Result: {event.data.get('result', '')[:200]}...")

        print("\n" + "="*80)
        print(f"Total Events: {len(events)}")
        print("="*80 + "\n")

        self._save_output("parliament_workflow", events)

    @pytest.mark.asyncio
    async def test_capture_parallel_instances_output(
        self, api_config, create_weather_agent
    ):
        """Capture parallel instances output pattern."""
        agents = [create_weather_agent()]
        events = []
        
        print("\n" + "="*80)
        print("PARALLEL INSTANCES - Multiple Instance Spawning")
        print("="*80)
        print(f"Prompt: Get weather for Beijing, Tokyo, and London")
        print(f"Agents: 1 (weather_agent)")
        print(f"Expected: Multiple instances of same agent")
        print("-"*80)

        async for event in run(
            prompt="Get weather for Beijing, Tokyo, and London",
            agents=agents,
            model_name=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
        ):
            events.append(event)
            if event.type in ["workflow_selected", "instance_spawn", "instance_start", "instance_end", "parallel_instances_start", "parallel_instances_complete", "final_result"]:
                print(f"\n[{event.type.upper()}]")
                if event.type == "workflow_selected":
                    print(f"  Workflow: {event.data.get('workflow')}")
                    print(f"  Parallel Strategy: {event.data.get('parallel_strategy', 'none')}")
                    print(f"  Total Instances: {event.data.get('total_instances', 1)}")
                elif event.type == "instance_spawn":
                    print(f"  Instance: {event.data.get('instance_id', '')[:30]}...")
                    print(f"  Agent: {event.data.get('agent_id')}")
                    print(f"  Task: {event.data.get('task', '')[:50]}...")
                elif event.type == "instance_start":
                    print(f"  Instance: {event.data.get('instance_id', '')[:30]}...")
                elif event.type == "instance_end":
                    print(f"  Instance: {event.data.get('instance_id', '')[:30]}...")
                    print(f"  Result: {event.data.get('result', '')[:60]}...")
                elif event.type == "parallel_instances_start":
                    print(f"  Instance Count: {event.data.get('instance_count')}")
                elif event.type == "final_result":
                    print(f"  Result: {event.data.get('result', '')[:200]}...")

        print("\n" + "="*80)
        print(f"Total Events: {len(events)}")
        print("="*80 + "\n")

        self._save_output("parallel_instances", events)

    @pytest.mark.asyncio
    async def test_capture_error_handling_output(
        self, api_config, create_weather_agent
    ):
        """Capture error handling output pattern."""
        events = []
        
        print("\n" + "="*80)
        print("ERROR HANDLING - Empty Agents List")
        print("="*80)
        print(f"Prompt: Test prompt")
        print(f"Agents: 0 (empty list)")
        print(f"Expected: ConfigurationError")
        print("-"*80)

        try:
            async for event in run(
                prompt="Test prompt",
                agents=[],
                model_name=api_config["model"],
                api_key=api_config["api_key"],
                base_url=api_config["base_url"],
            ):
                events.append(event)
        except Exception as e:
            print(f"\n[ERROR RAISED]")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")

        print("\n" + "="*80 + "\n")

    @pytest.mark.asyncio
    async def test_capture_high_level_api_output(self, api_config):
        """Capture high-level API (Kiva) output pattern."""
        print("\n" + "="*80)
        print("HIGH-LEVEL API - Kiva Client")
        print("="*80)
        print(f"Prompt: What's the weather in Paris?")
        print(f"API: Kiva.run_async()")
        print("-"*80)

        kiva = Kiva(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"],
            model=api_config["model"],
        )

        @kiva.agent("weather", "Gets weather information")
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"{city}: Sunny, 25°C"

        result = await kiva.run_async("What's the weather in Paris?", console=False)
        
        print(f"\n[FINAL RESULT]")
        print(f"  Result: {result}")
        print("\n" + "="*80 + "\n")

    def _save_output(self, scenario_name: str, events: list):
        """Save captured events to JSON file."""
        output_dir = Path("docs/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{scenario_name}.json"
        
        # Convert events to serializable format
        events_data = []
        for event in events:
            event_dict = {
                "type": event.type,
                "timestamp": event.timestamp,
                "agent_id": event.agent_id,
                "data": {}
            }
            
            # Safely extract data
            for key, value in event.data.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    event_dict["data"][key] = value
                elif isinstance(value, list):
                    event_dict["data"][key] = [
                        str(v) if not isinstance(v, (str, int, float, bool, type(None), dict)) else v
                        for v in value
                    ]
                elif isinstance(value, dict):
                    event_dict["data"][key] = value
                else:
                    event_dict["data"][key] = str(value)
            
            events_data.append(event_dict)
        
        output_data = {
            "scenario": scenario_name,
            "timestamp": datetime.now().isoformat(),
            "total_events": len(events),
            "event_types": list(set(e.type for e in events)),
            "events": events_data
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Output saved to: {output_file}")
