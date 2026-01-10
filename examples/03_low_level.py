"""Low-Level API Example - Using the run() generator.

Full control over the event stream for custom UIs or system integration.
"""

import asyncio

from kiva import ChatOpenAI, create_agent, run, tool

API_BASE = ""
API_KEY = ""
MODEL = ""


def create_model() -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=0.7,
    )


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: Sunny, 25°C"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def create_agents():
    """Create the worker agents."""
    weather_agent = create_agent(model=create_model(), tools=[get_weather])
    weather_agent.name = "weather_agent"
    weather_agent.description = "Gets weather information"

    calc_agent = create_agent(model=create_model(), tools=[calculate])
    calc_agent.name = "calculator_agent"
    calc_agent.description = "Performs calculations"

    return [weather_agent, calc_agent]


async def main():
    """Run orchestration with custom event handling."""
    agents = create_agents()

    print("=" * 50)
    print("Kiva Low-Level API Demo")
    print("=" * 50)

    async for event in run(
        prompt="What's the weather in Beijing? Calculate 100 / 4",
        agents=agents,
        base_url=API_BASE,
        api_key=API_KEY,
        model_name=MODEL,
    ):
        # Handle events based on type
        match event.type:
            case "token":
                # Streaming LLM output
                print(event.data["content"], end="", flush=True)

            case "workflow_selected":
                print(f"\n\n[Workflow] {event.data['workflow'].upper()}")
                print(f"[Complexity] {event.data.get('complexity', 'N/A')}")
                tasks = event.data.get("task_assignments", [])
                for task in tasks:
                    print(f"  - {task.get('agent_id')}: {task.get('task', '')[:50]}...")

            case "parallel_start":
                agent_ids = event.data.get("agent_ids", [])
                print(f"\n[Parallel Start] Agents: {', '.join(agent_ids)}")

            case "agent_start":
                agent_id = event.data.get("agent_id", event.agent_id or "unknown")
                print(f"\n[Agent Start] {agent_id}")

            case "agent_end":
                result_data = event.data.get("result", {})
                if isinstance(result_data, dict):
                    agent_id = result_data.get("agent_id", "unknown")
                    result = result_data.get("result", "")
                else:
                    agent_id = event.agent_id or "unknown"
                    result = str(result_data)
                print(f"[Agent End] {agent_id}: {result[:100]}...")

            case "parallel_complete":
                print("\n[Parallel Complete]")

            case "final_result":
                print("\n" + "=" * 50)
                print("[Final Result]")
                print(event.data.get("result", ""))
                print("=" * 50)

            case "error":
                print(f"\n[Error] {event.data.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
