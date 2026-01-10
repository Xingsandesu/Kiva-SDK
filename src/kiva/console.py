"""Rich console output for Kiva orchestration.

This module provides beautiful terminal visualization for orchestration
execution using the Rich library. It displays real-time progress including
workflow phases, agent status, and final results.

Requires the 'rich' package: pip install kiva-sdk[console]
"""

import json
import re

from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from kiva.run import run


class KivaLiveRenderer:
    """Dynamic live renderer for Kiva orchestration visualization.

    Manages the visual state and rendering of orchestration progress,
    including phase indicators, agent status tables, and result panels.
    """

    AGENT_COLORS = ["cyan", "magenta", "yellow", "green", "blue", "red"]

    def __init__(self, prompt: str):
        """Initialize the renderer with the user prompt."""
        self.prompt = prompt
        self.token_buffer = ""
        self.workflow_info: dict = {}
        self.task_assignments: list = []
        self.agent_states: dict[str, dict] = {}
        self.final_result: str | None = None
        self.citations: list = []
        self.phase = "initializing"
        self.color_index = 0

    def _get_agent_color(self, agent_id: str) -> str:
        """Get the assigned color for an agent."""
        if agent_id not in self.agent_states:
            return "white"
        return self.agent_states[agent_id].get("color", "white")

    def _assign_agent_color(self, agent_id: str) -> str:
        """Assign a color to a new agent."""
        color = self.AGENT_COLORS[self.color_index % len(self.AGENT_COLORS)]
        self.color_index += 1
        return color

    def _try_parse_json(self, text: str) -> dict | None:
        """Attempt to parse JSON from text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _render_header(self) -> Panel:
        """Render the header panel with the prompt."""
        return Panel(
            Text(self.prompt, style="bold white"),
            title="[bold cyan]KIVA Orchestrator[/]",
            subtitle="[dim]Multi-Agent Workflow Engine[/]",
            border_style="cyan",
            box=DOUBLE,
        )

    def _render_phase_indicator(self) -> Text:
        """Render the phase progress indicator."""
        phases = ["initializing", "analyzing", "executing", "synthesizing", "complete"]
        phase_icons = {
            "initializing": ("[ ]", "dim"),
            "analyzing": ("[~]", "yellow"),
            "executing": ("[>]", "green"),
            "synthesizing": ("[*]", "magenta"),
            "complete": ("[v]", "bold green"),
        }
        text = Text()
        for i, p in enumerate(phases):
            icon, style = phase_icons[p]
            if p == self.phase:
                text.append(f" {icon} {p.upper()} ", style=f"bold reverse {style}")
            else:
                text.append(f" {icon} {p} ", style="dim")
            if i < len(phases) - 1:
                text.append("->", style="dim")
        return text

    def _render_analyzing_panel(self) -> Panel:
        """Render the analysis phase panel."""
        if not self.token_buffer:
            spinner = Spinner("dots", text="Analyzing task...", style="cyan")
            return Panel(
                spinner,
                title="[yellow]Analyzing[/]",
                border_style="yellow",
                box=ROUNDED,
            )
        parsed = self._try_parse_json(self.token_buffer.strip())
        if parsed:
            content = self._render_workflow_json(parsed)
        else:
            content = Text(self.token_buffer + "_", style="white")
        return Panel(
            content,
            title="[yellow]Lead Agent Thinking[/]",
            border_style="yellow",
            box=ROUNDED,
        )

    def _render_workflow_json(self, data: dict) -> Table:
        """Render workflow analysis as a table."""
        table = Table(
            box=ROUNDED, show_header=False, border_style="yellow", padding=(0, 1)
        )
        table.add_column("Key", style="bold yellow")
        table.add_column("Value", style="white")
        if "workflow" in data:
            table.add_row(
                "Workflow", Text(data["workflow"].upper(), style="bold magenta")
            )
        if "complexity" in data:
            complexity_style = {"low": "green", "medium": "yellow", "high": "red"}.get(
                data["complexity"], "white"
            )
            table.add_row(
                "Complexity", Text(data["complexity"], style=complexity_style)
            )
        if "reasoning" in data:
            reasoning = data["reasoning"]
            if len(reasoning) > 120:
                reasoning = reasoning[:120] + "..."
            table.add_row("Reasoning", Text(reasoning, style="italic dim"))
        if "task_assignments" in data:
            table.add_row(
                "Tasks", Text(str(len(data["task_assignments"])), style="cyan")
            )
        return table

    def _render_workflow_info(self) -> Panel | None:
        """Render the workflow info panel."""
        if not self.workflow_info:
            return None
        table = Table(
            box=ROUNDED, show_header=False, border_style="blue", padding=(0, 1)
        )
        table.add_column("", style="bold blue")
        table.add_column("", style="white")
        workflow = self.workflow_info.get("workflow", "unknown")
        complexity = self.workflow_info.get("complexity", "N/A")
        table.add_row("Workflow", Text(workflow.upper(), style="bold magenta"))
        complexity_style = {"low": "green", "medium": "yellow", "high": "red"}.get(
            complexity, "white"
        )
        table.add_row("Complexity", Text(complexity, style=complexity_style))
        return Panel(
            table, title="[blue]Workflow Selected[/]", border_style="blue", box=ROUNDED
        )

    def _render_agents_status(self) -> Panel | None:
        """Render the agent status table."""
        if not self.agent_states:
            return None
        table = Table(box=ROUNDED, border_style="green", show_lines=True)
        table.add_column("Agent", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Task / Result", overflow="fold")
        for agent_id, state in self.agent_states.items():
            color = state.get("color", "white")
            status = state.get("status", "pending")
            task = state.get("task", "")
            result = state.get("result", "")
            if status == "pending":
                status_text = Text("[ ] PENDING", style="dim")
            elif status == "running":
                status_text = Text("[~] RUNNING", style=f"bold {color}")
            elif status == "completed":
                status_text = Text("[v] DONE", style="bold green")
            elif status == "error":
                status_text = Text("[x] ERROR", style="bold red")
            else:
                status_text = Text(status, style="dim")
            if status == "completed" and result:
                content = result[:80] + "..." if len(result) > 80 else result
            else:
                content = task[:60] + "..." if len(task) > 60 else task
            table.add_row(
                Text(agent_id, style=f"bold {color}"),
                status_text,
                Text(content, style="white" if status == "completed" else "dim"),
            )
        return Panel(
            table, title="[green]Agent Execution[/]", border_style="green", box=ROUNDED
        )

    def _render_final_result(self) -> Panel | None:
        """Render the final result panel with citations."""
        if not self.final_result:
            return None
        result_text = self.final_result
        citation_pattern = r"\[([^\]]+_agent)\]"
        parts = re.split(citation_pattern, result_text)
        styled_text = Text()
        for part in parts:
            if part.endswith("_agent") and part in self.agent_states:
                color = self._get_agent_color(part)
                styled_text.append(f"[{part}]", style=f"bold {color}")
            else:
                bold_parts = re.split(r"\*\*([^*]+)\*\*", part)
                for j, bp in enumerate(bold_parts):
                    if j % 2 == 1:
                        styled_text.append(bp, style="bold white")
                    else:
                        styled_text.append(bp, style="white")
        content_parts = [styled_text]
        citations_found = re.findall(citation_pattern, result_text)
        if citations_found:
            content_parts.append(Text())
            cite_table = Table(
                box=ROUNDED, border_style="dim blue", show_header=True, title="Sources"
            )
            cite_table.add_column("Agent", style="bold")
            cite_table.add_column("Contribution", style="dim")
            seen = set()
            for agent_id in citations_found:
                if agent_id in seen:
                    continue
                seen.add(agent_id)
                color = self._get_agent_color(agent_id)
                result = self.agent_states.get(agent_id, {}).get("result", "")
                preview = result[:50] + "..." if len(result) > 50 else result
                cite_table.add_row(Text(agent_id, style=f"bold {color}"), preview)
            content_parts.append(cite_table)
        return Panel(
            Group(*content_parts),
            title="[bold white on blue] FINAL RESULT [/]",
            border_style="bold blue",
            box=HEAVY,
            padding=(1, 2),
        )

    def _render_synthesizing(self) -> Panel:
        """Render the synthesis phase panel."""
        if self.token_buffer and self.phase == "synthesizing":
            content = Text(self.token_buffer + "_", style="white")
            return Panel(
                content,
                title="[magenta]Synthesizing Results[/]",
                border_style="magenta",
                box=ROUNDED,
            )
        else:
            spinner = Spinner(
                "dots", text="Synthesizing agent results...", style="magenta"
            )
            return Panel(
                spinner,
                title="[magenta]Synthesizing[/]",
                border_style="magenta",
                box=ROUNDED,
            )

    def build_display(self) -> Group:
        """Build the complete display for the current state."""
        components = []
        components.append(self._render_header())
        components.append(Text())
        components.append(self._render_phase_indicator())
        components.append(Text())
        if self.phase == "initializing":
            spinner = Spinner(
                "dots", text="Initializing orchestration...", style="cyan"
            )
            components.append(Panel(spinner, border_style="cyan", box=ROUNDED))
        elif self.phase == "analyzing":
            components.append(self._render_analyzing_panel())
        elif self.phase == "executing":
            if wf := self._render_workflow_info():
                components.append(wf)
                components.append(Text())
            if agents := self._render_agents_status():
                components.append(agents)
        elif self.phase == "synthesizing":
            if wf := self._render_workflow_info():
                components.append(wf)
                components.append(Text())
            if agents := self._render_agents_status():
                components.append(agents)
                components.append(Text())
            components.append(self._render_synthesizing())
        elif self.phase == "complete":
            if wf := self._render_workflow_info():
                components.append(wf)
                components.append(Text())
            if agents := self._render_agents_status():
                components.append(agents)
                components.append(Text())
            if final := self._render_final_result():
                components.append(final)
            components.append(Text())
            components.append(
                Text.from_markup("[bold green]✓ Orchestration Complete[/]")
            )
        return Group(*components)

    def on_token(self, content: str):
        """Handle a streaming token event."""
        self.token_buffer += content
        if self.phase == "initializing":
            self.phase = "analyzing"

    def on_workflow_selected(
        self, workflow: str, complexity: str, task_assignments: list
    ):
        """Handle workflow selection event."""
        self.workflow_info = {"workflow": workflow, "complexity": complexity}
        self.task_assignments = task_assignments
        self.token_buffer = ""
        self.phase = "executing"
        for i, task in enumerate(task_assignments):
            agent_id = task.get("agent_id", f"agent_{i}")
            self.agent_states[agent_id] = {
                "status": "pending",
                "task": task.get("task", ""),
                "result": "",
                "color": self._assign_agent_color(agent_id),
            }

    def on_parallel_start(self, agent_ids: list):
        """Handle parallel execution start event."""
        for agent_id in agent_ids:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]["status"] = "running"
            else:
                self.agent_states[agent_id] = {
                    "status": "running",
                    "task": "",
                    "result": "",
                    "color": self._assign_agent_color(agent_id),
                }

    def on_agent_start(self, agent_id: str, task: str):
        """Handle individual agent start event."""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {
                "status": "running",
                "task": task,
                "result": "",
                "color": self._assign_agent_color(agent_id),
            }
        else:
            self.agent_states[agent_id]["status"] = "running"
            if task:
                self.agent_states[agent_id]["task"] = task

    def on_agent_end(self, agent_id: str, result: str):
        """Handle agent completion event."""
        if agent_id in self.agent_states:
            self.agent_states[agent_id]["status"] = "completed"
            self.agent_states[agent_id]["result"] = result

    def on_parallel_complete(self):
        """Handle parallel execution completion event."""
        pass

    def on_synthesize_start(self):
        """Handle synthesis phase start."""
        self.phase = "synthesizing"
        self.token_buffer = ""

    def on_final_result(self, result: str, citations: list = None):
        """Handle final result event."""
        self.final_result = result
        self.citations = citations or []
        self.phase = "complete"

    def on_error(self, agent_id: str, error: str):
        """Handle error event."""
        if agent_id in self.agent_states:
            self.agent_states[agent_id]["status"] = "error"
            self.agent_states[agent_id]["result"] = f"Error: {error}"


async def run_with_console(
    prompt: str,
    agents: list,
    base_url: str,
    api_key: str,
    model_name: str,
    refresh_per_second: int = 12,
) -> str | None:
    """Run orchestration with rich console visualization.

    Provides a beautiful terminal UI showing real-time progress of the
    orchestration, including phase indicators, agent status, and results.

    Args:
        prompt: The user prompt or task description.
        agents: List of worker agents to orchestrate.
        base_url: API endpoint URL.
        api_key: API authentication key.
        model_name: Model identifier for the lead agent.
        refresh_per_second: Console refresh rate. Defaults to 12.

    Returns:
        Final result string, or None if no result was produced.

    Example:
        >>> result = await run_with_console(
        ...     prompt="What's the weather?",
        ...     agents=[weather_agent],
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="sk-...",
        ...     model_name="gpt-4o",
        ... )
    """
    console = Console()
    renderer = KivaLiveRenderer(prompt)

    with Live(
        renderer.build_display(),
        console=console,
        refresh_per_second=refresh_per_second,
        transient=False,
    ) as live:
        async for event in run(
            prompt=prompt,
            agents=agents,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        ):
            if event.type == "token":
                renderer.on_token(event.data["content"])
            elif event.type == "workflow_selected":
                renderer.on_workflow_selected(
                    event.data["workflow"],
                    event.data.get("complexity", ""),
                    event.data.get("task_assignments", []),
                )
            elif event.type == "parallel_start":
                renderer.on_parallel_start(event.data["agent_ids"])
            elif event.type == "agent_start":
                renderer.on_agent_start(
                    event.data.get("agent_id", event.agent_id or "unknown"),
                    event.data.get("task", ""),
                )
            elif event.type == "agent_end":
                result_data = event.data.get("result", {})
                if isinstance(result_data, dict):
                    agent_id = result_data.get("agent_id", event.agent_id or "unknown")
                    result = result_data.get("result", "")
                else:
                    agent_id = event.agent_id or "unknown"
                    result = str(result_data)
                renderer.on_agent_end(agent_id, result)
            elif event.type == "parallel_complete":
                renderer.on_parallel_complete()
                all_done = all(
                    s["status"] == "completed" for s in renderer.agent_states.values()
                )
                if all_done and renderer.phase == "executing":
                    renderer.on_synthesize_start()
            elif event.type == "final_result":
                renderer.on_final_result(
                    event.data["result"],
                    event.data.get("citations"),
                )
            elif event.type == "error":
                renderer.on_error(
                    event.agent_id or "unknown",
                    event.data.get("error", "Unknown error"),
                )
            live.update(renderer.build_display())

    return renderer.final_result
