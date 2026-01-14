"""Rich console output for Kiva orchestration.

This module provides beautiful terminal visualization for orchestration
execution using the Rich library. It displays real-time progress including
workflow phases, agent status, verification status, retry attempts, and
final results.

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

from kiva.events import (
    FINAL_VERIFICATION_ERROR,
    FINAL_VERIFICATION_FAILED,
    FINAL_VERIFICATION_MAX_REACHED,
    FINAL_VERIFICATION_PASSED,
    FINAL_VERIFICATION_START,
    RETRY_COMPLETED,
    RETRY_SKIPPED,
    RETRY_TRIGGERED,
    WORKER_VERIFICATION_ERROR,
    WORKER_VERIFICATION_FAILED,
    WORKER_VERIFICATION_MAX_REACHED,
    WORKER_VERIFICATION_PASSED,
    WORKER_VERIFICATION_START,
)
from kiva.run import run


class KivaLiveRenderer:
    """Dynamic live renderer for Kiva orchestration visualization.

    Manages the visual state and rendering of orchestration progress,
    including phase indicators, agent status tables, instance tracking,
    and result panels.
    """

    AGENT_COLORS = ["cyan", "magenta", "yellow", "green", "blue", "red"]

    def __init__(self, prompt: str):
        """Initialize the renderer with the user prompt."""
        self.prompt = prompt
        self.token_buffer = ""
        self.workflow_info: dict = {}
        self.task_assignments: list = []
        self.agent_states: dict[str, dict] = {}
        self.instance_states: dict[str, dict] = {}  # Track individual instances
        self.final_result: str | None = None
        self.citations: list = []
        self.phase = "initializing"
        self.color_index = 0
        self.parallel_strategy: str = "none"
        self.total_instances: int = 0
        
        # Verification state tracking
        self.worker_verification_state: dict = {
            "status": "pending",  # pending, verifying, passed, failed, max_reached
            "iteration": 0,
            "max_iterations": 3,
            "failed_agents": [],
            "results": [],
        }
        self.final_verification_state: dict = {
            "status": "pending",  # pending, verifying, passed, failed, max_reached
            "iteration": 0,
            "max_iterations": 2,
            "rejection_reason": None,
        }
        self.retry_state: dict = {
            "active": False,
            "iteration": 0,
            "agents": [],
        }

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
        phases = [
            "initializing",
            "analyzing",
            "executing",
            "verifying",
            "synthesizing",
            "final_verify",
            "complete",
        ]
        phase_icons = {
            "initializing": ("[ ]", "dim"),
            "analyzing": ("[~]", "yellow"),
            "executing": ("[>]", "green"),
            "verifying": ("[?]", "cyan"),
            "synthesizing": ("[*]", "magenta"),
            "final_verify": ("[!]", "blue"),
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
        if "parallel_strategy" in data and data["parallel_strategy"] != "none":
            table.add_row(
                "Parallel Strategy",
                Text(data["parallel_strategy"].upper(), style="bold green"),
            )
        if "total_instances" in data and data["total_instances"] > 1:
            table.add_row(
                "Total Instances",
                Text(str(data["total_instances"]), style="bold cyan"),
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

        # Show parallel strategy if not "none"
        if self.parallel_strategy and self.parallel_strategy != "none":
            table.add_row(
                "Parallel Strategy",
                Text(self.parallel_strategy.upper(), style="bold green"),
            )
        if self.total_instances > 1:
            table.add_row(
                "Total Instances",
                Text(str(self.total_instances), style="bold cyan"),
            )

        return Panel(
            table, title="[blue]Workflow Selected[/]", border_style="blue", box=ROUNDED
        )

    def _render_agents_status(self) -> Panel | None:
        """Render the agent status table with instance support."""
        # Filter out hidden agent states (used only for color mapping in parallel mode)
        visible_agent_states = {
            k: v for k, v in self.agent_states.items() if not v.get("_hidden", False)
        }

        if not visible_agent_states and not self.instance_states:
            return None

        table = Table(box=ROUNDED, border_style="green", show_lines=True)
        table.add_column("Agent", style="bold")
        table.add_column("Instance", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Task / Result", overflow="fold")

        # Render regular agent states (only when not using parallel instances)
        for agent_id, state in visible_agent_states.items():
            color = state.get("color", "white")
            status = state.get("status", "pending")
            task = state.get("task", "")
            result = state.get("result", "")

            status_text = self._get_status_text(status, color)

            if status == "completed" and result:
                content = result[:80] + "..." if len(result) > 80 else result
            else:
                content = task[:60] + "..." if len(task) > 60 else task

            table.add_row(
                Text(agent_id, style=f"bold {color}"),
                Text("-", style="dim"),
                status_text,
                Text(content, style="white" if status == "completed" else "dim"),
            )

        # Render instance states (for parallel instances)
        for instance_id, state in self.instance_states.items():
            agent_id = state.get("agent_id", "unknown")
            color = state.get("color", "white")
            status = state.get("status", "pending")
            task = state.get("task", "")
            result = state.get("result", "")

            status_text = self._get_status_text(status, color)

            if status == "completed" and result:
                content = result[:80] + "..." if len(result) > 80 else result
            else:
                content = task[:60] + "..." if len(task) > 60 else task

            # Show shortened instance ID
            short_instance = (
                instance_id.split("-")[-1][:8]
                if "-" in instance_id
                else instance_id[:8]
            )

            table.add_row(
                Text(agent_id, style=f"bold {color}"),
                Text(short_instance, style="dim cyan"),
                status_text,
                Text(content, style="white" if status == "completed" else "dim"),
            )

        title = "[green]Agent Execution[/]"
        if self.instance_states:
            title = "[green]Agent Instances Execution[/]"

        return Panel(table, title=title, border_style="green", box=ROUNDED)

    def _get_status_text(self, status: str, color: str) -> Text:
        """Get formatted status text."""
        if status == "pending":
            return Text("[ ] PENDING", style="dim")
        elif status == "spawned":
            return Text("[+] SPAWNED", style=f"bold {color}")
        elif status == "running":
            return Text("[~] RUNNING", style=f"bold {color}")
        elif status == "completed":
            return Text("[v] DONE", style="bold green")
        elif status == "error":
            return Text("[x] ERROR", style="bold red")
        else:
            return Text(status, style="dim")

    def _render_worker_verification_panel(self) -> Panel | None:
        """Render the worker verification status panel."""
        state = self.worker_verification_state
        status = state.get("status", "pending")

        if status == "pending":
            return None

        table = Table(
            box=ROUNDED, show_header=False, border_style="cyan", padding=(0, 1)
        )
        table.add_column("", style="bold cyan")
        table.add_column("", style="white")

        # Status row with icon
        status_icons = {
            "verifying": ("🔍", "yellow"),
            "passed": ("✓", "green"),
            "failed": ("✗", "red"),
            "max_reached": ("⚠", "yellow"),
        }
        icon, style = status_icons.get(status, ("?", "dim"))
        table.add_row(
            "Status",
            Text(f"{icon} {status.upper()}", style=f"bold {style}"),
        )
        
        # Iteration progress
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        progress_text = f"{iteration}/{max_iterations}"
        table.add_row("Iteration", Text(progress_text, style="cyan"))

        # Failed agents (if any)
        failed_agents = state.get("failed_agents", [])
        if failed_agents:
            failed_text = ", ".join(failed_agents[:3])
            if len(failed_agents) > 3:
                failed_text += f" (+{len(failed_agents) - 3} more)"
            table.add_row("Failed Agents", Text(failed_text, style="red"))

        title = "[cyan]Worker Verification[/]"
        if status == "verifying":
            spinner = Spinner("dots", text="Verifying worker outputs...", style="cyan")
            return Panel(
                spinner,
                title=title,
                border_style="cyan",
                box=ROUNDED,
            )
        
        return Panel(table, title=title, border_style="cyan", box=ROUNDED)

    def _render_final_verification_panel(self) -> Panel | None:
        """Render the final verification status panel."""
        state = self.final_verification_state
        status = state.get("status", "pending")
        
        if status == "pending":
            return None
        
        table = Table(
            box=ROUNDED, show_header=False, border_style="blue", padding=(0, 1)
        )
        table.add_column("", style="bold blue")
        table.add_column("", style="white")
        
        # Status row with icon
        status_icons = {
            "verifying": ("🔍", "yellow"),
            "passed": ("✓", "green"),
            "failed": ("✗", "red"),
            "max_reached": ("⚠", "yellow"),
        }
        icon, style = status_icons.get(status, ("?", "dim"))
        table.add_row(
            "Status",
            Text(f"{icon} {status.upper()}", style=f"bold {style}"),
        )
        
        # Iteration progress
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 2)
        progress_text = f"{iteration}/{max_iterations}"
        table.add_row("Workflow Iteration", Text(progress_text, style="blue"))
        
        # Rejection reason (if any)
        rejection_reason = state.get("rejection_reason")
        if rejection_reason:
            reason_text = (
                rejection_reason[:60] + "..."
                if len(rejection_reason) > 60
                else rejection_reason
            )
            table.add_row("Reason", Text(reason_text, style="red italic"))
        
        title = "[blue]Final Verification[/]"
        if status == "verifying":
            spinner = Spinner(
                "dots", text="Verifying final result...", style="blue"
            )
            return Panel(
                spinner,
                title=title,
                border_style="blue",
                box=ROUNDED,
            )
        
        return Panel(table, title=title, border_style="blue", box=ROUNDED)

    def _render_retry_panel(self) -> Panel | None:
        """Render the retry status panel."""
        if not self.retry_state.get("active", False):
            return None
        
        iteration = self.retry_state.get("iteration", 0)
        agents = self.retry_state.get("agents", [])
        
        table = Table(
            box=ROUNDED, show_header=False, border_style="yellow", padding=(0, 1)
        )
        table.add_column("", style="bold yellow")
        table.add_column("", style="white")
        
        table.add_row(
            "Status",
            Text("🔄 RETRYING", style="bold yellow"),
        )
        table.add_row("Retry Attempt", Text(str(iteration), style="yellow"))
        
        if agents:
            agents_text = ", ".join(agents[:3])
            if len(agents) > 3:
                agents_text += f" (+{len(agents) - 3} more)"
            table.add_row("Retrying Agents", Text(agents_text, style="yellow"))
        
        return Panel(
            table,
            title="[yellow]Worker Retry[/]",
            border_style="yellow",
            box=ROUNDED,
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
            # Show retry panel if active
            if retry := self._render_retry_panel():
                components.append(Text())
                components.append(retry)
        elif self.phase == "verifying":
            if wf := self._render_workflow_info():
                components.append(wf)
                components.append(Text())
            if agents := self._render_agents_status():
                components.append(agents)
                components.append(Text())
            # Show worker verification panel
            if worker_verify := self._render_worker_verification_panel():
                components.append(worker_verify)
            # Show retry panel if active
            if retry := self._render_retry_panel():
                components.append(Text())
                components.append(retry)
        elif self.phase == "synthesizing":
            if wf := self._render_workflow_info():
                components.append(wf)
                components.append(Text())
            if agents := self._render_agents_status():
                components.append(agents)
                components.append(Text())
            # Show worker verification result if available
            if worker_verify := self._render_worker_verification_panel():
                components.append(worker_verify)
                components.append(Text())
            components.append(self._render_synthesizing())
        elif self.phase == "final_verify":
            if wf := self._render_workflow_info():
                components.append(wf)
                components.append(Text())
            if agents := self._render_agents_status():
                components.append(agents)
                components.append(Text())
            # Show worker verification result
            if worker_verify := self._render_worker_verification_panel():
                components.append(worker_verify)
                components.append(Text())
            # Show final verification panel
            if final_verify := self._render_final_verification_panel():
                components.append(final_verify)
        elif self.phase == "complete":
            if wf := self._render_workflow_info():
                components.append(wf)
                components.append(Text())
            if agents := self._render_agents_status():
                components.append(agents)
                components.append(Text())
            # Show verification results summary
            if worker_verify := self._render_worker_verification_panel():
                components.append(worker_verify)
                components.append(Text())
            if final_verify := self._render_final_verification_panel():
                components.append(final_verify)
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
        self,
        workflow: str,
        complexity: str,
        task_assignments: list,
        parallel_strategy: str = "none",
        total_instances: int = 1,
    ):
        """Handle workflow selection event."""
        self.workflow_info = {"workflow": workflow, "complexity": complexity}
        self.task_assignments = task_assignments
        self.parallel_strategy = parallel_strategy
        self.total_instances = total_instances
        self.token_buffer = ""
        self.phase = "executing"

        # Only populate agent_states if NOT using parallel instances
        # When using fan_out/map_reduce, instance_states will be populated
        # by instance events
        if parallel_strategy == "none":
            for i, task in enumerate(task_assignments):
                agent_id = task.get("agent_id", f"agent_{i}")
                self.agent_states[agent_id] = {
                    "status": "pending",
                    "task": task.get("task", ""),
                    "result": "",
                    "color": self._assign_agent_color(agent_id),
                    "instances": task.get("instances", 1),
                }
        else:
            # For parallel strategies, just pre-assign colors for agents
            for i, task in enumerate(task_assignments):
                agent_id = task.get("agent_id", f"agent_{i}")
                if agent_id not in self.agent_states:
                    # Only store color mapping, don't create display entry
                    self._assign_agent_color(agent_id)
                    # Store in a temporary dict for color lookup
                    self.agent_states[agent_id] = {
                        "color": self.AGENT_COLORS[
                            (self.color_index - 1) % len(self.AGENT_COLORS)
                        ],
                        "_hidden": True,  # Mark as hidden, won't be rendered
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

    def on_instance_spawn(self, instance_id: str, agent_id: str, task: str):
        """Handle instance spawn event."""
        color = self._get_agent_color(agent_id)
        if not color or color == "white":
            color = self._assign_agent_color(agent_id)
        self.instance_states[instance_id] = {
            "agent_id": agent_id,
            "status": "spawned",
            "task": task,
            "result": "",
            "color": color,
        }

    def on_instance_start(self, instance_id: str, agent_id: str, task: str):
        """Handle instance start event."""
        if instance_id in self.instance_states:
            self.instance_states[instance_id]["status"] = "running"
            if task:
                self.instance_states[instance_id]["task"] = task
        else:
            color = self._get_agent_color(agent_id)
            if not color or color == "white":
                color = self._assign_agent_color(agent_id)
            self.instance_states[instance_id] = {
                "agent_id": agent_id,
                "status": "running",
                "task": task,
                "result": "",
                "color": color,
            }

    def on_instance_end(self, instance_id: str, agent_id: str, result: str):
        """Handle instance end event."""
        if instance_id in self.instance_states:
            self.instance_states[instance_id]["status"] = "completed"
            self.instance_states[instance_id]["result"] = result

    def on_instance_complete(self, instance_id: str, agent_id: str, success: bool):
        """Handle instance complete event."""
        if instance_id in self.instance_states:
            self.instance_states[instance_id]["status"] = (
                "completed" if success else "error"
            )

    def on_instance_result(
        self, instance_id: str, agent_id: str, result: str, error: str | None
    ):
        """Handle instance result event."""
        if instance_id in self.instance_states:
            self.instance_states[instance_id]["status"] = (
                "error" if error else "completed"
            )
            self.instance_states[instance_id]["result"] = error or result
        else:
            color = self._get_agent_color(agent_id)
            if not color or color == "white":
                color = self._assign_agent_color(agent_id)
            self.instance_states[instance_id] = {
                "agent_id": agent_id,
                "status": "error" if error else "completed",
                "task": "",
                "result": error or result,
                "color": color,
            }

    def on_parallel_instances_start(self, instance_count: int, agent_ids: list):
        """Handle parallel instances start event."""
        # Don't create visible agent_states entries for parallel instances
        # The instance_states will be populated by instance_spawn events
        for agent_id in agent_ids:
            if agent_id not in self.agent_states:
                # Only store color mapping
                color = self._assign_agent_color(agent_id)
                self.agent_states[agent_id] = {
                    "color": color,
                    "_hidden": True,
                }

    def on_parallel_instances_complete(self, results: list):
        """Handle parallel instances complete event."""
        for result in results:
            instance_id = result.get("instance_id")
            success = result.get("success", True)
            if instance_id and instance_id in self.instance_states:
                self.instance_states[instance_id]["status"] = (
                    "completed" if success else "error"
                )

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

    # Verification event handlers
    def on_worker_verification_start(self, iteration: int, max_iterations: int):
        """Handle worker verification start event."""
        self.phase = "verifying"
        self.worker_verification_state.update({
            "status": "verifying",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "failed_agents": [],
        })

    def on_worker_verification_passed(self, iteration: int, results: list):
        """Handle worker verification passed event."""
        self.worker_verification_state.update({
            "status": "passed",
            "iteration": iteration,
            "results": results,
            "failed_agents": [],
        })

    def on_worker_verification_failed(
        self, iteration: int, failed_agents: list, results: list
    ):
        """Handle worker verification failed event."""
        self.worker_verification_state.update({
            "status": "failed",
            "iteration": iteration,
            "failed_agents": failed_agents,
            "results": results,
        })

    def on_worker_verification_max_reached(
        self, iteration: int, failed_agents: list
    ):
        """Handle worker verification max iterations reached event."""
        self.worker_verification_state.update({
            "status": "max_reached",
            "iteration": iteration,
            "failed_agents": failed_agents,
        })

    def on_worker_verification_error(self, error: str, action: str):
        """Handle worker verification error event (graceful degradation).

        This is called when the verifier itself fails, triggering graceful
        degradation where verification is bypassed.
        """
        self.worker_verification_state.update({
            "status": "error",
            "error": error,
            "action": action,
        })

    def on_final_verification_start(self, iteration: int, max_iterations: int):
        """Handle final verification start event."""
        self.phase = "final_verify"
        self.final_verification_state.update({
            "status": "verifying",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "rejection_reason": None,
        })

    def on_final_verification_passed(self, iteration: int):
        """Handle final verification passed event."""
        self.final_verification_state.update({
            "status": "passed",
            "iteration": iteration,
        })

    def on_final_verification_failed(
        self, iteration: int, rejection_reason: str | None
    ):
        """Handle final verification failed event."""
        self.final_verification_state.update({
            "status": "failed",
            "iteration": iteration,
            "rejection_reason": rejection_reason,
        })

    def on_final_verification_max_reached(
        self, iteration: int, rejection_reason: str | None
    ):
        """Handle final verification max iterations reached event."""
        self.final_verification_state.update({
            "status": "max_reached",
            "iteration": iteration,
            "rejection_reason": rejection_reason,
        })

    def on_final_verification_error(self, error: str, action: str):
        """Handle final verification error event (graceful degradation).

        This is called when the verifier itself fails, triggering graceful
        degradation where verification is bypassed.
        """
        self.final_verification_state.update({
            "status": "error",
            "error": error,
            "action": action,
        })

    def on_retry_triggered(self, iteration: int, agents: list):
        """Handle retry triggered event."""
        self.retry_state.update({
            "active": True,
            "iteration": iteration,
            "agents": agents,
        })
        # Reset agent states for retrying agents
        for agent_id in agents:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]["status"] = "running"

    def on_retry_completed(self, iteration: int, results: list):
        """Handle retry completed event."""
        self.retry_state.update({
            "active": False,
            "iteration": iteration,
        })

    def on_retry_skipped(self):
        """Handle retry skipped event."""
        self.retry_state.update({
            "active": False,
        })


async def run_with_console(
    prompt: str,
    agents: list,
    base_url: str,
    api_key: str,
    model_name: str,
    refresh_per_second: int = 12,
    max_iterations: int = 10,
    custom_verifiers: list | None = None,
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
        max_iterations: Maximum iterations for verification retry. Defaults to 10.
        custom_verifiers: List of custom verifier functions. Defaults to None.

    Returns:
        Final result string, or None if no result was produced.

    Example:
        >>> result = await run_with_console(
        ...     prompt="What's the weather?",
        ...     agents=[weather_agent],
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="sk-...",
        ...     model_name="gpt-4o",
        ...     max_iterations=5,
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
            max_iterations=max_iterations,
        ):
            if event.type == "token":
                renderer.on_token(event.data["content"])
            elif event.type == "workflow_selected":
                renderer.on_workflow_selected(
                    event.data["workflow"],
                    event.data.get("complexity", ""),
                    event.data.get("task_assignments", []),
                    event.data.get("parallel_strategy", "none"),
                    event.data.get("total_instances", 1),
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
            # Parallel instance events
            elif event.type == "instance_spawn":
                renderer.on_instance_spawn(
                    event.data.get("instance_id", ""),
                    event.data.get("agent_id", ""),
                    event.data.get("task", ""),
                )
            elif event.type == "instance_start":
                renderer.on_instance_start(
                    event.data.get("instance_id", ""),
                    event.data.get("agent_id", ""),
                    event.data.get("task", ""),
                )
            elif event.type == "instance_end":
                renderer.on_instance_end(
                    event.data.get("instance_id", ""),
                    event.data.get("agent_id", ""),
                    event.data.get("result", ""),
                )
            elif event.type == "instance_complete":
                renderer.on_instance_complete(
                    event.data.get("instance_id", ""),
                    event.data.get("agent_id", ""),
                    event.data.get("success", True),
                )
            elif event.type == "instance_result":
                renderer.on_instance_result(
                    event.data.get("instance_id", ""),
                    event.data.get("agent_id", ""),
                    event.data.get("result", ""),
                    event.data.get("error"),
                )
            elif event.type == "parallel_instances_start":
                renderer.on_parallel_instances_start(
                    event.data.get("instance_count", 0),
                    event.data.get("agent_ids", []),
                )
            elif event.type == "parallel_instances_complete":
                renderer.on_parallel_instances_complete(
                    event.data.get("results", []),
                )
                # Check if all instances are done
                all_instances_done = (
                    all(
                        s["status"] in ("completed", "error")
                        for s in renderer.instance_states.values()
                    )
                    if renderer.instance_states
                    else True
                )
                all_agents_done = (
                    all(
                        s["status"] in ("completed", "error")
                        for s in renderer.agent_states.values()
                    )
                    if renderer.agent_states
                    else True
                )
                if (
                    all_instances_done
                    and all_agents_done
                    and renderer.phase == "executing"
                ):
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
            # Worker verification events
            elif event.type == WORKER_VERIFICATION_START:
                renderer.on_worker_verification_start(
                    event.data.get("iteration", 0),
                    event.data.get("max_iterations", 3),
                )
            elif event.type == WORKER_VERIFICATION_PASSED:
                renderer.on_worker_verification_passed(
                    event.data.get("iteration", 0),
                    event.data.get("results", []),
                )
            elif event.type == WORKER_VERIFICATION_FAILED:
                renderer.on_worker_verification_failed(
                    event.data.get("iteration", 0),
                    event.data.get("failed_agents", []),
                    event.data.get("results", []),
                )
            elif event.type == WORKER_VERIFICATION_MAX_REACHED:
                renderer.on_worker_verification_max_reached(
                    event.data.get("iteration", 0),
                    event.data.get("failed_agents", []),
                )
            elif event.type == WORKER_VERIFICATION_ERROR:
                renderer.on_worker_verification_error(
                    event.data.get("error", "Unknown error"),
                    event.data.get("action", "bypass_verification"),
                )
            # Final verification events
            elif event.type == FINAL_VERIFICATION_START:
                renderer.on_final_verification_start(
                    event.data.get("iteration", 0),
                    event.data.get("max_iterations", 2),
                )
            elif event.type == FINAL_VERIFICATION_PASSED:
                renderer.on_final_verification_passed(
                    event.data.get("iteration", 0),
                )
            elif event.type == FINAL_VERIFICATION_FAILED:
                renderer.on_final_verification_failed(
                    event.data.get("iteration", 0),
                    event.data.get("reason"),
                )
            elif event.type == FINAL_VERIFICATION_MAX_REACHED:
                renderer.on_final_verification_max_reached(
                    event.data.get("iteration", 0),
                    event.data.get("reason"),
                )
            elif event.type == FINAL_VERIFICATION_ERROR:
                renderer.on_final_verification_error(
                    event.data.get("error", "Unknown error"),
                    event.data.get("action", "bypass_verification"),
                )
            # Retry events
            elif event.type == RETRY_TRIGGERED:
                renderer.on_retry_triggered(
                    event.data.get("iteration", 0),
                    event.data.get("agents", []),
                )
            elif event.type == RETRY_COMPLETED:
                renderer.on_retry_completed(
                    event.data.get("iteration", 0),
                    event.data.get("results", []),
                )
            elif event.type == RETRY_SKIPPED:
                renderer.on_retry_skipped()
            live.update(renderer.build_display())

    return renderer.final_result
