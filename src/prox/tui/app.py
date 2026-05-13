from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Static, RichLog, Label, Button
from textual.reactive import reactive
from textual import events


class StatusBar(Static):
    mode = reactive("interactive")
    context_tokens = reactive(0)
    project = reactive("")

    def render(self) -> str:
        return (
            f"ctx: {self.context_tokens}k/200k"
            f" · project: {self.project}"
            f" · mode: {self.mode}"
        )


class AgentStatusWidget(Static):
    agents = reactive({})

    def render(self) -> str:
        lines = ["Swarm Status", "────────────"]
        if not self.agents:
            lines.append("(idle)")
        else:
            for name, status in sorted(self.agents.items()):
                icon = {"idle": "⚪", "thinking": "🟡", "working": "🟢", "done": "✅", "blocked": "🔴"}.get(status, "⚪")
                lines.append(f"{icon} {name}: {status}")
        return "\n".join(lines)


class TaskBoard(Static):
    tasks = reactive([])
    parking_lot = reactive([])

    def render(self) -> str:
        lines = ["Tasks", "─────"]
        if not self.tasks:
            lines.append("(nessun task)")
        else:
            for t in self.tasks:
                icon = {"done": "✅", "in_progress": "🔄", "blocked": "🔴", "pending": "⏳"}.get(t.get("status"), "⏳")
                lines.append(f"{icon} {t.get('description', '')[:50]}")
        if self.parking_lot:
            lines.append("")
            lines.append(f"Parking lot ({len(self.parking_lot)})")
            for item in self.parking_lot[:5]:
                lines.append(f"  - {item[:40]}")
        return "\n".join(lines)


class ProxApp(App):
    CSS = """
    Container {
        height: 100%;
    }
    #side-panel {
        width: 35%;
        border: solid $primary;
    }
    #main-panel {
        width: 65%;
    }
    #chat-log {
        height: 1fr;
    }
    #input-area {
        height: 3;
    }
    StatusBar {
        height: 1;
        background: $primary-darken-2;
        color: $text;
    }
    AgentStatusWidget {
        height: auto;
        border: solid $primary;
        padding: 1;
    }
    TaskBoard {
        height: auto;
        border: solid $primary;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="main-panel"):
                yield RichLog(id="chat-log", highlight=True, markup=True)
                yield Input(id="input-area", placeholder="Cosa vuoi fare? (CTRL+Q per uscire)")
            with Vertical(id="side-panel"):
                yield AgentStatusWidget(id="agent-status")
                yield TaskBoard(id="task-board")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-log", RichLog).write("Prox · coding swarm\nPronto.\n")
        self.query_one("#status-bar", StatusBar).mode = "interactive"
        self.query_one("#status-bar", StatusBar).project = "prox"
        self.query_one("#status-bar", StatusBar).context_tokens = 0

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        chat = self.query_one("#chat-log", RichLog)
        chat.write(f"\n[bold]tu:[/bold] {query}")

        self.query_one("#agent-status", AgentStatusWidget).agents = {
            "MainOrch": "thinking",
        }

        response = _run_prox_query(query)

        chat.write(f"\n[bold]Prox:[/bold] {response}")

        self.query_one("#agent-status", AgentStatusWidget).agents = {}

        event.input.value = ""


def _run_prox_query(query: str) -> str:
    """Helper to run a query through the Prox graph from TUI."""
    import os
    from prox.graph import (
        Mode,
        main_orchestrator_node,
        mini_orchestrator_node,
        director_node,
        create_prox_graph,
    )
    from prox.agents import (
        coder_node,
        reviewer_node,
        researcher_node,
        tester_node,
        memory_node,
    )
    from prox.graph.state import AgentState

    project_dir = os.getcwd()
    project_id = os.path.basename(project_dir.rstrip("/"))

    graph = create_prox_graph(
        main_orch_node=main_orchestrator_node,
        mini_orch_node=mini_orchestrator_node,
        director_node=director_node,
        worker_router=None,
        coder_node=coder_node,
        reviewer_node=reviewer_node,
        researcher_node=researcher_node,
        tester_node=tester_node,
        memory_node=memory_node,
    )

    initial_state: AgentState = {
        "messages": [],
        "mode": "interactive",
        "project_dir": project_dir,
        "project_id": project_id,
        "tasks": [],
        "parking_lot": [],
        "session_log": [],
        "context_tokens": 0,
        "max_tokens": 200000,
        "warn_tokens": 100000,
        "auto_save_tokens": 150000,
        "active_mini_orch_count": 0,
        "consensus_reached": False,
        "director_plan": {},
        "worker_results": {},
        "next_agent": "",
        "user_query": query,
        "needs_user_input": False,
        "user_question": "",
        "needs_passphrase": False,
    }

    config = {"configurable": {"thread_id": project_id}}
    result = graph.invoke(initial_state, config)

    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        return last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
    return "(nessuna risposta)"
