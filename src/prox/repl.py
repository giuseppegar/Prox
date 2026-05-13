"""Prox REPL — interactive terminal interface like Claude Code / opencode.

Uses prompt_toolkit for input with history and rich for markdown rendering.
Streams the LangGraph execution in real-time.
"""
import os
import sys
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

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

HISTORY_FILE = os.path.expanduser("~/.prox/repl_history")
CONFIG_PATH = os.path.expanduser("~/.prox/config.yaml")

REPL_STYLE = Style.from_dict({
    "prompt": "bold green",
    "separator": "dim",
})

COMMANDS = WordCompleter([
    "/plan", "/auto", "/interactive", "/mode",
    "/clear", "/status", "/parking", "/help", "/exit",
], ignore_case=True)


def _load_keys_from_config() -> None:
    if not os.path.exists(CONFIG_PATH):
        return
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    providers = config.get("providers", {})
    for name, info in providers.items():
        key_env = info.get("key_env", f"{name.upper()}_API_KEY")
        if key_env not in os.environ:
            key = info.get("key")
            if key:
                os.environ[key_env] = key


class ProxREPL:
    def __init__(self, mode: str = "interactive", project_dir: Optional[str] = None):
        self._console = Console()
        self._mode = Mode(mode)
        self._project_dir = project_dir or os.getcwd()
        self._project_id = os.path.basename(self._project_dir.rstrip("/"))
        self._session = PromptSession(
            history=FileHistory(HISTORY_FILE),
            style=REPL_STYLE,
            completer=COMMANDS,
        )
        _load_keys_from_config()
        self._check_config()
        self._graph = self._build_graph()
        self._session_log: list[str] = []
        self._tasks: list[dict] = []
        self._parking_lot: list[str] = []
        self._context_tokens = 0
        self._thread_id = f"repl-{os.getpid()}"
        self._running = True

    def _build_graph(self):
        return create_prox_graph(
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

    def _check_config(self) -> None:
        if not os.path.exists(CONFIG_PATH):
            self._console.print("[yellow]Nessuna configurazione trovata. Avvio setup guidato...[/yellow]")
            self._run_setup()
            return
        import yaml
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        providers = config.get("providers", {})
        if not providers:
            self._console.print("[yellow]Nessun provider configurato. Avvio setup...[/yellow]")
            self._run_setup()

    def _run_setup(self) -> None:
        self._console.print("[bold]Setup guidato[/bold]")
        self._console.print("Inserisci i provider LLM. INVIO senza nome per finire.\n")
        import yaml
        providers = {}
        while True:
            name = input("  Nome provider (es. deepseek, openai): ").strip()
            if not name:
                break
            prov_type = input("  Tipo [N]ative o [P]roxy? ").strip().upper()
            prov_type = "proxy" if prov_type == "P" else "native"
            key = input("  API Key: ").strip()
            key_env = f"{name.upper()}_API_KEY"
            if key:
                os.environ[key_env] = key
            providers[name] = {"type": prov_type, "key_env": key_env}
            if key:
                providers[name]["key"] = key

        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
        config["providers"] = providers

        assignments = {}
        for role in ["main_orchestrator", "mini_orchestrator", "director",
                      "coder", "reviewer", "researcher", "tester", "memory"]:
            assignments[role] = f"deepseek/deepseek-v4-pro"
        config["model_assignments"] = assignments

        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f)

        self._console.print(f"[green]Config salvata in {CONFIG_PATH}[/green]")

    def run(self) -> None:
        self._print_header()
        while self._running:
            try:
                user_input = self._session.prompt(
                    [("class:prompt", "> ")],
                    multiline=False,
                ).strip()
            except (KeyboardInterrupt, EOFError):
                self._console.print("\n[dim]Uscita...[/dim]")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            self._process_query(user_input)

    def _print_header(self) -> None:
        header = Text()
        header.append("Prox · coding swarm · ", style="bold")
        header.append(f"{self._mode.value}", style="bold cyan")
        header.append(f"\nproject: ", style="dim")
        header.append(f"{self._project_id}", style="yellow")
        header.append(f" · ctx: ", style="dim")
        header.append(f"{self._context_tokens//1000}k", style="dim")
        header.append("\n/help per comandi · Ctrl+C o /exit per uscire")
        self._console.print(Panel(header, box=box.ROUNDED))

    def _handle_command(self, cmd: str) -> None:
        parts = cmd.lower().strip().split()
        command = parts[0]

        if command == "/exit":
            self._running = False
            self._console.print("[dim]Arrivederci.[/dim]")

        elif command == "/plan":
            self._mode = Mode.PLAN
            self._console.print("[bold yellow]Modalità: PLAN (solo analisi, nessuna scrittura)[/bold yellow]")

        elif command == "/auto":
            self._mode = Mode.AUTO
            self._console.print("[bold red]Modalità: AUTO (esecuzione automatica, nessuna conferma)[/bold red]")

        elif command == "/interactive":
            self._mode = Mode.INTERACTIVE
            self._console.print("[bold green]Modalità: INTERACTIVE (chiede solo per operazioni pericolose)[/bold green]")

        elif command == "/mode":
            colors = {Mode.PLAN: "yellow", Mode.AUTO: "red", Mode.INTERACTIVE: "green"}
            self._console.print(f"Modalità corrente: [{colors.get(self._mode, 'white')}]{self._mode.value}[/]")

        elif command == "/clear":
            self._console.clear()
            self._print_header()

        elif command == "/status":
            self._print_status()

        elif command == "/parking":
            self._print_parking_lot()

        elif command == "/help":
            self._print_help()

        else:
            self._console.print(f"[red]Comando sconosciuto: {command}[/red]")

    def _process_query(self, query: str) -> None:
        state = {
            "messages": [],
            "mode": self._mode.value,
            "project_dir": self._project_dir,
            "project_id": self._project_id,
            "tasks": self._tasks,
            "parking_lot": self._parking_lot,
            "session_log": self._session_log,
            "context_tokens": self._context_tokens,
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

        config = {"configurable": {"thread_id": self._thread_id}}

        try:
            for chunk in self._graph.stream(state, config, stream_mode="values"):
                messages = chunk.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    content = ""
                    if isinstance(last_msg, dict):
                        content = last_msg.get("content", "")
                    elif hasattr(last_msg, "content"):
                        content = last_msg.content
                    else:
                        content = str(last_msg)

                    if content:
                        self._render_output(content)

                new_tasks = chunk.get("tasks", [])
                if new_tasks:
                    self._tasks = new_tasks

                new_pl = chunk.get("parking_lot", [])
                if new_pl:
                    self._parking_lot = new_pl

                new_log = chunk.get("session_log", [])
                if new_log:
                    self._session_log = new_log

        except Exception as e:
            self._console.print(f"[red]Errore: {e}[/red]")

    def _render_output(self, content: str) -> None:
        content = content.strip()
        if not content:
            return

        if content.startswith("COMPLEXITY:") or content.startswith("Director dispatcha"):
            self._console.print(f"  🧠 [bold blue]MainOrch[/] · {content[:120]}")
        elif "[write_file]" in content or "[read_file]" in content or "[list_directory]" in content:
            self._console.print(f"  ✏️  [bold green]Coder[/] · {content[:120]}")
        elif "[analyze_diff]" in content or "[read_file_review]" in content or "⚠️" in content:
            self._console.print(f"  🔍 [bold magenta]Reviewer[/] · {content[:150]}")
        elif "[web_search]" in content or "[fetch_docs]" in content or "[check_package_version]" in content:
            self._console.print(f"  🌐 [bold cyan]Researcher[/] · {content[:150]}")
        elif "[run_" in content or "test" in content.lower():
            self._console.print(f"  🧪 [bold yellow]Tester[/] · {content[:120]}")
        elif "Tutti i task completati" in content or "Sessione completata" in content:
            self._console.print(f"  ✅ [bold green]{content}[/bold green]")
        elif "Errore" in content or "Error" in content:
            short = content[:200]
            self._console.print(f"  [red]{short}[/red]")
        else:
            if len(content) < 300:
                self._console.print(f"  {content}")
            else:
                try:
                    md = Markdown(content)
                    self._console.print(md)
                except Exception:
                    self._console.print(f"  {content[:300]}")

    def _print_status(self) -> None:
        self._console.print("[bold]Swarm Status[/bold]")
        self._console.print(f"  Modalità: {self._mode.value}")
        self._console.print(f"  Project:  {self._project_dir}")
        self._console.print(f"  Context:  {self._context_tokens//1000}k / 200k")
        self._console.print(f"  Session:  {self._thread_id}")

        if self._tasks:
            self._console.print("\n[bold]Tasks[/bold]")
            status_icons = {
                "pending": "⏳", "in_progress": "🔄",
                "done": "✅", "blocked": "🔴",
            }
            for t in self._tasks:
                icon = status_icons.get(t.get("status", ""), "⏳")
                self._console.print(f"  {icon} {t.get('description', '?')[:60]}")
        else:
            self._console.print("  [dim]Nessun task attivo[/dim]")

    def _print_parking_lot(self) -> None:
        if not self._parking_lot:
            self._console.print("[dim]Parking lot vuoto[/dim]")
            return
        self._console.print(f"[bold]Parking Lot ({len(self._parking_lot)})[/bold]")
        for item in self._parking_lot[:5]:
            self._console.print(f"  - {item[:80]}")

    def _print_help(self) -> None:
        help_text = """
[bold]Comandi Prox REPL[/bold]

[bold green]Input:[/] scrivi un task e premi Invio per eseguirlo

[bold yellow]Comandi:[/]
  /plan          Modalità plan (solo analisi, nessuna modifica)
  /auto          Modalità auto (esecuzione automatica)
  /interactive   Modalità interactive (chiede solo per pericoli)
  /mode          Mostra modalità corrente
  /status        Mostra stato agenti e task
  /parking       Mostra parking lot
  /clear         Pulisce lo schermo
  /help          Questo aiuto
  /exit o Ctrl+C Esce

[bold dim]Prox esegue lo swarm nella directory corrente.
All'interno della directory: lettura/scrittura automatica.
Fuori dalla directory: chiede conferma (in interactive mode).[/]
"""
        self._console.print(Markdown(help_text))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prox REPL")
    parser.add_argument("--mode", "-m", choices=["plan", "auto", "interactive"],
                        default="interactive")
    parser.add_argument("--project", "-p", default=None)
    parser.add_argument("--run", "-r", default=None, help="Esegui task one-shot")
    args = parser.parse_args()

    if args.run:
        _run_oneshot(args.run, args.mode, args.project)
        return

    repl = ProxREPL(mode=args.mode, project_dir=args.project)
    repl.run()


def _run_oneshot(query: str, mode: str, project_dir: Optional[str]):
    _load_keys_from_config()
    from prox.graph.state import AgentState
    console = Console()

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

    pd = project_dir or os.getcwd()
    pid = os.path.basename(pd.rstrip("/"))

    state: AgentState = {
        "messages": [], "mode": mode, "project_dir": pd, "project_id": pid,
        "tasks": [], "parking_lot": [], "session_log": [],
        "context_tokens": 0, "max_tokens": 200000, "warn_tokens": 100000, "auto_save_tokens": 150000,
        "active_mini_orch_count": 0, "consensus_reached": False, "director_plan": {},
        "worker_results": {}, "next_agent": "",
        "user_query": query, "needs_user_input": False, "user_question": "", "needs_passphrase": False,
    }

    config = {"configurable": {"thread_id": f"oneshot-{os.getpid()}"}}
    result = graph.invoke(state, config)

    messages = result.get("messages", [])
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        if content.strip():
            console.print(f"  {content[:500]}")

    tasks = result.get("tasks", [])
    if tasks:
        console.print(f"\n[bold]Tasks:[/] {[(t.get('description','')[:40], t.get('status')) for t in tasks]}")

    parking_lot = result.get("parking_lot", [])
    if parking_lot:
        console.print(f"[bold]Parking lot:[/] {parking_lot}")
