import os
import sys
from typing import Optional

import click

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


@click.group()
@click.version_option(version="0.1.0", prog_name="prox")
def main():
    """Prox - AI coding swarm with neural memory."""


@main.command()
@click.argument("task", required=False)
@click.option("--mode", "-m", type=click.Choice(["plan", "auto", "interactive"]),
              default="interactive", help="Modalità operativa")
@click.option("--scope", "-s", default=None, help="Directory del progetto")
@click.option("--resume", is_flag=True, help="Riprendi ultima sessione")
def run(task: Optional[str], mode: str, scope: Optional[str], resume: bool):
    """Avvia Prox ed esegui un task."""
    project_dir = scope or os.getcwd()
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
        "mode": mode,
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
        "user_query": task or "",
        "needs_user_input": False,
        "user_question": "",
        "needs_passphrase": False,
    }

    if mode == "plan":
        click.echo(f"Prox · PLAN MODE · scope: {project_dir}")
    elif mode == "auto":
        click.echo(f"Prox · AUTO MODE · scope: {project_dir}")
    else:
        click.echo(f"Prox · INTERACTIVE MODE · scope: {project_dir}")

    if not task:
        task = click.prompt("Cosa vuoi fare?")

    initial_state["user_query"] = task

    config = {"configurable": {"thread_id": project_id}}
    result = graph.invoke(initial_state, config)

    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
        click.echo(f"\n{content}")

    parking_lot = result.get("parking_lot", [])
    if parking_lot:
        click.echo(f"\nParking lot ({len(parking_lot)} idee rimandate):")
        for item in parking_lot[:5]:
            click.echo(f"  - {item}")


@main.command()
def setup():
    """Configurazione guidata di Prox (primo avvio)."""
    click.echo("Prox · Setup guidato")
    click.echo("======================")

    config_dir = os.path.expanduser("~/.prox")
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, "config.yaml")

    openai_key = click.prompt("OpenAI API Key (lascia vuoto se non hai)", default="", hide_input=True)
    anthropic_key = click.prompt("Anthropic API Key (lascia vuoto se non hai)", default="", hide_input=True)
    deepseek_key = click.prompt("DeepSeek API Key (lascia vuoto se non hai)", default="", hide_input=True)

    default_model = click.prompt("Modello default per orchestratori", default="deepseek/deepseek-chat")
    worker_model = click.prompt("Modello default per worker", default="gpt-4o-mini")

    import yaml
    config = {
        "llm": {
            "main_orchestrator": {"model": default_model},
            "mini_orchestrator": {"model": default_model},
            "director": {"model": default_model},
            "worker": {"model": worker_model},
        },
        "api_keys": {},
    }

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        config["api_keys"]["openai"] = "${OPENAI_API_KEY}"
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        config["api_keys"]["anthropic"] = "${ANTHROPIC_API_KEY}"
    if deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        config["api_keys"]["deepseek"] = "${DEEPSEEK_API_KEY}"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    vault_passphrase = click.prompt(
        "Passphrase per il vault credenziali (a memoria, non salvata)",
        default="", hide_input=True
    )

    if vault_passphrase:
        from prox.vault import CredentialStore
        store = CredentialStore()
        store.passphrase = vault_passphrase
        click.echo("Vault inizializzato.")

    click.echo(f"\nConfig salvata in {config_path}")
    click.echo("Prox e' pronto. Usa 'prox run \"tuo task\"' per iniziare.")


@main.command()
def doctor():
    """Diagnostica: verifica dipendenze e configurazione."""
    click.echo("Prox · Doctor")
    click.echo("=============")

    checks = []

    try:
        import chromadb
        checks.append(("chromadb", True, chromadb.__version__))
    except ImportError:
        checks.append(("chromadb", False, "non installato"))

    try:
        import langgraph
        checks.append(("langgraph", True, langgraph.__version__ if hasattr(langgraph, '__version__') else "ok"))
    except ImportError:
        checks.append(("langgraph", False, "non installato"))

    try:
        import litellm
        checks.append(("litellm", True, litellm.__version__))
    except ImportError:
        checks.append(("litellm", False, "non installato"))

    try:
        import pyrage
        checks.append(("pyrage", True, "ok"))
    except ImportError:
        checks.append(("pyrage", False, "non installato"))

    try:
        import textual
        checks.append(("textual", True, textual.__version__))
    except ImportError:
        checks.append(("textual", False, "non installato"))

    config_path = os.path.expanduser("~/.prox/config.yaml")
    config_exists = os.path.exists(config_path)
    checks.append(("config.yaml", config_exists, config_path if config_exists else "non trovato"))

    synapse_path = os.path.expanduser("~/.prox/synapse/chroma")
    synapse_exists = os.path.exists(synapse_path)
    checks.append(("synapse db", synapse_exists, synapse_path if synapse_exists else "non inizializzato"))

    openai_key = os.environ.get("OPENAI_API_KEY")
    checks.append(("OPENAI_API_KEY", bool(openai_key), "presente" if openai_key else "non impostata"))

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    checks.append(("ANTHROPIC_API_KEY", bool(anthropic_key), "presente" if anthropic_key else "non impostata"))

    all_ok = True
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        click.echo(f"  {status:6} {name:20} {detail}")

    if all_ok:
        click.echo("\nTutto ok. Prox e' pronto.")
    else:
        click.echo("\nAlcuni controlli falliti. Esegui 'prox setup' per configurare.")


@main.command()
def tui():
    """Avvia la TUI (Textual User Interface)."""
    try:
        from prox.tui.app import ProxApp
        app = ProxApp()
        app.run()
    except ImportError as e:
        click.echo(f"TUI non disponibile: {e}")
        click.echo("Esegui 'prox run' per la modalita' CLI.")


@main.group()
def plugin():
    """Gestione plugin di Prox."""


@plugin.command(name="list")
def plugin_list():
    """Elenca i plugin installati."""
    from prox.plugins import PluginRegistry
    registry = PluginRegistry()
    plugins = registry.list_installed()
    if not plugins:
        click.echo("Nessun plugin installato.")
        click.echo(f"Installa plugin in {registry._plugins_dir}")
        return
    for p in plugins:
        click.echo(f"  {p.name}@{p.version} - {p.description}")


@plugin.command(name="install")
@click.argument("path")
def plugin_install(path: str):
    """Installa un plugin da directory locale."""
    from prox.plugins import PluginRegistry
    registry = PluginRegistry()
    manifest = registry.install(path)
    if manifest:
        click.echo(f"Plugin '{manifest.name}@{manifest.version}' installato.")
    else:
        click.echo(f"Installazione fallita. Verifica che {path} esista e contenga plugin.yaml")


@plugin.command(name="remove")
@click.argument("name")
def plugin_remove(name: str):
    """Rimuove un plugin."""
    from prox.plugins import PluginRegistry
    registry = PluginRegistry()
    if registry.remove(name):
        click.echo(f"Plugin '{name}' rimosso.")
    else:
        click.echo(f"Plugin '{name}' non trovato.")


@main.group()
def mcp():
    """Gestione server MCP."""


@mcp.command(name="list-servers")
def mcp_list_servers():
    """Elenca i server MCP configurati."""
    from prox.mcp import list_mcp_servers
    servers = list_mcp_servers()
    if not servers:
        click.echo("Nessun server MCP configurato.")
        click.echo("Configura ~/.prox/mcp.json")
        return
    for s in servers:
        click.echo(f"  {s['name']}: {s.get('command', '?')}")


@mcp.command(name="list-tools")
def mcp_list_tools():
    """Elenca i tool MCP esposti da Prox."""
    from prox.mcp import list_mcp_tools
    tools = list_mcp_tools()
    for t in tools:
        click.echo(f"  {t['name']}: {t['description']}")


@mcp.command(name="add")
@click.argument("name")
@click.argument("command")
@click.argument("args", nargs=-1)
def mcp_add(name: str, command: str, args: tuple):
    """Aggiunge un server MCP."""
    from prox.mcp import add_mcp_server
    add_mcp_server(name, command, list(args))
    click.echo(f"Server MCP '{name}' aggiunto.")


@mcp.command(name="remove")
@click.argument("name")
def mcp_remove(name: str):
    """Rimuove un server MCP."""
    from prox.mcp import remove_mcp_server
    remove_mcp_server(name)
    click.echo(f"Server MCP '{name}' rimosso.")


if __name__ == "__main__":
    main()
