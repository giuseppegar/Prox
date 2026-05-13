import os
import sys
from typing import Optional

import click


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="prox")
@click.option("--mode", "-m", type=click.Choice(["plan", "auto", "interactive"]),
              default="interactive", help="Modalità REPL")
@click.option("--project", "-p", default=None, help="Directory del progetto")
@click.option("--run", "-r", default=None, help="Esegui task one-shot ed esci")
@click.option("--swarm", "-s", is_flag=True, help="Avvia in modalità swarm")
@click.pass_context
def main(ctx, mode: str, project: str, run: str, swarm: bool):
    """Prox - AI coding swarm with neural memory."""
    if run:
        from prox.repl import _run_oneshot
        _run_oneshot(run, mode, project)
        ctx.exit()

    if ctx.invoked_subcommand is None:
        from prox.repl import ProxREPL
        repl = ProxREPL(mode=mode, project_dir=project, swarm=swarm)
        repl.run()
        ctx.exit()


@main.command()
@click.argument("task", required=False)
@click.option("--mode", "-m", type=click.Choice(["plan", "auto", "interactive"]),
              default="interactive", help="Modalità operativa")
@click.option("--scope", "-s", default=None, help="Directory del progetto")
def run(task: Optional[str], mode: str, scope: Optional[str]):
    """Esegui un task one-shot."""
    from prox.repl import _run_oneshot
    query = task or click.prompt("Cosa vuoi fare?")
    _run_oneshot(query, mode, scope)


@main.command()
def setup():
    """Configurazione guidata di Prox (primo avvio)."""
    click.echo("Prox · Setup guidato")
    click.echo("======================")

    config_dir = os.path.expanduser("~/.prox")
    os.makedirs(config_dir, exist_ok=True)

    from prox.llm import ModelRouter, ProviderType, discover_and_assign

    router = ModelRouter()

    click.echo("\nConfigura i provider LLM. Puoi aggiungerne piu' di uno.")
    click.echo("Un provider NATIVE da' accesso solo ai suoi modelli.")
    click.echo("Un provider PROXY (es. OpenRouter) da' accesso a TUTTI i modelli.\n")

    while True:
        name = click.prompt("Nome provider (es. deepseek, openai, openrouter) o INVIO per finire", default="")
        if not name:
            break

        click.echo(f"  Tipo: [N]ative (solo modelli {name}) o [P]roxy (tutti i modelli)?")
        type_choice = click.prompt("  Scegli N/P", default="N")
        prov_type = "proxy" if type_choice.upper() == "P" else "native"

        key_env = f"{name.upper()}_API_KEY"
        key = click.prompt(f"  API Key (o INVIO per usare ${key_env})", default="", hide_input=True)

        if key:
            os.environ[key_env] = key

        router.add_provider(name, prov_type, key=key if key else None, key_env=key_env)
        click.echo(f"  Provider '{name}' ({prov_type}) aggiunto.\n")

    click.echo("\nScoperta modelli disponibili...")
    assignments = router.discover_and_assign()

    click.echo("\nAssegnazione automatica modelli per ruolo:")
    click.echo(router.summary())

    auto_assign = click.confirm("\nConfermi queste assegnazioni?", default=True)
    if not auto_assign:
        click.echo("Puoi modificare ~/.prox/config.yaml manualmente.")

    vault_passphrase = click.prompt(
        "\nPassphrase per il vault credenziali (a memoria, non salvata)",
        default="", hide_input=True
    )

    if vault_passphrase:
        from prox.vault import CredentialStore
        store = CredentialStore()
        store.passphrase = vault_passphrase
        click.echo("Vault inizializzato.")

    click.echo(f"\nConfig salvata in ~/.prox/config.yaml")
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
        version = getattr(litellm, '__version__', None) or getattr(litellm, '_version', None)
        if hasattr(version, '__call__'):
            version = 'ok'
        checks.append(("litellm", True, str(version) if version else 'ok'))
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
