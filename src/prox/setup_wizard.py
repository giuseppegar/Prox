"""Setup wizard condiviso: auto-detect provider, test connessione, modalità default."""
import os
import sys
from typing import Optional

import yaml
import httpx

CONFIG_PATH = os.path.expanduser("~/.prox/config.yaml")

PROVIDER_TEMPLATES = {
    "deepseek": {"type": "native", "base_url": "https://api.deepseek.com/v1/chat/completions"},
    "openai": {"type": "native", "base_url": "https://api.openai.com/v1/chat/completions"},
    "anthropic": {"type": "native", "base_url": "https://api.anthropic.com/v1/messages"},
    "openrouter": {"type": "proxy", "base_url": "https://openrouter.ai/api/v1/chat/completions"},
    "groq": {"type": "proxy", "base_url": "https://api.groq.com/openai/v1/chat/completions"},
    "google": {"type": "native", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
}

KNOWN_PROVIDERS = list(PROVIDER_TEMPLATES.keys())


def run_setup_wizard(console=None, rich_console=None) -> dict:
    """Esegue il wizard di configurazione. Restituisce la config."""
    cprint = rich_console.print if rich_console else print

    config = _load_existing_config()

    cprint("\n[bold]Prox · Setup guidato[/bold]")
    cprint("Configura i provider LLM.\n")

    providers = config.get("providers", {})
    if not providers:
        providers = _configure_providers(cprint)

    config["providers"] = providers

    assignments = _configure_models(cprint, providers)
    config["model_assignments"] = assignments

    default_mode = _ask_default_mode(cprint)
    config["default_mode"] = default_mode

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    cprint(f"\n[green]Config salvata in {CONFIG_PATH}[/green]")
    return config


def _load_existing_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _configure_providers(cprint) -> dict:
    providers = {}
    cprint(f"Provider disponibili: {', '.join(KNOWN_PROVIDERS)}")
    cprint("INVIO senza nome per finire.\n")

    while True:
        name = _ask("  Nome provider", "").strip().lower()
        if not name:
            break

        if name not in PROVIDER_TEMPLATES:
            cprint(f"  [yellow]Provider '{name}' non nella lista. Aggiungo come nativo.[/yellow]")
            prov_type = "native"
        else:
            prov_type = PROVIDER_TEMPLATES[name]["type"]
            cprint(f"  Tipo: [cyan]{prov_type}[/cyan]")

        key = _ask_secret("  API Key", "")
        if not key:
            cprint("  [dim]Saltato, nessuna key inserita.[/dim]")
            continue

        key_env = f"{name.upper()}_API_KEY"
        os.environ[key_env] = key

        cprint("  [dim]Test connessione...[/dim]", end=" ")
        ok, msg = _test_connection(name, key)
        if ok:
            cprint(f"[green]OK![/green] {msg}")
        else:
            cprint(f"[red]Fallito:[/red] {msg}")
            if not _ask_yn("  Continuare comunque?", True):
                continue

        providers[name] = {
            "type": prov_type,
            "key_env": key_env,
            "key": key,
        }
        cprint(f"  [green]Provider '{name}' configurato.[/green]\n")

    if not providers:
        cprint("[yellow]Nessun provider configurato. Useremo deepseek con key da ambiente.[/yellow]")
        providers["deepseek"] = {
            "type": "native",
            "key_env": "DEEPSEEK_API_KEY",
        }

    return providers


def _configure_models(cprint, providers: dict) -> dict:
    provider_names = list(providers.keys())
    if not provider_names:
        return {}

    main_provider = provider_names[0]
    main_model = f"{main_provider}/deepseek-v4-pro" if main_provider == "deepseek" else f"{main_provider}/gpt-4o-mini"
    worker_model = f"{main_provider}/deepseek-v4-flash" if main_provider == "deepseek" else f"{main_provider}/gpt-4o-mini"

    cprint("\nDefault modelli per ruolo:")
    cprint(f"  Orchestratori: {main_model}")
    cprint(f"  Worker:        {worker_model}")

    if _ask_yn("  Vuoi personalizzare?", False):
        for role in ["main_orchestrator", "mini_orchestrator", "director",
                      "coder", "reviewer", "researcher", "tester", "memory"]:
            default = main_model if "orchestrator" in role or role == "director" else worker_model
            model = _ask(f"  {role:25}", default)
            assignments[role] = model
        return assignments

    return {
        "main_orchestrator": main_model,
        "mini_orchestrator": main_model,
        "director": main_model,
        "coder": worker_model,
        "reviewer": worker_model,
        "researcher": worker_model,
        "tester": worker_model,
        "memory": worker_model,
    }


def _ask_default_mode(cprint) -> str:
    cprint("\nModalità default all'avvio:")
    cprint("  [1] simple  — agente diretto, veloce, per task semplici")
    cprint("  [2] swarm   — orchestrazione completa, per task complessi")
    choice = _ask("  Scegli [1/2]", "1")
    return "swarm" if choice == "2" else "simple"


def _test_connection(provider: str, api_key: str) -> tuple:
    template = PROVIDER_TEMPLATES.get(provider)
    if not template:
        return False, "provider sconosciuto"

    url = template["base_url"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if provider in ("deepseek", "openrouter", "groq", "google") or template["type"] == "proxy":
        payload = {
            "model": "deepseek-v4-pro" if provider == "deepseek" else "gpt-4o-mini",
            "messages": [{"role": "user", "content": "say OK"}],
            "max_tokens": 10,
        }
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "say OK"}],
        }
    else:
        return False, "test non supportato per questo provider"

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=15.0)
        if resp.status_code in (200, 201):
            try:
                data = resp.json()
                if "choices" in data:
                    return True, data["choices"][0]["message"]["content"][:50]
                if "content" in data:
                    return True, str(data["content"])[:50]
                return True, "connesso"
            except Exception:
                return True, "connesso"
        elif resp.status_code == 401:
            return False, "API key non valida"
        elif resp.status_code == 403:
            return False, "accesso negato (controlla permessi)"
        else:
            return False, f"HTTP {resp.status_code}"
    except httpx.ConnectError:
        return False, "impossibile connettersi (rete?)"
    except Exception as e:
        return False, str(e)[:60]


def _ask(prompt: str, default: str = "") -> str:
    try:
        val = input(f"{prompt} [{default}]: ")
        return val.strip() if val.strip() else default
    except (KeyboardInterrupt, EOFError):
        return default


def _ask_secret(prompt: str, default: str = "") -> str:
    import getpass
    try:
        val = getpass.getpass(f"{prompt}: ")
        return val.strip() if val.strip() else default
    except (KeyboardInterrupt, EOFError):
        return default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    val = _ask(f"{prompt} {suffix}", "y" if default else "n")
    return val.lower().startswith("y")
