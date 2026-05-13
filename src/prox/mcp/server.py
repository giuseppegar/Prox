"""Simple MCP server that exposes Prox agents as MCP tools."""
import json
import os
from typing import Any


MCP_JSON_PATH = os.path.expanduser("~/.prox/mcp.json")


def load_mcp_config() -> dict:
    if os.path.exists(MCP_JSON_PATH):
        with open(MCP_JSON_PATH) as f:
            return json.load(f)
    return {}


def save_mcp_config(config: dict) -> None:
    os.makedirs(os.path.dirname(MCP_JSON_PATH), exist_ok=True)
    with open(MCP_JSON_PATH, "w") as f:
        json.dump(config, f, indent=2)


def list_mcp_tools() -> list[dict]:
    return [
        {
            "name": "prox.task",
            "description": "Invia un task allo swarm di Prox",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Il task da eseguire"},
                    "mode": {"type": "string", "enum": ["plan", "auto", "interactive"], "default": "interactive"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "prox.review",
            "description": "Chiedi una code review al Reviewer agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Percorso del file da analizzare"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "prox.research",
            "description": "Cerca documentazione o informazioni via Researcher agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La query di ricerca"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "prox.memory.query",
            "description": "Interroga la memoria neurale Synapse",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query per la memoria"},
                    "project_id": {"type": "string", "description": "ID del progetto"},
                },
                "required": ["query"],
            },
        },
    ]


def list_mcp_servers() -> list[dict]:
    config = load_mcp_config()
    return [{"name": name, **info} for name, info in config.get("mcpServers", {}).items()]


def add_mcp_server(name: str, command: str, args: list[str] = None, env: dict = None) -> None:
    config = load_mcp_config()
    config.setdefault("mcpServers", {})[name] = {
        "command": command,
        "args": args or [],
        "env": env or {},
    }
    save_mcp_config(config)


def remove_mcp_server(name: str) -> None:
    config = load_mcp_config()
    config.get("mcpServers", {}).pop(name, None)
    save_mcp_config(config)
