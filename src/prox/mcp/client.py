"""MCP client: connect to external MCP servers for Prox agents to use."""
import json
import os
import subprocess
import asyncio
from typing import Any, Optional


MCP_JSON_PATH = os.path.expanduser("~/.prox/mcp.json")


class MCPClient:
    def __init__(self, config_path: str = MCP_JSON_PATH):
        self._config_path = config_path
        self._servers: dict[str, dict] = {}
        self._load()

    def list_servers(self) -> list[str]:
        return list(self._servers.keys())

    def get_server(self, name: str) -> Optional[dict]:
        return self._servers.get(name)

    def list_tools(self, server_name: str) -> list[dict]:
        server = self._servers.get(server_name)
        if not server:
            return []
        return server.get("tools", [])

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        server = self._servers.get(server_name)
        if not server:
            return {"error": f"Server '{server_name}' non trovato"}

        command = server.get("command", "")
        args = server.get("args", [])
        env_vars = server.get("env", {})

        env = os.environ.copy()
        for k, v in env_vars.items():
            env[k] = v

        try:
            result = subprocess.run(
                [command] + args,
                input=json.dumps({tool_name: arguments}),
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if result.stdout:
                return json.loads(result.stdout)
            return {"error": result.stderr or "nessun output"}
        except subprocess.TimeoutExpired:
            return {"error": f"Tool '{tool_name}' timeout dopo 30s"}
        except Exception as e:
            return {"error": str(e)}

    def _load(self) -> None:
        if os.path.exists(self._config_path):
            with open(self._config_path) as f:
                config = json.load(f)
                self._servers = config.get("mcpServers", {})

    def _save(self) -> None:
        config = {"mcpServers": self._servers}
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        with open(self._config_path, "w") as f:
            json.dump(config, f, indent=2)
