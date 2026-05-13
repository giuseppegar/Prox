"""Plugin loader and installer."""
import importlib
import os
import json
import shutil
from typing import Optional

import yaml

from .schema import PluginManifest

PLUGINS_DIR = os.path.expanduser("~/.prox/plugins")
MANIFEST_PATH = os.path.join(PLUGINS_DIR, "manifest.yaml")


class PluginRegistry:
    def __init__(self, plugins_dir: str = PLUGINS_DIR):
        self._plugins_dir = plugins_dir
        os.makedirs(plugins_dir, exist_ok=True)
        self._loaded: dict[str, PluginManifest] = {}
        self._refresh()

    def list_installed(self) -> list[PluginManifest]:
        return list(self._loaded.values())

    def get(self, name: str) -> Optional[PluginManifest]:
        return self._loaded.get(name)

    def install(self, source_path: str) -> Optional[PluginManifest]:
        if not os.path.exists(source_path):
            return None

        if os.path.isdir(source_path):
            plugin_yaml = os.path.join(source_path, "plugin.yaml")
            if not os.path.exists(plugin_yaml):
                return None

            with open(plugin_yaml) as f:
                data = yaml.safe_load(f) or {}

            manifest = PluginManifest(
                name=data.get("name", ""),
                version=data.get("version", "0.1.0"),
                description=data.get("description", ""),
                provides=data.get("provides", {}),
            )

            dest = os.path.join(self._plugins_dir, manifest.name)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(source_path, dest)

            self._loaded[manifest.name] = manifest
            self._save_manifest()
            return manifest

        return None

    def remove(self, name: str) -> bool:
        plugin_dir = os.path.join(self._plugins_dir, name)
        if os.path.exists(plugin_dir):
            shutil.rmtree(plugin_dir)
            self._loaded.pop(name, None)
            self._save_manifest()
            return True
        return False

    def load_agents(self, plugin_name: str) -> dict:
        manifest = self._loaded.get(plugin_name)
        if not manifest:
            return {}

        agents = manifest.provides.get("agents", [])
        loaded = {}
        plugin_dir = os.path.join(self._plugins_dir, plugin_name)

        for agent_def in agents:
            agent_name = agent_def.get("name", "")
            agent_file = agent_def.get("file", f"agent.py")
            agent_path = os.path.join(plugin_dir, agent_file)
            if os.path.exists(agent_path):
                spec = importlib.util.spec_from_file_location(
                    f"prox_plugins.{plugin_name}.{agent_name}", agent_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, f"{agent_name}_node"):
                    loaded[agent_name] = getattr(module, f"{agent_name}_node")

        return loaded

    def _refresh(self) -> None:
        self._loaded = {}
        for entry in os.listdir(self._plugins_dir):
            entry_path = os.path.join(self._plugins_dir, entry)
            if os.path.isdir(entry_path):
                plugin_yaml = os.path.join(entry_path, "plugin.yaml")
                if os.path.exists(plugin_yaml):
                    with open(plugin_yaml) as f:
                        data = yaml.safe_load(f) or {}
                    self._loaded[entry] = PluginManifest(
                        name=data.get("name", entry),
                        version=data.get("version", "0.1.0"),
                        description=data.get("description", ""),
                        provides=data.get("provides", {}),
                    )

    def _save_manifest(self) -> None:
        data = {}
        for name, manifest in self._loaded.items():
            data[name] = {
                "name": manifest.name,
                "version": manifest.version,
                "description": manifest.description,
            }
        with open(MANIFEST_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
