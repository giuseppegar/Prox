import os
from typing import Optional

import yaml

VAULT_DIR = os.path.expanduser("~/.prox/vault")
MANIFEST_PATH = os.path.join(VAULT_DIR, "manifest.yaml")


class Manifest:
    def __init__(self, path: str = MANIFEST_PATH):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._data: dict = self._load()

    def list_credentials(self) -> list[dict]:
        return self._data.get("credentials", [])

    def get_credential(self, name: str) -> Optional[dict]:
        for cred in self._data.get("credentials", []):
            if cred.get("name") == name:
                return cred
        return None

    def add_credential(self, name: str, category: str = "general", project_id: str = "global") -> None:
        if self.get_credential(name):
            return
        self._data.setdefault("credentials", []).append({
            "name": name,
            "category": category,
            "project_id": project_id,
        })
        self._save()

    def remove_credential(self, name: str) -> None:
        self._data["credentials"] = [
            c for c in self._data.get("credentials", []) if c.get("name") != name
        ]
        self._save()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)


SENSITIVE_FIELD_NAMES = {
    "password", "pass", "pwd", "secret", "token", "key",
    "api_key", "api", "private_key", "passphrase", "pin",
}


def is_sensitive_field(name: str) -> bool:
    normalized = name.lower().replace("_", "").replace("-", "")
    return normalized in {n.lower().replace("_", "").replace("-", "") for n in SENSITIVE_FIELD_NAMES}
