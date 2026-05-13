import os
from typing import Optional

from .age import AgeVault, VAULT_DIR
from .manifest import Manifest, is_sensitive_field


class CredentialStore:
    def __init__(self, vault_dir: str = VAULT_DIR):
        self._vault = AgeVault(vault_dir)
        self._manifest = Manifest(os.path.join(vault_dir, "manifest.yaml"))
        self._passphrase: Optional[str] = None

    @property
    def passphrase(self) -> Optional[str]:
        return self._passphrase

    @passphrase.setter
    def passphrase(self, value: str):
        self._passphrase = value
        self._vault.set_passphrase(value)

    def add(self, name: str, value: str, category: str = "general", project_id: str = "global") -> None:
        if not self._passphrase:
            raise ValueError("Passphrase non impostata. Usa store.passphrase = '...'")
        self._vault.encrypt(value, name, self._passphrase)
        self._manifest.add_credential(name, category, project_id)

    def get(self, name: str) -> str:
        if not self._passphrase:
            raise ValueError("Passphrase non impostata. Usa store.passphrase = '...'")
        return self._vault.decrypt(name, self._passphrase)

    def remove(self, name: str) -> None:
        self._vault.delete(name)
        self._manifest.remove_credential(name)

    def list_names(self) -> list[str]:
        return self._vault.list_names()

    def list_details(self) -> list[dict]:
        return self._manifest.list_credentials()

    def exists(self, name: str) -> bool:
        return self._vault.exists(name)

    def clear(self) -> None:
        self._passphrase = None
        self._vault.clear_passphrase()
