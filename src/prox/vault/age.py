import os
from typing import Optional

import pyrage

VAULT_DIR = os.path.expanduser("~/.prox/vault")
PASSPHRASE_ENV = "PROX_PASSPHRASE"


class AgeVault:
    def __init__(self, vault_dir: str = VAULT_DIR):
        self._vault_dir = vault_dir
        os.makedirs(vault_dir, exist_ok=True)

    def encrypt(self, value: str, name: str, passphrase: str) -> str:
        path = os.path.join(self._vault_dir, f"{name}.age")
        recipients = pyrage.passphrase.PassphraseRecipient(passphrase)
        encrypted = pyrage.encrypt(value.encode(), recipients=[recipients])
        with open(path, "wb") as f:
            f.write(encrypted)
        return path

    def decrypt(self, name: str, passphrase: str) -> str:
        path = os.path.join(self._vault_dir, f"{name}.age")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Credenziale '{name}' non trovata in {path}")

        identity = pyrage.passphrase.PassphraseIdentity(passphrase)
        with open(path, "rb") as f:
            encrypted_data = f.read()

        decrypted = pyrage.decrypt(encrypted_data, identities=[identity])
        return decrypted.decode()

    def delete(self, name: str) -> None:
        path = os.path.join(self._vault_dir, f"{name}.age")
        if os.path.exists(path):
            os.remove(path)

    def exists(self, name: str) -> bool:
        path = os.path.join(self._vault_dir, f"{name}.age")
        return os.path.exists(path)

    def list_names(self) -> list[str]:
        result = []
        if os.path.exists(self._vault_dir):
            for f in os.listdir(self._vault_dir):
                if f.endswith(".age"):
                    result.append(f[:-4])
        return result

    def set_passphrase(self, passphrase: str) -> None:
        os.environ[PASSPHRASE_ENV] = passphrase

    def get_passphrase(self) -> Optional[str]:
        return os.environ.get(PASSPHRASE_ENV)

    def clear_passphrase(self) -> None:
        os.environ.pop(PASSPHRASE_ENV, None)
