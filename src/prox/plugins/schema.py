"""Plugin system schema validation."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str = ""
    provides: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors = []
        if not self.name:
            errors.append("name e' obbligatorio")
        if not self.version:
            errors.append("version e' obbligatoria")
        return errors
