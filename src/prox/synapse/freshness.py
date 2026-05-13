import os
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml

from prox.synapse.store import NeuralStore, MemoryTrace

FRESHNESS_CONFIG_PATH = os.path.expanduser("~/.prox/synapse/freshness.yaml")
DEFAULT_TTL_DAYS = 7


@dataclass
class PackageEntry:
    name: str
    version: str = "unknown"
    last_checked: float = 0.0
    ttl_days: int = DEFAULT_TTL_DAYS
    breaking_changes: list[str] = field(default_factory=list)
    deprecations: list[str] = field(default_factory=list)
    cve_ids: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    notes: str = ""


class FreshnessLayer:
    def __init__(self, store: Optional[NeuralStore] = None, config_path: str = FRESHNESS_CONFIG_PATH):
        self._store = store or NeuralStore()
        self._config_path = config_path
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        self._packages: dict[str, PackageEntry] = {}
        self._load()

    def is_fresh(self, package_name: str) -> bool:
        entry = self._packages.get(package_name)
        if not entry:
            return False
        age_days = (time.time() - entry.last_checked) / 86400.0
        return age_days < entry.ttl_days

    def get_package(self, package_name: str) -> Optional[PackageEntry]:
        return self._packages.get(package_name)

    def update_package(self, entry: PackageEntry) -> None:
        self._packages[entry.name] = entry
        entry.last_checked = time.time()
        self._save()

    def needs_refresh(self, package_name: str) -> bool:
        return not self.is_fresh(package_name)

    def list_stale(self) -> list[str]:
        return [name for name in self._packages if self.needs_refresh(name)]

    def pre_task_check(self, task_description: str) -> dict:
        import re
        found_packages = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_-]+)\b', task_description)
        warnings = []
        stale = []

        for pkg in found_packages:
            if pkg in self._packages:
                entry = self._packages[pkg]
                if self.needs_refresh(pkg):
                    stale.append(pkg)
                if entry.breaking_changes:
                    warnings.append(f"BREAKING: {pkg}@{entry.version} ha breaking changes: {', '.join(entry.breaking_changes)}")
                if entry.cve_ids:
                    warnings.append(f"SECURITY: {pkg}@{entry.version} ha CVE: {', '.join(entry.cve_ids)}")

        return {
            "packages_found": found_packages,
            "stale": stale,
            "warnings": warnings,
        }

    def record_from_search(self, package_name: str, version: str, notes: str = "",
                           breaking: list = None, deprecations: list = None,
                           cve: list = None, alternatives: list = None) -> None:
        entry = PackageEntry(
            name=package_name,
            version=version,
            last_checked=time.time(),
            ttl_days=DEFAULT_TTL_DAYS,
            breaking_changes=breaking or [],
            deprecations=deprecations or [],
            cve_ids=cve or [],
            alternatives=alternatives or [],
            notes=notes,
        )
        self._packages[package_name] = entry

        trace = MemoryTrace(
            content=f"{package_name}@{version}: {notes}",
            trace_type="freshness",
            metadata={
                "package": package_name,
                "version": version,
                "last_checked": time.time(),
                "breaking_changes": breaking or [],
                "cve_ids": cve or [],
            },
        )
        self._store.add(trace)
        self._save()

    def _save(self) -> None:
        data = {}
        for name, entry in self._packages.items():
            data[name] = {
                "version": entry.version,
                "last_checked": entry.last_checked,
                "ttl_days": entry.ttl_days,
                "breaking_changes": entry.breaking_changes,
                "deprecations": entry.deprecations,
                "cve_ids": entry.cve_ids,
                "alternatives": entry.alternatives,
                "notes": entry.notes,
            }
        with open(self._config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _load(self) -> None:
        if os.path.exists(self._config_path):
            with open(self._config_path) as f:
                data = yaml.safe_load(f) or {}
                for name, info in data.items():
                    self._packages[name] = PackageEntry(
                        name=name,
                        version=info.get("version", "unknown"),
                        last_checked=info.get("last_checked", 0.0),
                        ttl_days=info.get("ttl_days", DEFAULT_TTL_DAYS),
                        breaking_changes=info.get("breaking_changes", []),
                        deprecations=info.get("deprecations", []),
                        cve_ids=info.get("cve_ids", []),
                        alternatives=info.get("alternatives", []),
                        notes=info.get("notes", ""),
                    )
