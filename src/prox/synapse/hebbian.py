import json
import os
import time
from collections import defaultdict

HEBBIAN_PATH = os.path.expanduser("~/.prox/synapse/hebbian.json")


class HebbianGraph:
    def __init__(self, path: str = HEBBIAN_PATH):
        self._path = path
        self._edges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._load()

    def reinforce(self, trace_id_a: str, trace_id_b: str, weight: float = 1.0) -> None:
        current = self._edges[trace_id_a][trace_id_b]
        self._edges[trace_id_a][trace_id_b] = current + weight
        self._edges[trace_id_b][trace_id_a] = current + weight
        self._save()

    def get_related(self, trace_id: str, min_weight: float = 2.0, top_k: int = 5) -> list[tuple[str, float]]:
        related = self._edges.get(trace_id, {})
        filtered = [(k, v) for k, v in related.items() if v >= min_weight]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]

    def decay_all(self, factor: float = 0.95) -> None:
        decayed = defaultdict(lambda: defaultdict(float))
        for src, targets in self._edges.items():
            for dst, weight in targets.items():
                new_weight = weight * factor
                if new_weight > 0.1:
                    decayed[src][dst] = new_weight
        self._edges = decayed
        self._save()

    def _save(self) -> None:
        serializable = {k: dict(v) for k, v in self._edges.items()}
        with open(self._path, "w") as f:
            json.dump(serializable, f, indent=2)

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path) as f:
                data = json.load(f)
                for src, targets in data.items():
                    for dst, weight in targets.items():
                        self._edges[src][dst] = weight
