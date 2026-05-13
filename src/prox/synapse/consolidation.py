import time
from typing import Optional


class ConsolidationLoop:
    def __init__(self, store, hebbian_graph=None):
        self._store = store
        self._hebbian = hebbian_graph

    def consolidate(self, project_id: Optional[str] = None) -> int:
        traces = self._store.list_all(project_id=project_id)
        if len(traces) < 5:
            return 0

        now = time.time()
        recent = [t for t in traces if (now - t.get("metadata", {}).get("created_at", now)) < 86400]
        if len(recent) < 3:
            return 0

        contents = [t.get("content", "") for t in recent]
        abstract = self._abstract(contents)

        if abstract:
            from ..synapse.store import MemoryTrace
            from ..synapse import NeuralStore

            trace = MemoryTrace(
                content=abstract,
                project_id=project_id or "global",
                trace_type="abstract",
                metadata={
                    "source_count": len(recent),
                    "consolidated_at": now,
                },
            )
            self._store.add(trace)
            return 1
        return 0

    def _abstract(self, contents: list[str]) -> str:
        if not contents:
            return ""
        unique = list(set(contents))
        if len(unique) == 1:
            return unique[0]
        return " | ".join(unique[:5])
