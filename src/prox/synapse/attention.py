import math
from typing import Optional


class AttentionEngine:
    def __init__(self, store):
        self._store = store

    def retrieve(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 10,
        min_strength: float = 0.1,
        associative_hop: bool = True,
    ) -> list[dict]:
        results = self._store.query(
            query_text=query,
            project_id=project_id,
            n_results=top_k * 2 if associative_hop else top_k,
        )

        scored = []
        for r in results:
            meta = r.get("metadata", {})
            strength = meta.get("strength", 1.0)
            distance = r.get("distance", 1.0)
            similarity = 1.0 - min(distance, 1.0)
            combined_score = similarity * 0.7 + strength * 0.3
            if strength >= min_strength:
                scored.append({**r, "score": combined_score, "similarity": similarity})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored[:top_k]

        if associative_hop:
            associated = self._associative_hop(top_results, project_id)
            seen_ids = {r["id"] for r in top_results}
            for a in associated:
                if a["id"] not in seen_ids:
                    top_results.append(a)
                    seen_ids.add(a["id"])

        return top_results[:top_k]

    def _associative_hop(self, results: list[dict], project_id: Optional[str] = None) -> list[dict]:
        associated = []
        for r in results:
            content = r.get("content", "")
            if content:
                related = self._store.query(
                    query_text=content,
                    project_id=project_id,
                    n_results=3,
                )
                associated.extend(related)
        return associated
