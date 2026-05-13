import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings

SYNAPSE_DIR = os.path.expanduser("~/.prox/synapse")
CHROMA_PATH = os.path.join(SYNAPSE_DIR, "chroma")


@dataclass
class MemoryTrace:
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    project_id: str = "global"
    trace_type: str = "general"
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = 0.0
    access_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "project_id": self.project_id,
            "trace_type": self.trace_type,
            "strength": self.strength,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }


class NeuralStore:
    def __init__(self, persist_dir: str = CHROMA_PATH):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self._client.get_collection("synapse_traces")
        except Exception:
            return self._client.create_collection(
                name="synapse_traces",
                metadata={"hnsw:space": "cosine"},
            )

    def add(self, trace: MemoryTrace) -> str:
        self._collection.add(
            ids=[trace.id],
            documents=[trace.content],
            metadatas=[trace.to_dict()],
            embeddings=[trace.embedding] if trace.embedding else None,
        )
        return trace.id

    def update(self, trace: MemoryTrace) -> None:
        self._collection.update(
            ids=[trace.id],
            documents=[trace.content],
            metadatas=[trace.to_dict()],
        )

    def delete(self, trace_id: str) -> None:
        self._collection.delete(ids=[trace_id])

    def query(
        self,
        query_text: str,
        project_id: Optional[str] = None,
        trace_type: Optional[str] = None,
        n_results: int = 10,
    ) -> list[dict]:
        where = {}
        if project_id:
            where["project_id"] = project_id
        if trace_type:
            where["trace_type"] = trace_type

        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where if where else None,
        )

        traces = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                traces.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return traces

    def get_by_id(self, trace_id: str) -> Optional[dict]:
        result = self._collection.get(ids=[trace_id])
        if result and result["ids"]:
            return {
                "id": result["ids"][0],
                "content": result["documents"][0] if result["documents"] else "",
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
            }
        return None

    def list_all(self, project_id: Optional[str] = None) -> list[dict]:
        where = {"project_id": project_id} if project_id else None
        result = self._collection.get(where=where)
        traces = []
        if result and result["ids"]:
            for i in range(len(result["ids"])):
                traces.append({
                    "id": result["ids"][i],
                    "content": result["documents"][i] if result["documents"] else "",
                    "metadata": result["metadatas"][i] if result["metadatas"] else {},
                })
        return traces

    def count(self) -> int:
        return self._collection.count()
