"""Vector Storage module (synopsis §6.3).

Two interchangeable backends behind one small interface:

* PineconeIndex - managed serverless Pinecone index (created on first use).
* LocalIndex    - exact cosine search over a NumPy matrix persisted to disk. For the
                  small/medium corpora this project targets, exact search is both faster to
                  set up and strictly more accurate than ANN.

Text chunks and figures live in separate indexes because the text model and CLIP produce
vectors of different sizes/spaces. A local JSON docstore keeps every chunk's text so BM25
keyword search and source display work regardless of the vector backend.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict


class LocalIndex:
    def __init__(self, path: Path, dim: int):
        self.path = Path(path)
        self.dim = dim
        self._lock = threading.RLock()
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.vecs = np.zeros((0, dim), np.float32)
        if (self.path / "vectors.npy").exists():
            self.vecs = np.load(self.path / "vectors.npy")
            data = json.loads((self.path / "meta.json").read_text())
            self.ids, self.metas = data["ids"], data["metas"]

    def _save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self.vecs)
        (self.path / "meta.json").write_text(json.dumps({"ids": self.ids, "metas": self.metas}))

    def upsert(self, ids: list[str], vecs: np.ndarray, metas: list[dict]):
        with self._lock:
            pos = {i: n for n, i in enumerate(self.ids)}
            new_rows = []
            for i, v, m in zip(ids, vecs, metas):
                if i in pos:
                    self.vecs[pos[i]] = v
                    self.metas[pos[i]] = m
                else:
                    self.ids.append(i)
                    self.metas.append(m)
                    new_rows.append(v)
            if new_rows:
                self.vecs = np.vstack([self.vecs, np.asarray(new_rows, np.float32)])
            self._save()

    def query(self, vec: np.ndarray, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        with self._lock:
            if not self.ids:
                return []
            scores = self.vecs @ vec.astype(np.float32)
            if doc_ids:
                allowed = set(doc_ids)
                mask = np.array([m.get("doc_id") in allowed for m in self.metas])
                scores = np.where(mask, scores, -np.inf)
            k = min(k, len(self.ids))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [Hit(self.ids[i], float(scores[i]), self.metas[i]) for i in top if np.isfinite(scores[i])]

    def delete_doc(self, doc_id: str):
        with self._lock:
            keep = [n for n, m in enumerate(self.metas) if m.get("doc_id") != doc_id]
            self.ids = [self.ids[n] for n in keep]
            self.metas = [self.metas[n] for n in keep]
            self.vecs = self.vecs[keep] if keep else np.zeros((0, self.dim), np.float32)
            self._save()

    def count(self) -> int:
        return len(self.ids)


class PineconeIndex:
    """Pinecone serverless index. Metadata values are flattened to Pinecone-legal types."""

    def __init__(self, name: str, dim: int, settings):
        from pinecone import Pinecone, ServerlessSpec

        self.name, self.dim, self.ns = name, dim, settings.pinecone_namespace
        pc = Pinecone(api_key=settings.pinecone_api_key)
        if not pc.has_index(name):
            log.info("creating Pinecone index %s (dim=%d)", name, dim)
            pc.create_index(
                name=name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
            for _ in range(60):
                if pc.describe_index(name).status.get("ready"):
                    break
                time.sleep(1)
        self.index = pc.Index(name)

    @staticmethod
    def _clean(meta: dict) -> dict:
        out = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v[:30000] if isinstance(v, str) else v
            elif isinstance(v, (list, tuple)):
                out[k] = [str(x) for x in v]
            else:
                out[k] = str(v)
        return out

    def upsert(self, ids, vecs, metas):
        batch = [{"id": i, "values": v.tolist(), "metadata": self._clean(m)} for i, v, m in zip(ids, vecs, metas)]
        for s in range(0, len(batch), 100):
            self.index.upsert(vectors=batch[s : s + 100], namespace=self.ns)

    def query(self, vec, k, doc_ids=None) -> list[Hit]:
        flt = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
        r = self.index.query(
            vector=vec.tolist(), top_k=k, include_metadata=True, filter=flt, namespace=self.ns
        )
        return [Hit(m["id"], float(m["score"]), dict(m.get("metadata") or {})) for m in r["matches"]]

    def delete_doc(self, doc_id: str):
        # Serverless indexes delete by id; all chunk ids start with "<doc_id>#".
        for ids in self.index.list(prefix=f"{doc_id}#", namespace=self.ns):
            if ids:
                self.index.delete(ids=list(ids), namespace=self.ns)

    def count(self) -> int:
        try:
            st = self.index.describe_index_stats()
            return int(st["namespaces"].get(self.ns, {}).get("vector_count", 0))
        except Exception:
            return -1


class DocStore:
    """Chunk text + metadata and the registry of ingested documents (local JSON)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.chunks: dict[str, dict] = {}
        self.documents: dict[str, dict] = {}
        self.version = 0
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.chunks, self.documents = data.get("chunks", {}), data.get("documents", {})

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"chunks": self.chunks, "documents": self.documents}))
        tmp.replace(self.path)
        self.version += 1

    def add(self, doc_info: dict, chunks: list[dict]):
        with self._lock:
            self.documents[doc_info["doc_id"]] = doc_info
            for c in chunks:
                self.chunks[c["metadata"]["id"]] = c
            self._save()

    def delete(self, doc_id: str):
        with self._lock:
            self.documents.pop(doc_id, None)
            self.chunks = {k: v for k, v in self.chunks.items() if v["metadata"].get("doc_id") != doc_id}
            self._save()

    def get(self, cid: str) -> dict | None:
        return self.chunks.get(cid)
