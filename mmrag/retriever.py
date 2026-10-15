"""Retrieval: hybrid multimodal search + fusion + cross-encoder re-ranking.

For one query, three retrievers run in parallel-by-design:

  1. dense text   - query embedded with the same text model used at ingestion (§6.4) and
                    searched against text, table and figure-description chunks;
  2. sparse BM25  - exact keyword match (model names, numbers, codes that embeddings blur);
  3. CLIP         - query embedded with CLIP's text tower and searched against figure pixels,
                    so a figure can be found even if its description is poor.

Ranked lists are merged with weighted Reciprocal Rank Fusion, then the top candidates are
re-scored by a cross-encoder. A small share of the fused score is kept as a prior so a
strong CLIP/BM25 signal is not thrown away by the re-ranker.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from .chunking import embed_text, tokenize

VISUAL_Q = re.compile(r"\b(chart|graph|plot|figure|fig\.?|diagram|image|picture|photo|illustration|drawing|flowchart|gantt|visual|shown|depicted)\b", re.I)
TABLE_Q = re.compile(r"\b(table|tabular|column|row|spec(ification)?s?)\b", re.I)


def rrf(rankings: list[tuple[list[str], float]], k: int = 60) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion: sum_w w / (k + rank)."""
    scores: dict[str, float] = {}
    for ids, w in rankings:
        for r, i in enumerate(ids):
            scores[i] = scores.get(i, 0.0) + w / (k + r + 1)
    return scores


class HybridSearcher:
    def __init__(self, settings, text_emb, text_index, docstore, clip=None, image_index=None, reranker=None):
        self.s = settings
        self.text_emb = text_emb
        self.text_index = text_index
        self.docstore = docstore
        self.clip = clip
        self.image_index = image_index
        self.reranker = reranker
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_version = -1
        self._lock = threading.Lock()

    # ---- BM25 --------------------------------------------------------------------------
    def _bm25_search(self, query: str, n: int, doc_ids) -> list[str]:
        with self._lock:
            if self._bm25_version != self.docstore.version or self._bm25 is None:
                self._bm25_ids = list(self.docstore.chunks)
                corpus = [
                    tokenize(embed_text(self._as_doc(self.docstore.chunks[i]), True)) or ["_"] for i in self._bm25_ids
                ]
                self._bm25 = BM25Okapi(corpus) if corpus else None
                self._bm25_version = self.docstore.version
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        if doc_ids:
            allowed = set(doc_ids)
            for n_, cid in enumerate(self._bm25_ids):
                if self.docstore.chunks[cid]["metadata"].get("doc_id") not in allowed:
                    scores[n_] = 0
        order = np.argsort(-scores)[:n]
        return [self._bm25_ids[i] for i in order if scores[i] > 0]

    @staticmethod
    def _as_doc(c: dict) -> Document:
        return Document(page_content=c["text"], metadata=c["metadata"])

    def _doc(self, cid: str, meta: dict | None = None) -> Document | None:
        c = self.docstore.get(cid)
        if c:
            return Document(page_content=c["text"], metadata=dict(c["metadata"]), id=cid)
        if meta and "text" in meta:  # vector backend has data the local docstore lacks
            m = {k: v for k, v in meta.items() if k != "text"}
            return Document(page_content=meta["text"], metadata=m, id=cid)
        return None

    # ---- main entry --------------------------------------------------------------------
    def search(self, query: str, k: int | None = None, doc_ids: list[str] | None = None, **ov: Any) -> list[Document]:
        g = lambda name: ov.get(name, getattr(self.s, name))  # per-call overrides (used by eval ablations)
        k = k or self.s.top_k
        n = g("candidates")
        rankings: list[tuple[list[str], float]] = []
        metas: dict[str, dict] = {}
        ranks: dict[str, dict[str, int]] = {}

        def record(name, hits_ids, w):
            rankings.append((hits_ids, w))
            for r, i in enumerate(hits_ids):
                ranks.setdefault(i, {})[name] = r + 1

        qv = self.text_emb.encode_query(query)
        dense = self.text_index.query(qv, n, doc_ids)
        for h in dense:
            metas[h.id] = h.metadata
        record("dense", [h.id for h in dense], g("w_dense"))

        if g("use_bm25"):
            record("bm25", self._bm25_search(query, n, doc_ids), g("w_bm25"))

        if g("use_clip") and self.clip is not None and self.image_index is not None and self.image_index.count():
            cv = self.clip.embed_text([query])[0]
            hits = self.image_index.query(cv, max(5, n // 3), doc_ids)
            for h in hits:
                metas.setdefault(h.id, h.metadata)
            # CLIP similarities between unrelated text/images are all ~0.2; drop the tail.
            w = g("w_clip") * (1.5 if VISUAL_Q.search(query) else 1.0)
            record("clip", [h.id for h in hits if h.score > 0.18], w)

        fused = rrf(rankings, g("rrf_k"))
        if not fused:
            return []
        cand_ids = sorted(fused, key=fused.get, reverse=True)[:n]
        docs = [d for d in (self._doc(i, metas.get(i)) for i in cand_ids) if d is not None]

        # dedupe identical content (e.g. a caption repeated as text)
        seen, uniq = set(), []
        for d in docs:
            key = re.sub(r"\W+", "", d.page_content.lower())[:400]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        docs = uniq

        max_f = max(fused[d.id] for d in docs)
        prior = np.array([fused[d.id] / max_f for d in docs])
        if g("use_rerank") and self.reranker is not None:
            ce = self.reranker.score(query, [embed_text(d, True)[:2000] for d in docs])
            final = ce + g("rerank_prior") * prior
        else:
            ce = None
            final = prior.copy()

        # light modality routing: questions that ask about a chart/table favour that type
        boost = g("modality_boost")
        if boost:
            vis, tab = bool(VISUAL_Q.search(query)), bool(TABLE_Q.search(query))
            for j, d in enumerate(docs):
                t = d.metadata.get("type")
                if (vis and t == "image") or (tab and t == "table"):
                    final[j] += boost

        order = np.argsort(-final)[:k]
        out = []
        for j in order:
            d = docs[j]
            d.metadata.update(
                score=round(float(final[j]), 4),
                rerank_score=None if ce is None else round(float(ce[j]), 4),
                ranks=ranks.get(d.id, {}),
            )
            out.append(d)
        return out


class MultimodalRetriever(BaseRetriever):
    """LangChain retriever wrapper so the searcher plugs into LCEL chains."""

    searcher: Any
    k: int = 5
    doc_ids: list[str] | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.searcher.search(query, k=self.k, doc_ids=self.doc_ids)
