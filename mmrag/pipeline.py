"""End-to-end Multimodal RAG pipeline (ingest -> embed -> store -> retrieve -> generate -> cite).

`RAGPipeline` is the single object the API, UI and evaluation scripts talk to.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import shutil
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .chunking import build_chunks, embed_text, figure_text
from .config import Settings, get_settings
from .ingestion import PDFParser
from .llm import get_chat_model
from .models import FigureDescriber, get_clip, get_reranker, get_text_embedder
from .retriever import HybridSearcher, MultimodalRetriever
from .store import DocStore, LocalIndex, PineconeIndex

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful assistant that answers questions about the user's documents.
Use ONLY the numbered sources provided. A source is a text passage, a table (Markdown), or a figure (given as a description plus the text visible inside the figure).
Rules:
- Cite every fact with its source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply: "I could not find this in the provided documents." Never guess or use outside knowledge.
- Answer directly and concisely (1-4 sentences unless the question asks for more). Quote exact names, numbers and units from the sources."""

HUMAN_PROMPT = "Sources:\n{context}\n\nQuestion: {question}\nAnswer (with citations):"

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question as a single standalone question that includes any context "
            "needed from the conversation. Output only the rewritten question.",
        ),
        ("human", "Conversation:\n{history}\n\nLatest question: {question}\n\nStandalone question:"),
    ]
)

NOT_FOUND = "I could not find this in the provided documents."
CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def parse_citations(text: str, n_sources: int) -> list[int]:
    nums: list[int] = []
    for grp in CITE_RE.findall(text):
        for x in re.split(r"[,;]", grp):
            v = int(x.strip())
            if 1 <= v <= n_sources and v not in nums:
                nums.append(v)
    return nums


def _source_label(d: Document) -> str:
    m = d.metadata
    lab = f"{m.get('source')} · page {m.get('page')} · {m.get('type')}"
    if m.get("section"):
        lab += f" · section: {m['section'][:80]}"
    return lab


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        s = self.s
        s.data_dir.mkdir(parents=True, exist_ok=True)
        dev = s.resolved_device
        self.text_emb = get_text_embedder(s.text_embed_model, dev)
        self.clip = get_clip(s.clip_model, dev) if s.use_clip else None
        self.reranker = get_reranker(s.rerank_model, dev) if s.use_rerank else None
        self.describer = FigureDescriber(s)
        self.backend = s.resolved_vector_backend

        tkey = s.index_key("text", s.text_embed_model)
        ikey = s.index_key("image", s.clip_model)
        if self.backend == "pinecone":
            self.text_index = PineconeIndex(tkey, self.text_emb.dim, s)
            self.image_index = PineconeIndex(ikey, self.clip.dim, s) if self.clip else None
        else:
            self.text_index = LocalIndex(s.data_dir / "index" / tkey, self.text_emb.dim)
            self.image_index = LocalIndex(s.data_dir / "index" / ikey, self.clip.dim) if self.clip else None
        self.docstore = DocStore(s.data_dir / "index" / f"docstore-{tkey}.json")
        self.searcher = HybridSearcher(
            s, self.text_emb, self.text_index, self.docstore, self.clip, self.image_index, self.reranker
        )
        self.llm, self.llm_desc = get_chat_model(s)
        self._ingest_lock = threading.Lock()

    # ====================================================================================
    # Ingestion
    # ====================================================================================
    def ingest(self, pdf_path: str | Path, filename: str | None = None, force: bool = False) -> dict:
        pdf_path = Path(pdf_path)
        filename = filename or pdf_path.name
        data = pdf_path.read_bytes()
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        with self._ingest_lock:
            if doc_id in self.docstore.documents and not force:
                return {**self.docstore.documents[doc_id], "skipped": True}
            if force:
                self.delete_document(doc_id)
            t0 = time.perf_counter()
            parser = PDFParser(self.s.images_dir, self.s.render_dpi, self.s.min_image_px, ocr=self.describer.ocr)
            parsed = parser.parse(pdf_path, doc_id, filename)
            t_parse = time.perf_counter()
            chunks = build_chunks(parsed, self.s.chunk_size, self.s.chunk_overlap)

            # figures -> searchable text (VLM/BLIP caption + OCR or native vector text)
            methods = set()
            for d in chunks:
                if d.metadata["type"] != "image":
                    continue
                m = d.metadata
                native = m.get("native_text", "")
                ocr = "" if len(native) > 20 else self.describer.ocr(m["image_path"])
                desc, method = self.describer.caption(m["image_path"])
                methods.add(method)
                d.page_content = figure_text(m.get("caption", ""), desc, ocr, native)
                m.update(description=desc, ocr=ocr[:1500], describe_method=method)
            t_desc = time.perf_counter()

            if not chunks:
                raise ValueError("No extractable text, tables or images were found in this PDF.")

            vecs = self.text_emb.encode_docs([embed_text(d, self.s.contextual_headers) for d in chunks])
            ids = [d.metadata["id"] for d in chunks]
            metas = [self._index_meta(d) for d in chunks]
            self.text_index.upsert(ids, vecs, metas)
            imgs = [d for d in chunks if d.metadata["type"] == "image"]
            if imgs and self.clip is not None and self.image_index is not None:
                ivecs = self.clip.embed_images([d.metadata["image_path"] for d in imgs])
                self.image_index.upsert([d.metadata["id"] for d in imgs], ivecs, [self._index_meta(d) for d in imgs])
            t_embed = time.perf_counter()

            counts = {t: sum(1 for d in chunks if d.metadata["type"] == t) for t in ("text", "table", "image")}
            info = {
                "doc_id": doc_id,
                "filename": filename,
                "title": parsed.title,
                "pages": parsed.n_pages,
                "chunks": counts,
                "warnings": parsed.warnings,
                "figure_descriptions": sorted(methods),
                "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timings": {
                    "parse_s": round(t_parse - t0, 2),
                    "describe_s": round(t_desc - t_parse, 2),
                    "embed_store_s": round(t_embed - t_desc, 2),
                    "total_s": round(t_embed - t0, 2),
                },
            }
            self.docstore.add(info, [{"text": d.page_content, "metadata": d.metadata} for d in chunks])
            return {**info, "skipped": False}

    def _index_meta(self, d: Document) -> dict:
        keep = ("doc_id", "source", "page", "type", "section", "title", "image_path", "caption")
        m = {k: d.metadata[k] for k in keep if d.metadata.get(k) not in (None, "")}
        if self.backend == "pinecone":
            m["text"] = d.page_content  # lets a fresh machine answer straight from Pinecone
        return m

    def list_documents(self) -> list[dict]:
        return sorted(self.docstore.documents.values(), key=lambda d: d.get("ingested_at", ""), reverse=True)

    def delete_document(self, doc_id: str) -> bool:
        existed = doc_id in self.docstore.documents
        self.text_index.delete_doc(doc_id)
        if self.image_index is not None:
            self.image_index.delete_doc(doc_id)
        self.docstore.delete(doc_id)
        shutil.rmtree(self.s.images_dir / doc_id, ignore_errors=True)
        return existed

    # ====================================================================================
    # Query
    # ====================================================================================
    def retriever(self, k: int | None = None, doc_ids: list[str] | None = None) -> MultimodalRetriever:
        return MultimodalRetriever(searcher=self.searcher, k=k or self.s.top_k, doc_ids=doc_ids)

    def condense(self, question: str, history: list[dict] | None) -> str:
        if not history:
            return question
        turns = history[-6:]
        if self.llm is None:
            prev = [t["content"] for t in turns if t.get("role") == "user"]
            return f"{prev[-1]} {question}" if prev and len(question.split()) < 8 else question
        hist = "\n".join(f"{t['role']}: {t['content'][:500]}" for t in turns)
        try:
            out = (CONDENSE_PROMPT | self.llm | StrOutputParser()).invoke({"history": hist, "question": question})
            out = out.strip().strip('"').splitlines()[0].strip()
            return out if 3 < len(out) < 400 else question
        except Exception as e:
            log.warning("condense failed: %s", e)
            return question

    def build_context(self, docs: list[Document]) -> tuple[str, list[Document]]:
        budget = self.s.max_context_tokens * 4  # ~4 characters per token
        parts, used = [], []
        for d in docs:
            if budget <= 200:
                break
            body = d.page_content[: min(2500, budget)]
            block = f"[{len(used) + 1}] {_source_label(d)}\n{body}"
            parts.append(block)
            used.append(d)
            budget -= len(block)
        return "\n\n".join(parts), used

    def _messages(self, context: str, question: str, docs: list[Document]):
        human = HUMAN_PROMPT.format(context=context, question=question)
        if not self.s.llm_vision:
            return [SystemMessage(SYSTEM_PROMPT), HumanMessage(human)]
        content: list[dict] = [{"type": "text", "text": human}]
        for n, d in enumerate(docs, 1):
            p = d.metadata.get("image_path")
            if d.metadata.get("type") == "image" and p and Path(p).exists():
                b64 = base64.b64encode(Path(p).read_bytes()).decode()
                content.append({"type": "text", "text": f"Image for source [{n}]:"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return [SystemMessage(SYSTEM_PROMPT), HumanMessage(content=content)]

    def _extractive(self, question: str, docs: list[Document]) -> str:
        units: list[tuple[str, int]] = []
        for n, d in enumerate(docs[:3], 1):
            t = d.metadata.get("type")
            if t == "table":
                lines = d.page_content.splitlines()
                hdr = next((l for l in lines if l.startswith("|")), "")
                units += [(f"{hdr} {l}" if l != hdr else l, n) for l in lines if l.startswith("|") and "---" not in l]
            elif t == "image":
                units += [(l, n) for l in d.page_content.splitlines() if len(l) > 3]
            else:
                units += [(u, n) for u in re.split(r"(?<=[.!?])\s+", d.page_content) if len(u) > 15]
        if not units:
            return NOT_FOUND
        units = list(dict.fromkeys(units))  # drop duplicate units (keeps order)
        if self.reranker is not None:
            sc = self.reranker.score(question, [u for u, _ in units])
        else:
            q = set(re.findall(r"\w+", question.lower()))
            sc = np.array([len(q & set(re.findall(r"\w+", u.lower()))) for u, _ in units], float)
        best = [i for i in np.argsort(-sc)[:2] if sc[i] > 0.05] or [int(np.argmax(sc))]
        return " ".join(f"{units[i][0].strip()} [{units[i][1]}]" for i in sorted(best))

    def answer_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        doc_ids: list[str] | None = None,
        k: int | None = None,
        **search_overrides,
    ) -> Iterator[dict]:
        """Yields events: {"type": "sources"}, {"type": "token"}*, {"type": "done"}."""
        t0 = time.perf_counter()
        standalone = self.condense(question, history)
        t1 = time.perf_counter()
        docs = self.searcher.search(standalone, k=k or self.s.top_k, doc_ids=doc_ids, **search_overrides)
        t2 = time.perf_counter()
        context, used = self.build_context(docs)
        sources = [self._source_dict(n, d) for n, d in enumerate(used, 1)]
        yield {"type": "sources", "sources": sources, "standalone_question": standalone}

        if not used:
            text = NOT_FOUND if self.docstore.documents else "No documents have been ingested yet."
            yield {"type": "token", "text": text}
        elif self.llm is None:
            text = self._extractive(standalone, used)
            yield {"type": "token", "text": text}
        else:
            text = ""
            for chunk in self.llm.stream(self._messages(context, standalone, used)):
                piece = chunk.content if isinstance(chunk.content, str) else ""
                if piece:
                    text += piece
                    yield {"type": "token", "text": piece}
        t3 = time.perf_counter()
        cited = parse_citations(text, len(used))
        yield {
            "type": "done",
            "answer": text.strip(),
            "cited": cited,
            "llm": self.llm_desc,
            "timings": {
                "condense_s": round(t1 - t0, 3),
                "retrieve_s": round(t2 - t1, 3),
                "generate_s": round(t3 - t2, 3),
                "total_s": round(t3 - t0, 3),
            },
        }

    def answer(self, question: str, history=None, doc_ids=None, k=None, **ov) -> dict:
        out: dict = {"question": question}
        for ev in self.answer_stream(question, history, doc_ids, k, **ov):
            if ev["type"] == "sources":
                out["sources"] = ev["sources"]
                out["standalone_question"] = ev["standalone_question"]
            elif ev["type"] == "done":
                out.update({k_: v for k_, v in ev.items() if k_ != "type"})
        for s_ in out.get("sources", []):
            s_["cited"] = s_["n"] in out.get("cited", [])
        return out

    def _source_dict(self, n: int, d: Document) -> dict:
        m = d.metadata
        img = m.get("image_path")
        return {
            "n": n,
            "id": d.id or m.get("id"),
            "doc_id": m.get("doc_id"),
            "source": m.get("source"),
            "page": m.get("page"),
            "type": m.get("type"),
            "section": m.get("section", ""),
            "content": d.page_content[:1500],
            "image": str(Path(img).relative_to(self.s.images_dir)) if img and Path(img).is_relative_to(self.s.images_dir) else None,
            "score": m.get("score"),
            "rerank_score": m.get("rerank_score"),
            "ranks": m.get("ranks", {}),
        }

    def info(self) -> dict:
        return {
            "vector_backend": self.backend,
            "llm": self.llm_desc,
            "text_embedding": self.s.text_embed_model,
            "image_embedding": self.s.clip_model if self.clip else None,
            "reranker": self.s.rerank_model if self.reranker else None,
            "figure_description": "hf-vlm" if self.describer._vlm_ok else "blip+ocr",
            "documents": len(self.docstore.documents),
            "chunks": len(self.docstore.chunks),
            "hybrid_bm25": self.s.use_bm25,
        }


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(get_settings())
