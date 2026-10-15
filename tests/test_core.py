import os
from pathlib import Path

import numpy as np
import pytest

from mmrag.chunking import build_chunks, linearize_table, tokenize
from mmrag.ingestion import PDFParser
from mmrag.pipeline import parse_citations
from mmrag.retriever import rrf
from mmrag.store import DocStore, LocalIndex

DATA = Path(__file__).resolve().parent.parent / "eval" / "data"
HELIX = DATA / "helix_r7_manual.pdf"
needs_data = pytest.mark.skipif(not HELIX.exists(), reason="run `python eval/build_dataset.py` first")


def test_rrf_rewards_agreement():
    s = rrf([(["a", "b", "c"], 1.0), (["b", "c", "a"], 1.0)], k=60)
    assert max(s, key=s.get) == "b"
    s = rrf([(["a"], 1.0), (["b"], 0.1)])
    assert s["a"] > s["b"]


def test_parse_citations():
    assert parse_citations("Payload is 7 kg [2]. Reach 912 mm [1, 3][9].", 3) == [2, 1, 3]
    assert parse_citations("no cites", 3) == []


def test_linearize_table_skips_placeholder_header():
    rows = [["Tech Stack", "Col2"], ["Component", "Technology"], ["Vector DB", "Pinecone"]]
    assert linearize_table(rows) == ["Tech Stack", "Component: Vector DB; Technology: Pinecone"]


def test_tokenize_keeps_compound_terms():
    toks = tokenize("Which model? all-MiniLM-L6-v2 and HX-OS 4.2.1")
    assert "all-minilm-l6-v2" in toks and "minilm" in toks and "4.2.1" in toks
    assert "which" not in toks


def test_local_index_roundtrip(tmp_path):
    idx = LocalIndex(tmp_path / "i", 3)
    v = np.eye(3, dtype=np.float32)
    idx.upsert(["d1#a", "d1#b", "d2#a"], v, [{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}])
    assert idx.query(np.array([0, 1, 0], np.float32), 1)[0].id == "d1#b"
    assert idx.query(np.array([0, 1, 0], np.float32), 3, doc_ids=["d2"])[0].id == "d2#a"
    idx.delete_doc("d1")
    again = LocalIndex(tmp_path / "i", 3)  # persisted
    assert again.ids == ["d2#a"]


def test_docstore(tmp_path):
    ds = DocStore(tmp_path / "ds.json")
    ds.add({"doc_id": "x"}, [{"text": "hi", "metadata": {"id": "x#1", "doc_id": "x"}}])
    assert DocStore(tmp_path / "ds.json").get("x#1")["text"] == "hi"
    ds.delete("x")
    assert not ds.chunks and not ds.documents


@needs_data
def test_parser_finds_tables_raster_and_vector_figures(tmp_path):
    d = PDFParser(tmp_path).parse(HELIX, "helix")
    kinds = [(e.page, e.type, e.origin) for e in d.elements if e.type != "text"]
    assert (1, "table", "native") in kinds and (3, "table", "native") in kinds
    assert (2, "image", "raster") in kinds and (2, "image", "vector") in kinds
    vec = next(e for e in d.elements if e.origin == "vector")
    assert "180°/s" in vec.content and vec.caption.startswith("Figure 2")
    assert not any("confidential draft" in e.content for e in d.elements)  # footer removed
    chunks = build_chunks(d)
    ids = [c.metadata["id"] for c in chunks]
    assert len(ids) == len(set(ids)) and all(i.startswith("helix#") for i in ids)
    spec = next(c for c in chunks if c.metadata["type"] == "table" and c.metadata["page"] == 1)
    assert "Parameter: Maximum payload; Value: 7 kg" in spec.metadata["table_text"]


@needs_data
@pytest.mark.skipif(os.getenv("FAST") == "1", reason="loads models")
def test_end_to_end_extractive(tmp_path):
    from mmrag.config import Settings
    from mmrag.pipeline import RAGPipeline

    s = Settings(data_dir=tmp_path, llm_backend="extractive", vector_backend="local", enable_caption=False)
    p = RAGPipeline(s)
    info = p.ingest(HELIX)
    assert info["chunks"]["table"] == 2 and info["chunks"]["image"] == 2
    assert p.ingest(HELIX)["skipped"]
    out = p.answer("What does error code E204 mean?")
    top = out["sources"][0]
    assert (top["page"], top["type"]) == (3, "table")
    assert "Encoder" in out["answer"] and out["cited"]
    assert p.delete_document(info["doc_id"]) and not p.list_documents()
