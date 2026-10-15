"""Backend API (synopsis §8): coordinates ingestion and question answering.

    uvicorn api.main:app --port 8000

POST   /documents            upload one or more PDFs (multipart "files")
GET    /documents            list ingested documents
DELETE /documents/{doc_id}   remove a document from the index
POST   /query                answer a question (JSON)
POST   /query/stream         answer a question, streamed as NDJSON events
GET    /images/{path}        figure crops referenced by sources
GET    /health               backends and index status
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from mmrag.config import get_settings
from mmrag.pipeline import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Multimodal RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_MB = 100


class Turn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[Turn] = []
    doc_ids: list[str] | None = None
    top_k: int | None = Field(None, ge=1, le=20)


@app.on_event("startup")
def _warm():
    # Load models once at startup rather than on the first request.
    p = get_pipeline()
    log.info("pipeline ready: %s", p.info())


@app.get("/health")
def health():
    return {"status": "ok", **get_pipeline().info()}


@app.get("/documents")
def list_documents():
    return get_pipeline().list_documents()


@app.post("/documents")
async def upload(files: list[UploadFile] = File(...)):
    p = get_pipeline()
    results = []
    for f in files:
        name = Path(f.filename or "upload.pdf").name
        if not name.lower().endswith(".pdf"):
            results.append({"filename": name, "error": "only PDF files are supported"})
            continue
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(f.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            if tmp_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"file larger than {MAX_UPLOAD_MB} MB")
            info = await run_in_threadpool(p.ingest, tmp_path, name)
            results.append(info)
        except Exception as e:
            log.exception("ingestion failed for %s", name)
            results.append({"filename": name, "error": f"{type(e).__name__}: {e}"})
        finally:
            tmp_path.unlink(missing_ok=True)
    return results


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not get_pipeline().delete_document(doc_id):
        raise HTTPException(404, "document not found")
    return {"deleted": doc_id}


@app.post("/query")
async def query(req: QueryRequest):
    p = get_pipeline()
    hist = [t.model_dump() for t in req.history]
    return await run_in_threadpool(p.answer, req.question, hist, req.doc_ids, req.top_k)


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    p = get_pipeline()
    hist = [t.model_dump() for t in req.history]

    def gen():
        try:
            for ev in p.answer_stream(req.question, hist, req.doc_ids, req.top_k):
                yield json.dumps(ev) + "\n"
        except Exception as e:  # surface errors to the client instead of a dropped stream
            log.exception("query failed")
            yield json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/images/{path:path}")
def image(path: str):
    root = get_settings().images_dir.resolve()
    f = (root / path).resolve()
    if not f.is_relative_to(root) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)
