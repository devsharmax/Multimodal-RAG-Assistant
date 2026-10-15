"""Central configuration, read from environment variables / a `.env` file.

Every external service is optional. When its key is absent the system falls back to a
local, free alternative so that it always runs on a laptop:

  vector store : Pinecone (PINECONE_API_KEY)  ->  local exact-search NumPy index
  LLM          : HF Inference API (HF_TOKEN)  ->  any OpenAI-compatible server (LLM_BASE_URL)
                 ->  local transformers model  ->  extractive answers (no LLM at all)
  figure text  : HF vision-language model     ->  local BLIP caption + OCR
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # ---- storage -------------------------------------------------------------------------
    data_dir: Path = ROOT / "data"

    # ---- models --------------------------------------------------------------------------
    text_embed_model: str = "BAAI/bge-small-en-v1.5"
    clip_model: str = "openai/clip-vit-base-patch32"
    rerank_model: str = "BAAI/bge-reranker-base"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    device: str = "auto"  # auto | cpu | mps | cuda

    # ---- vector database -----------------------------------------------------------------
    vector_backend: Literal["auto", "pinecone", "local"] = "auto"
    pinecone_api_key: str | None = None
    pinecone_index_prefix: str = "mmrag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "default"

    # ---- generation ----------------------------------------------------------------------
    llm_backend: Literal["auto", "hf", "openai", "local", "extractive"] = "auto"
    hf_token: str | None = None
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_base_url: str | None = None  # e.g. http://localhost:11434/v1 for Ollama
    llm_api_key: str | None = None
    local_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    # Send retrieved figures to the LLM as images (needs a vision model, e.g. Qwen2.5-VL).
    llm_vision: bool = False
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # used to describe figures at ingestion when HF_TOKEN is set
    describe_with_vlm: bool = True

    # ---- ingestion -----------------------------------------------------------------------
    chunk_size: int = 900  # characters
    chunk_overlap: int = 150
    min_image_px: int = 80
    enable_ocr: bool = True
    enable_caption: bool = True
    render_dpi: int = 150

    # ---- retrieval -----------------------------------------------------------------------
    top_k: int = 5
    candidates: int = 20
    use_bm25: bool = True
    use_clip: bool = True
    use_rerank: bool = True
    rrf_k: int = 60
    w_dense: float = 1.0
    w_bm25: float = 1.0
    w_clip: float = 1.0
    # final = reranker_prob + rerank_prior * normalised fused score (+ modality_boost when the
    # question asks about a chart/table and the chunk is one). Tuned in eval/: prior 0 is best.
    rerank_prior: float = 0.0
    modality_boost: float = 0.05
    contextual_headers: bool = True
    max_context_tokens: int = 3000

    # ---- api -----------------------------------------------------------------------------
    api_url: str = "http://localhost:8000"

    # ------------------------------------------------------------------------------------
    def model_post_init(self, __context) -> None:  # noqa: D401
        # Accept the conventional Hugging Face env var names too.
        if not self.hf_token:
            self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_API_TOKEN")
        self.data_dir = Path(self.data_dir)

    @property
    def resolved_vector_backend(self) -> str:
        if self.vector_backend != "auto":
            return self.vector_backend
        return "pinecone" if self.pinecone_api_key else "local"

    @property
    def resolved_llm_backend(self) -> str:
        if self.llm_backend != "auto":
            return self.llm_backend
        if self.llm_base_url:
            return "openai"
        if self.hf_token:
            return "hf"
        return "local"

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def index_key(self, kind: str, model: str) -> str:
        """Stable, Pinecone-safe index name tied to the embedding model (vectors from
        different models must never be mixed in one index)."""
        h = hashlib.sha1(model.encode()).hexdigest()[:8]
        return f"{self.pinecone_index_prefix}-{kind}-{h}".lower()

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
