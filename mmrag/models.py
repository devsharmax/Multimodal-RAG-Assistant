"""Multimodal Embedding Module (synopsis §6.2) plus the re-ranker and figure describer.

* TextEmbedder   - Sentence-Transformers model, exposed as a LangChain `Embeddings`.
* ClipEmbedder   - CLIP image + text towers; text queries and figures share one space,
                   which is what makes text -> image retrieval possible.
* Reranker       - cross-encoder that re-scores (query, chunk) pairs.
* FigureDescriber- turns a figure into searchable text: caption from a vision-language model
                   (HF Inference) or local BLIP, plus OCR of any text in the image.

All models load lazily and are cached per process.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import threading
from functools import lru_cache

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from PIL import Image

log = logging.getLogger(__name__)
_lock = threading.Lock()

# Models that expect an instruction prefix on the query side only.
_QUERY_PREFIX = {
    "bge-small-en": "Represent this sentence for searching relevant passages: ",
    "bge-base-en": "Represent this sentence for searching relevant passages: ",
    "bge-large-en": "Represent this sentence for searching relevant passages: ",
    "e5-": "query: ",
    "nomic-embed": "search_query: ",
}
_DOC_PREFIX = {"e5-": "passage: ", "nomic-embed": "search_document: "}


def _prefix(table: dict, name: str) -> str:
    return next((v for k, v in table.items() if k in name.lower()), "")


class TextEmbedder(Embeddings):
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.q_prefix = _prefix(_QUERY_PREFIX, model_name)
        self.d_prefix = _prefix(_DOC_PREFIX, model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        with _lock:
            return self.model.encode(
                [self.d_prefix + t for t in texts], batch_size=32, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        with _lock:
            return self.model.encode([self.q_prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(
                np.float32
            )

    # LangChain Embeddings interface
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode_docs(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.encode_query(text).tolist()


class ClipEmbedder:
    def __init__(self, model_name: str, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.name = model_name
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def embed_images(self, paths: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(paths), 16):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + 16]]
            with _lock:
                inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
                f = self.model.get_image_features(**inputs)
            out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        with _lock:
            inputs = self.processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(
                self.device
            )
            f = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()


class Reranker:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        kw = {"model_kwargs": {"dtype": torch.float16}} if device in ("mps", "cuda") else {}
        self.model = CrossEncoder(model_name, device=device, max_length=512, **kw)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        with _lock:
            s = self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        s = np.asarray(s, dtype=np.float32)
        if s.min() < 0 or s.max() > 1:  # raw logits -> probability
            s = 1 / (1 + np.exp(-s))
        return s


class FigureDescriber:
    """Produces a text description of a figure so it can be found by text search and read
    by a text-only LLM."""

    VLM_PROMPT = (
        "Describe this figure from a document for a search index. State what kind of figure it is, "
        "its title, axes, legend, every label and every number you can read, and the main trend or "
        "takeaway. Be factual and concise (max 150 words)."
    )

    def __init__(self, settings):
        self.s = settings
        self._ocr = None
        self._blip = None
        self._vlm_ok = bool(settings.hf_token and settings.describe_with_vlm)
        self._cache_path = settings.data_dir / "cache" / "figures.json"
        try:
            self._cache = json.loads(self._cache_path.read_text())
        except Exception:
            self._cache = {}

    def _cached(self, kind: str, path: str, fn):
        """Disk cache keyed by image content, so re-ingestion never re-runs OCR / captioning."""
        key = f"{kind}:{hashlib.sha1(open(path, 'rb').read()).hexdigest()}"
        if key not in self._cache:
            self._cache[key] = fn()
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))
        return self._cache[key]

    # ---- OCR ---------------------------------------------------------------------------
    def ocr(self, path: str) -> str:
        if not self.s.enable_ocr:
            return ""
        return self._cached("ocr", path, lambda: self._ocr_raw(path))

    def _ocr_raw(self, path: str) -> str:
        try:
            if self._ocr is None:
                from rapidocr_onnxruntime import RapidOCR

                self._ocr = RapidOCR()
            result, _ = self._ocr(path)
        except Exception as e:  # pragma: no cover
            log.warning("OCR failed for %s: %s", path, e)
            return ""
        if not result:
            return ""
        items = []
        for box, text, conf in result:
            if float(conf) < 0.5 or not text.strip():
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append((sum(ys) / 4, min(xs), max(ys) - min(ys), text.strip()))
        items.sort()
        # group into visual lines
        lines, cur, cur_y = [], [], None
        for y, x, h, t in items:
            if cur_y is not None and abs(y - cur_y) > max(8, 0.6 * h):
                lines.append(" ".join(t for _, t in sorted(cur)))
                cur = []
            cur.append((x, t))
            cur_y = y
        if cur:
            lines.append(" ".join(t for _, t in sorted(cur)))
        return "\n".join(lines)

    # ---- captioning --------------------------------------------------------------------
    def _caption_vlm(self, path: str) -> str:
        from huggingface_hub import InferenceClient

        img = Image.open(path).convert("RGB")
        img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        client = InferenceClient(token=self.s.hf_token, timeout=60)
        r = client.chat_completion(
            model=self.s.vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": self.VLM_PROMPT},
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.1,
        )
        return r.choices[0].message.content.strip()

    @torch.no_grad()
    def _caption_blip(self, path: str) -> str:
        if self._blip is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            dev = self.s.resolved_device
            self._blip = (
                BlipProcessor.from_pretrained(self.s.caption_model),
                BlipForConditionalGeneration.from_pretrained(self.s.caption_model).to(dev).eval(),
                dev,
            )
        proc, model, dev = self._blip
        img = Image.open(path).convert("RGB")
        with _lock:
            inputs = proc(images=img, return_tensors="pt").to(dev)
            out = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    def caption(self, path: str) -> tuple[str, str]:
        """Returns (caption, method)."""
        if not self.s.enable_caption:
            return "", "none"
        if self._vlm_ok:
            try:
                return tuple(self._cached(f"vlm:{self.s.vlm_model}", path, lambda: [self._caption_vlm(path), "vlm"]))
            except Exception as e:
                log.warning("VLM description failed (%s); falling back to BLIP", e)
                self._vlm_ok = False
        try:
            return tuple(self._cached(f"blip:{self.s.caption_model}", path, lambda: [self._caption_blip(path), "blip"]))
        except Exception as e:  # pragma: no cover
            log.warning("BLIP caption failed for %s: %s", path, e)
            return "", "none"


# ---- cached factories --------------------------------------------------------------------
@lru_cache(maxsize=4)
def get_text_embedder(name: str, device: str) -> TextEmbedder:
    return TextEmbedder(name, device)


@lru_cache(maxsize=2)
def get_clip(name: str, device: str) -> ClipEmbedder:
    return ClipEmbedder(name, device)


@lru_cache(maxsize=2)
def get_reranker(name: str, device: str) -> Reranker:
    return Reranker(name, device)
