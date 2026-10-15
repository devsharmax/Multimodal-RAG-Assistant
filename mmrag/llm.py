"""Generation Engine (synopsis §6.5): returns a LangChain chat model for the configured backend.

* hf     - Hugging Face Inference API (router.huggingface.co, OpenAI-compatible), e.g.
           meta-llama/Llama-3.1-8B-Instruct or mistralai/Mistral-7B-Instruct-v0.3. Needs HF_TOKEN.
* openai - any OpenAI-compatible server: Ollama, vLLM, LM Studio, Groq, ... (LLM_BASE_URL).
* local  - a small instruct model run in-process with transformers (no key, no network
           after the first download). Default: Qwen2.5-1.5B-Instruct.
* extractive - no LLM; the pipeline answers with the best-matching source sentences.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr

log = logging.getLogger(__name__)

HF_ROUTER = "https://router.huggingface.co/v1"


def _to_chat(messages: list[BaseMessage]) -> list[dict]:
    role = {"system": "system", "human": "user", "ai": "assistant"}
    out = []
    for m in messages:
        c = m.content
        if isinstance(c, list):  # multimodal content: keep the text parts only
            c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        out.append({"role": role.get(m.type, "user"), "content": c})
    return out


class LocalChatModel(BaseChatModel):
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cpu"
    temperature: float = 0.1
    max_new_tokens: int = 512

    _tok: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "local-transformers"

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.float32 if self.device == "cpu" else torch.float16
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()

    def _inputs(self, messages):
        self._load()
        ids = self._tok.apply_chat_template(
            _to_chat(messages), add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return {k: v.to(self.device) for k, v in ids.items()}

    def _gen_kwargs(self, stop=None):
        kw = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self._tok.eos_token_id, repetition_penalty=1.05)
        if self.temperature > 0.05:
            kw.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        else:
            kw.update(do_sample=False)
        return kw

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kw) -> ChatResult:
        text = "".join(c.message.content for c in self._stream(messages, stop, run_manager))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator[ChatGenerationChunk]:
        import torch
        from transformers import TextIteratorStreamer

        with self._lock:
            inputs = self._inputs(messages)
            streamer = TextIteratorStreamer(self._tok, skip_prompt=True, skip_special_tokens=True)

            def run():
                with torch.no_grad():
                    self._model.generate(**inputs, streamer=streamer, **self._gen_kwargs(stop))

            t = threading.Thread(target=run, daemon=True)
            t.start()
            for piece in streamer:
                if piece:
                    if run_manager:
                        run_manager.on_llm_new_token(piece)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
            t.join()


@lru_cache(maxsize=4)
def _local(model_name: str, device: str, temperature: float, max_tokens: int) -> LocalChatModel:
    return LocalChatModel(model_name=model_name, device=device, temperature=temperature, max_new_tokens=max_tokens)


def get_chat_model(s) -> tuple[BaseChatModel | None, str]:
    """Returns (chat model or None for extractive mode, human-readable description)."""
    backend = s.resolved_llm_backend
    if backend == "extractive":
        return None, "extractive (no LLM)"
    if backend in ("hf", "openai"):
        from langchain_openai import ChatOpenAI

        base = HF_ROUTER if backend == "hf" else s.llm_base_url
        key = s.hf_token if backend == "hf" else (s.llm_api_key or "not-needed")
        llm = ChatOpenAI(
            model=s.llm_model,
            base_url=base,
            api_key=key,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            timeout=120,
            max_retries=2,
        )
        return llm, f"{backend}: {s.llm_model}"
    return _local(s.local_llm_model, s.resolved_device, s.llm_temperature, s.llm_max_tokens), f"local: {s.local_llm_model}"
