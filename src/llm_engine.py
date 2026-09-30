"""
src/llm_engine.py
=================
On-Device Large Language Model Inference Engine

Handles:
  1. Llama-3.2-3B-Instruct (ONNX, QNN EP) – chat & content generation
  2. Streaming token generation with TTFT measurement
  3. NPU -> GPU -> CPU execution provider cascade
  4. Chat history management (sliding window)
  5. Content generation tasks: email drafting, rephrasing, summarisation
  6. Privacy-safe system prompt enforcement

All inference is 100% local. No tokens are sent to external APIs.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

# HuggingFace tokenizer for Llama (offline, local weights only)
try:
    from transformers import AutoTokenizer  # type: ignore
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("transformers not installed; using stub tokenizer.")

from config.settings import (
    AIHUB_MODELS,
    DML_EP_OPTIONS,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_NEW_TOKENS,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    MODELS_DIR,
    ONNX_EP_PRIORITY,
    QNN_EP_OPTIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str     # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    num_tokens: int
    ttft_ms: float          # Time-To-First-Token
    total_ms: float
    tokens_per_second: float
    provider: str


@dataclass
class ContentTask:
    """Describes a content generation task."""
    task_type: str          # "draft_email" | "rephrase" | "summarise" | "extract"
    input_text: str
    extra_instructions: str = ""


# ---------------------------------------------------------------------------
# ONNX Session builder
# ---------------------------------------------------------------------------

def _build_ort_session(model_path: str, provider_hint: Optional[str] = None):
    if not ORT_AVAILABLE or not Path(model_path).exists():
        return None, "unavailable"

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    eps = []
    for ep in ONNX_EP_PRIORITY:
        if ep == "QNNExecutionProvider":
            eps.append((ep, QNN_EP_OPTIONS))
        elif ep == "DmlExecutionProvider":
            eps.append((ep, DML_EP_OPTIONS))
        else:
            eps.append(ep)

    try:
        sess = ort.InferenceSession(model_path, sess_options=opts, providers=eps)
        active_ep = sess.get_providers()[0]
        logger.info("LLM ORT session: %s → EP=%s", Path(model_path).name, active_ep)
        return sess, active_ep
    except Exception as exc:
        logger.error("Failed to load LLM model %s: %s", model_path, exc)
        return None, "unavailable"


# ---------------------------------------------------------------------------
# Stub Tokenizer (when HF transformers is not installed)
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Minimal stand-in tokenizer using whitespace splitting."""

    eos_token_id: int = 2
    bos_token_id: int = 1

    def encode(self, text: str, **_) -> list[int]:
        # Assign deterministic IDs based on word hash
        return [abs(hash(w)) % 32000 for w in text.split()]

    def decode(self, ids: list[int], **_) -> str:
        return f"[stub decode: {len(ids)} tokens]"

    def apply_chat_template(self, messages: list[dict], **_) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


# ---------------------------------------------------------------------------
# Sampler (temperature + top-p nucleus sampling)
# ---------------------------------------------------------------------------

class _Sampler:
    """Applies temperature and top-p sampling to raw logits."""

    @staticmethod
    def sample(logits: np.ndarray, temperature: float, top_p: float) -> int:
        if temperature == 0:
            return int(np.argmax(logits))

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Softmax
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        # Top-p nucleus filtering
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, top_p)
        nucleus = sorted_idx[: cutoff + 1]

        # Sample from nucleus
        nucleus_probs = probs[nucleus]
        nucleus_probs /= nucleus_probs.sum()
        return int(np.random.choice(nucleus, p=nucleus_probs))


# ---------------------------------------------------------------------------
# LLM Engine
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Local Llama-3.2-3B-Instruct inference engine via ONNX Runtime + QNN EP.

    Usage
    -----
    llm = LLMEngine()

    # Single-turn generation
    result = llm.generate("Summarise quantum computing in 3 sentences.")
    print(result.text)

    # Streaming
    for token in llm.generate_stream("Write a professional email to…"):
        print(token, end="", flush=True)

    # Chat (multi-turn)
    llm.chat("Hello, what can you do?")
    llm.chat("Draft an email about Q3 results.")
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        default_dir = str(MODELS_DIR / "llama-3.2-3b-instruct")
        self.model_dir = Path(model_dir or default_dir)

        # Try to find ONNX files for encoder / decoder-with-kv-cache
        self._encoder_path = str(self.model_dir / "model_encoder.onnx")
        self._decoder_path = str(self.model_dir / "model_decoder_kv.onnx")
        self._tokenizer_path = str(self.model_dir)

        self._encoder, self._active_ep = _build_ort_session(self._encoder_path)
        self._decoder, _ = _build_ort_session(self._decoder_path)

        # Load tokenizer
        if HF_AVAILABLE:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer_path, local_files_only=True
                )
                logger.info("Llama tokenizer loaded from %s", self._tokenizer_path)
            except Exception:
                logger.warning("Could not load HF tokenizer; using stub.")
                self._tokenizer = _StubTokenizer()
        else:
            self._tokenizer = _StubTokenizer()

        self._history: list[ChatMessage] = []
        self._sampler = _Sampler()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> str:
        return self._active_ep

    @property
    def is_available(self) -> bool:
        return self._encoder is not None or self._decoder is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        system_prompt: str = LLM_SYSTEM_PROMPT,
    ) -> GenerationResult:
        """
        Non-streaming generation. Returns complete GenerationResult.
        Thread-safe via internal lock.
        """
        with self._lock:
            return self._generate_impl(
                prompt, max_new_tokens, temperature, top_p, system_prompt
            )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        system_prompt: str = LLM_SYSTEM_PROMPT,
    ) -> Generator[str, None, None]:
        """
        Streaming generator that yields decoded token strings one at a time.
        Use this for live UI updates and TTFT measurement.
        """
        with self._lock:
            yield from self._stream_impl(
                prompt, max_new_tokens, temperature, top_p, system_prompt
            )

    def chat(
        self,
        user_message: str,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Multi-turn conversational interface.
        Maintains sliding-window history within LLM_CONTEXT_WINDOW.
        """
        self._history.append(ChatMessage(role="user", content=user_message))
        prompt = self._build_chat_prompt()

        full_text = ""
        if stream and on_token:
            for token in self.generate_stream(prompt):
                on_token(token)
                full_text += token
        else:
            result = self.generate(prompt)
            full_text = result.text

        self._history.append(ChatMessage(role="assistant", content=full_text))
        self._trim_history()
        return full_text

    def reset_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Content generation tasks
    # ------------------------------------------------------------------

    def run_content_task(self, task: ContentTask) -> str:
        """
        Execute a structured content task with a specialised prompt template.
        """
        prompt = self._build_task_prompt(task)
        result = self.generate(prompt, temperature=0.5)
        return result.text

    def _build_task_prompt(self, task: ContentTask) -> str:
        templates = {
            "draft_email": (
                "Draft a professional, concise email based on the following context. "
                "Include a subject line, greeting, body, and sign-off.\n\n"
                f"CONTEXT:\n{task.input_text}\n\n"
                f"ADDITIONAL INSTRUCTIONS: {task.extra_instructions}\n\n"
                "EMAIL:"
            ),
            "rephrase": (
                "Rephrase the following text to be clearer, more professional, "
                "and concise while preserving all original meaning.\n\n"
                f"ORIGINAL:\n{task.input_text}\n\n"
                f"STYLE GUIDANCE: {task.extra_instructions}\n\n"
                "REPHRASED:"
            ),
            "summarise": (
                "Summarise the following text into 3-5 bullet points capturing "
                "the most important information.\n\n"
                f"TEXT:\n{task.input_text}\n\n"
                "SUMMARY:"
            ),
            "extract": (
                "Extract all action items and key decisions from the following text. "
                "Format as numbered list.\n\n"
                f"TEXT:\n{task.input_text}\n\n"
                "ACTION ITEMS:"
            ),
        }
        return templates.get(task.task_type, task.input_text)

    # ------------------------------------------------------------------
    # Internal generation logic
    # ------------------------------------------------------------------

    def _generate_impl(
        self, prompt: str, max_new_tokens: int,
        temperature: float, top_p: float, system_prompt: str,
    ) -> GenerationResult:
        if not self.is_available:
            return GenerationResult(
                text=(
                    "[LLM model not loaded – place ONNX files in "
                    f"{self.model_dir}. Download via: "
                    "python scripts/download_models.py --model llama3-3b]"
                ),
                num_tokens=0, ttft_ms=0, total_ms=0,
                tokens_per_second=0, provider=self._active_ep,
            )

        tokens = []
        ttft_ms = 0.0
        t_start = time.perf_counter()

        for i, token_str in enumerate(
            self._autoregress(prompt, max_new_tokens, temperature, top_p, system_prompt)
        ):
            if i == 0:
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
            tokens.append(token_str)
            if len(tokens) >= max_new_tokens:
                break

        total_ms = (time.perf_counter() - t_start) * 1000.0
        text = "".join(tokens)
        n = len(tokens)
        return GenerationResult(
            text=text, num_tokens=n,
            ttft_ms=ttft_ms, total_ms=total_ms,
            tokens_per_second=(n / (total_ms / 1000.0)) if total_ms > 0 else 0,
            provider=self._active_ep,
        )

    def _stream_impl(
        self, prompt: str, max_new_tokens: int,
        temperature: float, top_p: float, system_prompt: str,
    ) -> Iterator[str]:
        if not self.is_available:
            yield "[LLM unavailable – see logs for setup instructions]"
            return
        for token_str in self._autoregress(
            prompt, max_new_tokens, temperature, top_p, system_prompt
        ):
            yield token_str

    def _autoregress(
        self, prompt: str, max_new_tokens: int,
        temperature: float, top_p: float, system_prompt: str,
    ) -> Iterator[str]:
        """
        Core autoregressive generation loop.
        This is a skeleton implementation; wire in full KV-cache ONNX
        decoder sessions from the Qualcomm AI Hub Llama model package.
        """
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}\n\n[ASSISTANT]\n"
        input_ids = self._tokenizer.encode(full_prompt, add_special_tokens=True)

        past_key_values = None
        generated_ids = []

        for step in range(max_new_tokens):
            input_tensor = np.array([input_ids if step == 0 else [generated_ids[-1]]],
                                    dtype=np.int64)
            attention_mask = np.ones_like(input_tensor, dtype=np.int64)

            try:
                feed = {
                    "input_ids": input_tensor,
                    "attention_mask": attention_mask,
                }
                if past_key_values is not None:
                    for k, v in past_key_values.items():
                        feed[k] = v

                outputs = self._decoder.run(None, feed)
                logits = outputs[0][0, -1, :]  # (vocab_size,)

                # Cache KV outputs (model-specific naming)
                # past_key_values = {f"past_key_values.{i}": outputs[i+1] ...}

                next_token = self._sampler.sample(logits, temperature, top_p)
                generated_ids.append(next_token)

                token_str = self._tokenizer.decode([next_token], skip_special_tokens=True)
                yield token_str

                if next_token == self._tokenizer.eos_token_id:
                    break

            except Exception as exc:
                logger.error("Autoregression step %d failed: %s", step, exc)
                yield "[generation error]"
                break

        gc.collect()

    # ------------------------------------------------------------------
    # Chat history management
    # ------------------------------------------------------------------

    def _build_chat_prompt(self) -> str:
        messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def _trim_history(self) -> None:
        """Keep history within context window by dropping oldest messages."""
        total_chars = sum(len(m.content) for m in self._history)
        while total_chars > LLM_CONTEXT_WINDOW * 3 and len(self._history) > 2:
            removed = self._history.pop(0)
            total_chars -= len(removed.content)


# ---------------------------------------------------------------------------
# Convenience callable for RAG pipeline integration
# ---------------------------------------------------------------------------

def make_llm_callable(engine: LLMEngine) -> Callable[[str], str]:
    """
    Returns a simple str->str callable wrapping `engine.generate`.
    Compatible with RAGPipeline.answer() and summarise_document().
    """
    def _call(prompt: str) -> str:
        result = engine.generate(prompt)
        return result.text
    return _call


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = LLMEngine()

    print("Provider:", llm.active_provider)
    print("Available:", llm.is_available)

    # Content task test (works even without model in stub mode)
    from src.llm_engine import ContentTask
    task = ContentTask(
        task_type="draft_email",
        input_text="Meeting rescheduled from Monday to Wednesday at 3pm.",
        extra_instructions="Keep it under 100 words.",
    )
    prompt_preview = llm._build_task_prompt(task)
    print("\nPrompt preview:\n", prompt_preview[:300])
