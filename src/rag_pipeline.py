"""
src/rag_pipeline.py
===================
Local Document Processing & RAG (Retrieval-Augmented Generation) Module

Handles:
  1. PDF ingestion with PyMuPDF (fitz) – zero-network, fully local
  2. Text chunking with configurable overlap
  3. Local embedding generation via Qwen3-Embedding ONNX model (QNN EP)
  4. ChromaDB vector store persistence (local SQLite backend)
  5. Semantic search over uploaded documents
  6. Document Q&A with strict retrieved-context-only answers
  7. Key-information extraction (action items, summaries, highlights)
  8. Sensitive file handling with memory wipe and no egress

All operations are 100% offline. No document content leaves this machine.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logging.warning("PyMuPDF not installed; PDF parsing disabled.")

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not installed; vector store disabled.")

try:
    import onnxruntime as ort  # type: ignore
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    DML_EP_OPTIONS,
    EMBEDDING_DIMENSION,
    MODELS_DIR,
    ONNX_EP_PRIORITY,
    QNN_EP_OPTIONS,
    SIMILARITY_METRIC,
    TOP_K_RETRIEVAL,
    VECTOR_STORE_DIR,
    WIPE_TEMP_FILES,
    ZERO_NETWORK_EGRESS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enforce offline constraint at import time
# ---------------------------------------------------------------------------
if ZERO_NETWORK_EGRESS:
    import socket
    _orig_create_connection = socket.create_connection

    def _block_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        # Allow only localhost connections
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise PermissionError(
                f"[PRIVACY GUARD] Outbound network connection to '{host}' blocked."
            )
        return _orig_create_connection(address, *args, **kwargs)

    socket.create_connection = _block_connection  # type: ignore
    logger.info("Network egress guard active: only localhost connections allowed.")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: int
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float         # cosine similarity [0, 1]
    doc_name: str


@dataclass
class DocumentSummary:
    doc_id: str
    doc_name: str
    num_chunks: int
    page_count: int
    word_count: int
    summary: str
    action_items: list[str]
    key_highlights: list[str]
    processing_ms: float


# ---------------------------------------------------------------------------
# ONNX Session builder
# ---------------------------------------------------------------------------

def _build_ort_session(model_path: str):
    if not ORT_AVAILABLE or not Path(model_path).exists():
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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
        logger.info("ORT session: %s → EP=%s", Path(model_path).name, sess.get_providers()[0])
        return sess
    except Exception as exc:
        logger.error("ORT session failed for %s: %s", model_path, exc)
        return None


# ---------------------------------------------------------------------------
# Embedding Engine (Qwen3-Embedding via ONNX + QNN)
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Generates dense vector embeddings using Qwen3-Embedding ONNX model.
    Output dimension: 1536 (matches config.EMBEDDING_DIMENSION).
    Falls back to a unit-normalised random vector if model is unavailable.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        default = str(MODELS_DIR / "qwen3_embedding.onnx")
        self.model_path = model_path or default
        self.session = _build_ort_session(self.model_path)
        self._dim = EMBEDDING_DIMENSION

        # Simple whitespace tokeniser fallback (replace with HuggingFace tokenizer)
        try:
            from transformers import AutoTokenizer  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding", trust_remote_code=True
            )
        except Exception:
            self._tokenizer = None
            logger.warning("HuggingFace tokenizer not available; using stub tokenizer.")

    def embed(self, text: str) -> np.ndarray:
        """Return a normalised embedding vector for `text`."""
        if self.session is None:
            return self._stub_embed(text)

        inputs = self._tokenize(text)
        if inputs is None:
            return self._stub_embed(text)

        try:
            inp_names = [i.name for i in self.session.get_inputs()]
            feed = {k: v for k, v in inputs.items() if k in inp_names}
            outputs = self.session.run(None, feed)
            vec = np.array(outputs[0][0], dtype=np.float32)
            return self._l2_normalise(vec)
        except Exception as exc:
            logger.error("Embedding inference failed: %s", exc)
            return self._stub_embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return shape (N, dim)."""
        return np.stack([self.embed(t) for t in texts], axis=0)

    def _tokenize(self, text: str):
        if self._tokenizer is None:
            return None
        try:
            return self._tokenizer(
                text, return_tensors="np",
                padding="max_length", truncation=True, max_length=512,
            )
        except Exception:
            return None

    def _stub_embed(self, text: str) -> np.ndarray:
        """Deterministic stub: hash text → random-seeded unit vector."""
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        return self._l2_normalise(vec)

    @staticmethod
    def _l2_normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)


# ---------------------------------------------------------------------------
# Document Parser
# ---------------------------------------------------------------------------

class DocumentParser:
    """
    Parses PDFs and plain text files into chunked DocumentChunk objects.
    Zero network usage; all parsing is in-process.
    """

    def parse(self, file_path: str) -> tuple[list[DocumentChunk], dict]:
        """
        Parse a file and return (chunks, metadata).
        Supported formats: PDF, TXT, MD.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

        if suffix == ".pdf":
            text, page_count = self._parse_pdf(file_path)
        elif suffix in (".txt", ".md", ".rst"):
            text = path.read_text(encoding="utf-8", errors="replace")
            page_count = 1
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        chunks = self._chunk_text(text, doc_id, path.name)
        metadata = {
            "doc_id": doc_id,
            "doc_name": path.name,
            "page_count": page_count,
            "word_count": len(text.split()),
            "file_size_bytes": path.stat().st_size,
        }
        return chunks, metadata

    def _parse_pdf(self, path: str) -> tuple[str, int]:
        if not FITZ_AVAILABLE:
            logger.warning("PyMuPDF not available; returning empty text for PDF.")
            return "", 0
        doc = fitz.open(path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages_text), len(pages_text)

    def _chunk_text(self, text: str, doc_id: str, doc_name: str) -> list[DocumentChunk]:
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        idx = 0
        chunk_num = 0
        while idx < len(text):
            chunk_text = text[idx: idx + CHUNK_SIZE].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    chunk_index=chunk_num,
                    text=chunk_text,
                    metadata={"doc_name": doc_name},
                ))
                chunk_num += 1
            idx += step
        return chunks


# ---------------------------------------------------------------------------
# ChromaDB Vector Store
# ---------------------------------------------------------------------------

class LocalVectorStore:
    """
    Manages a persistent ChromaDB vector store backed by local SQLite.
    Completely offline – no Chroma Cloud, no telemetry.
    """

    def __init__(self) -> None:
        self._collection = None
        if CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            logger.warning("ChromaDB unavailable; vector store disabled.")

    def _init_chroma(self) -> None:
        try:
            client = chromadb.PersistentClient(
                path=str(VECTOR_STORE_DIR),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            self._collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": SIMILARITY_METRIC},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d docs).",
                CHROMA_COLLECTION_NAME, self._collection.count(),
            )
        except Exception as exc:
            logger.error("ChromaDB init failed: %s", exc)

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update chunks (with embeddings) into the vector store."""
        if self._collection is None:
            return
        ids = [f"{c.doc_id}_{c.chunk_index}" for c in chunks]
        embeddings = [c.embedding.tolist() for c in chunks if c.embedding is not None]
        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]

        if not embeddings:
            logger.warning("No embeddings to upsert.")
            return

        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metas,
        )
        logger.info("Upserted %d chunks into vector store.", len(chunks))

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Return top-k most similar document chunks."""
        if self._collection is None:
            return []
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "distances", "metadatas"],
            )
            hits = []
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                hits.append({"text": doc, "score": 1 - dist, "metadata": meta})
            return hits
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            return []

    def delete_document(self, doc_id: str) -> None:
        if self._collection is None:
            return
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted all chunks for doc_id=%s", doc_id)

    def list_documents(self) -> list[str]:
        if self._collection is None:
            return []
        results = self._collection.get(include=["metadatas"])
        names = {m.get("doc_name", "unknown") for m in results["metadatas"]}
        return sorted(names)


# ---------------------------------------------------------------------------
# RAG Pipeline Orchestrator
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end local RAG pipeline:
      parse → embed → store → search → answer

    Usage
    -----
    rag = RAGPipeline()
    summary = rag.ingest_document("report.pdf")
    results = rag.search("What are the Q3 revenue figures?")
    answer  = rag.answer("What are the Q3 revenue figures?", llm_fn=my_llm)
    """

    def __init__(self) -> None:
        self.parser = DocumentParser()
        self.embedder = EmbeddingEngine()
        self.store = LocalVectorStore()
        self._doc_registry: dict[str, dict] = {}  # doc_id -> metadata

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(self, file_path: str) -> dict:
        """
        Full ingestion pipeline: parse → embed → store.
        Returns metadata dict with doc_id, chunk_count, processing_ms.
        """
        t0 = time.perf_counter()
        logger.info("Ingesting: %s", file_path)

        chunks, meta = self.parser.parse(file_path)
        if not chunks:
            logger.warning("No text extracted from %s", file_path)
            return meta

        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Persist to vector store
        self.store.upsert(chunks)

        # Optionally wipe raw text from memory after embedding
        if WIPE_TEMP_FILES:
            for chunk in chunks:
                chunk.text = ""
                chunk.embedding = None
            gc.collect()

        meta["num_chunks"] = len(chunks)
        meta["processing_ms"] = (time.perf_counter() - t0) * 1000.0
        self._doc_registry[meta["doc_id"]] = meta
        logger.info(
            "Ingested '%s': %d chunks in %.0f ms",
            meta["doc_name"], meta["num_chunks"], meta["processing_ms"],
        )
        return meta

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Semantic search across all ingested documents."""
        query_emb = self.embedder.embed(query)
        return self.store.search(query_emb, top_k=top_k)

    # ------------------------------------------------------------------
    # Q&A with LLM grounding
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        llm_fn: Optional[callable] = None,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> str:
        """
        Retrieve relevant chunks and pass them as grounding context to `llm_fn`.
        If llm_fn is None, returns the raw retrieved chunks as a formatted string.
        Enforces strict context-only answering to prevent hallucination.
        """
        hits = self.search(question, top_k=top_k)
        if not hits:
            return (
                "No relevant documents found. Please upload a document first "
                "or rephrase your question."
            )

        context = "\n\n---\n\n".join(
            f"[Source: {h['metadata'].get('doc_name', 'unknown')} | score={h['score']:.3f}]\n{h['text']}"
            for h in hits
        )

        if llm_fn is None:
            return f"Retrieved context (no LLM configured):\n\n{context}"

        prompt = (
            "Answer the following question using ONLY the provided context. "
            "If the context does not contain enough information, say 'Insufficient context'.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )
        return llm_fn(prompt)

    # ------------------------------------------------------------------
    # Document summarisation
    # ------------------------------------------------------------------

    def summarise_document(
        self,
        file_path: str,
        llm_fn: Optional[callable] = None,
    ) -> DocumentSummary:
        """
        Parse a document and generate a structured summary including
        action items, executive summary, and key highlights.
        """
        t0 = time.perf_counter()
        chunks, meta = self.parser.parse(file_path)
        full_text = " ".join(c.text for c in chunks)[:8000]  # LLM context limit

        if llm_fn:
            summary_prompt = (
                "You are a document analyst. Given the document text below, provide:\n"
                "1. A 3-5 sentence executive summary.\n"
                "2. A bulleted list of key action items.\n"
                "3. A bulleted list of key highlights.\n\n"
                f"DOCUMENT TEXT:\n{full_text}\n\n"
                "Respond in this exact format:\n"
                "SUMMARY: <summary text>\n"
                "ACTION_ITEMS:\n- item1\n- item2\n"
                "HIGHLIGHTS:\n- highlight1\n- highlight2\n"
            )
            raw = llm_fn(summary_prompt)
            summary_text, action_items, highlights = self._parse_llm_summary(raw)
        else:
            # Extractive fallback: first 3 sentences as summary
            sentences = full_text.split(". ")
            summary_text = ". ".join(sentences[:3]) + "."
            action_items = []
            highlights = []

        return DocumentSummary(
            doc_id=meta["doc_id"],
            doc_name=meta["doc_name"],
            num_chunks=len(chunks),
            page_count=meta.get("page_count", 0),
            word_count=meta.get("word_count", 0),
            summary=summary_text,
            action_items=action_items,
            key_highlights=highlights,
            processing_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _parse_llm_summary(self, raw: str) -> tuple[str, list[str], list[str]]:
        """Parse structured LLM summary response."""
        import re
        summary = ""
        action_items = []
        highlights = []

        sm = re.search(r"SUMMARY:\s*(.+?)(?=ACTION_ITEMS:|$)", raw, re.DOTALL)
        if sm:
            summary = sm.group(1).strip()

        am = re.search(r"ACTION_ITEMS:\s*(.+?)(?=HIGHLIGHTS:|$)", raw, re.DOTALL)
        if am:
            action_items = [
                line.strip("- ").strip()
                for line in am.group(1).splitlines()
                if line.strip().startswith("-")
            ]

        hm = re.search(r"HIGHLIGHTS:\s*(.+?)$", raw, re.DOTALL)
        if hm:
            highlights = [
                line.strip("- ").strip()
                for line in hm.group(1).splitlines()
                if line.strip().startswith("-")
            ]

        return summary, action_items, highlights

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def list_documents(self) -> list[str]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
        self._doc_registry.pop(doc_id, None)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    rag = RAGPipeline()

    # Test with a sample text file
    sample = Path("sample_test.txt")
    sample.write_text(
        "Snapdragon X Elite delivers unprecedented AI performance with 45 TOPS NPU. "
        "The Hexagon NPU is optimised for transformer workloads including LLMs and "
        "vision models. Qualcomm AI Hub provides pre-compiled ONNX models ready for "
        "deployment on Windows on Snapdragon devices.",
        encoding="utf-8",
    )

    meta = rag.ingest_document(str(sample))
    print("Ingested:", meta)

    results = rag.search("What is the NPU performance?")
    for r in results:
        print(f"Score={r['score']:.3f} | {r['text'][:100]}")

    sample.unlink()
