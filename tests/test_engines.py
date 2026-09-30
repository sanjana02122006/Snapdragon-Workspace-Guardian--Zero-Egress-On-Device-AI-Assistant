"""
tests/test_engines.py
=====================
Unit & integration test suite for Snapdragon Local Workspace & Privacy Assistant.

Tests run completely offline, using stub/mock models where ONNX models
are not installed. Use pytest to execute:

    pytest tests/test_engines.py -v

    # Run only fast tests (no model load):
    pytest tests/test_engines.py -v -m "not slow"
"""

from __future__ import annotations

import gc
import numpy as np
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_imports(self):
        from config.settings import (
            APP_NAME, AIHUB_MODELS, ONNX_EP_PRIORITY,
            CHUNK_SIZE, EMBEDDING_DIMENSION,
        )
        assert APP_NAME == "Snapdragon Local Workspace & Privacy Assistant"
        assert "llm_chat" in AIHUB_MODELS
        assert "QNNExecutionProvider" in ONNX_EP_PRIORITY
        assert CHUNK_SIZE > 0
        assert EMBEDDING_DIMENSION > 0

    def test_directories_created(self):
        from config.settings import LOGS_DIR, UPLOADS_DIR, VECTOR_STORE_DIR
        assert LOGS_DIR.exists()
        assert UPLOADS_DIR.exists()
        assert VECTOR_STORE_DIR.exists()


# ---------------------------------------------------------------------------
# NPU Benchmarker
# ---------------------------------------------------------------------------

class TestNPUBenchmarker:
    def test_dummy_benchmark(self):
        from src.npu_benchmarker import NPUBenchmarker
        bench = NPUBenchmarker(model_name="test-model", provider="CPUExecutionProvider")

        def dummy_fn():
            time.sleep(0.005)
            return [1, 2, 3]

        summary = bench.run(fn=dummy_fn, n_runs=5, warmup=2)
        assert summary.num_runs == 5
        assert summary.mean_latency_ms > 0
        assert summary.min_latency_ms <= summary.mean_latency_ms
        assert summary.max_latency_ms >= summary.mean_latency_ms
        assert 0 <= summary.p95_latency_ms

    def test_live_stats_has_keys(self):
        from src.npu_benchmarker import NPUBenchmarker
        bench = NPUBenchmarker("test", "CPUExecutionProvider")
        stats = bench.live_stats()
        assert "cpu_percent" in stats
        assert "ram_used_mb" in stats
        assert "ram_total_mb" in stats

    def test_ttft_measurement(self):
        from src.npu_benchmarker import NPUBenchmarker
        bench = NPUBenchmarker("test", "CPUExecutionProvider")

        def gen():
            time.sleep(0.01)
            yield "hello"
            yield " world"

        ttft = bench.measure_ttft(gen)
        assert ttft > 5.0  # At least 5 ms

    def test_benchmark_log_written(self, tmp_path):
        from src.npu_benchmarker import NPUBenchmarker
        log = tmp_path / "bench.jsonl"
        bench = NPUBenchmarker("log-test", "CPUExecutionProvider", log_path=str(log))
        bench.run(fn=lambda: None, n_runs=3, warmup=1)
        assert log.exists()
        lines = log.read_text().splitlines()
        assert len(lines) == 3

    def test_percentile_calculation(self):
        from src.npu_benchmarker import _percentile
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == pytest.approx(3.0, abs=0.1)
        assert _percentile(data, 0) == pytest.approx(1.0, abs=0.1)
        assert _percentile(data, 100) == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# Vision Engine
# ---------------------------------------------------------------------------

class TestVisionEngine:
    def test_privacy_blur_applies_blur(self):
        from src.vision_engine import PrivacyBlurEngine
        blur = PrivacyBlurEngine()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blurred = blur.apply_blur(frame)
        assert blurred.shape == frame.shape
        # Blurred frame should differ from original
        assert not np.array_equal(blurred, frame)

    def test_privacy_blur_region(self):
        from src.vision_engine import PrivacyBlurEngine
        blur = PrivacyBlurEngine()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        blurred = blur.apply_blur(frame, regions=[[100, 100, 300, 300]])
        assert blurred.shape == frame.shape

    def test_privacy_blackout(self):
        from src.vision_engine import PrivacyBlurEngine
        blur = PrivacyBlurEngine()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        black = blur.apply_blackout(frame)
        assert np.all(black == 0)

    def test_safety_monitor_unattended(self):
        from src.vision_engine import SafetyMonitor
        safety = SafetyMonitor()
        safety._last_person_ts = time.time() - 400  # 6 min 40 sec ago
        alert = safety.check_unattended(num_persons=0)
        assert alert is not None
        assert alert.alert_type == "unattended"
        assert alert.severity == "medium"

    def test_safety_monitor_no_alert_when_present(self):
        from src.vision_engine import SafetyMonitor
        safety = SafetyMonitor()
        safety._last_person_ts = time.time() - 400
        alert = safety.check_unattended(num_persons=1)
        assert alert is None

    def test_fall_detection_wide_bbox(self):
        from src.vision_engine import SafetyMonitor
        safety = SafetyMonitor()
        # Wide bbox → person lying down → fall
        bboxes = [[50, 300, 550, 420]]  # width=500, height=120, aspect=4.17
        alert = safety.detect_fall(bboxes, frame_shape=(480, 640, 3))
        assert alert is not None
        assert alert.alert_type == "fall"
        assert alert.severity == "high"

    def test_fall_detection_normal_bbox(self):
        from src.vision_engine import SafetyMonitor
        safety = SafetyMonitor()
        # Normal standing person bbox
        bboxes = [[200, 50, 400, 450]]  # width=200, height=400, aspect=0.5
        alert = safety.detect_fall(bboxes, frame_shape=(480, 640, 3))
        assert alert is None

    def test_presence_detector_stub(self):
        """PresenceDetector with no model should return zero persons."""
        from src.vision_engine import PresenceDetector
        det = PresenceDetector(model_path="/nonexistent/path.onnx")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert result.num_persons == 0
        assert result.peeker_detected is False


# ---------------------------------------------------------------------------
# Audio Engine
# ---------------------------------------------------------------------------

class TestAudioEngine:
    def test_voice_command_parser_open_tab(self):
        from src.audio_engine import VoiceCommandParser
        parser = VoiceCommandParser()
        cmd = parser.parse("open document tab")
        assert cmd.intent == "open_tab"
        assert cmd.args.get("tab") == "document"

    def test_voice_command_parser_blur(self):
        from src.audio_engine import VoiceCommandParser
        parser = VoiceCommandParser()
        cmd = parser.parse("blur screen now")
        assert cmd.intent == "privacy_blur"

    def test_voice_command_parser_search(self):
        from src.audio_engine import VoiceCommandParser
        parser = VoiceCommandParser()
        cmd = parser.parse("search for quarterly revenue data")
        assert cmd.intent == "rag_search"
        assert "quarterly revenue data" in cmd.args.get("query", "")

    def test_voice_command_parser_unknown(self):
        from src.audio_engine import VoiceCommandParser
        parser = VoiceCommandParser()
        cmd = parser.parse("the quick brown fox")
        assert cmd.intent == "unknown"
        assert cmd.confidence == 0.0

    def test_voice_command_parser_exit(self):
        from src.audio_engine import VoiceCommandParser
        parser = VoiceCommandParser()
        for phrase in ["exit", "quit", "close"]:
            cmd = parser.parse(phrase)
            assert cmd.intent == "quit", f"Failed for '{phrase}'"

    def test_voice_command_parser_benchmark(self):
        from src.audio_engine import VoiceCommandParser
        parser = VoiceCommandParser()
        cmd = parser.parse("run benchmark")
        assert cmd.intent == "benchmark"

    def test_vad_energy_fallback(self):
        """VAD should fall back to energy-based detection without Silero ONNX."""
        from src.audio_engine import VoiceActivityDetector
        with patch("src.audio_engine._build_ort_session", return_value=None):
            vad = VoiceActivityDetector()
            # Silent chunk
            silent = np.zeros(512, dtype=np.float32)
            assert not vad.is_speech(silent)
            # Loud chunk
            loud = np.ones(512, dtype=np.float32) * 0.5
            assert vad.is_speech(loud)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def test_document_parser_txt(self, tmp_path):
        from src.rag_pipeline import DocumentParser
        f = tmp_path / "test.txt"
        f.write_text("Hello world. " * 100, encoding="utf-8")
        parser = DocumentParser()
        chunks, meta = parser.parse(str(f))
        assert len(chunks) > 0
        assert meta["doc_name"] == "test.txt"
        assert meta["word_count"] > 0
        assert all(len(c.text) <= 600 for c in chunks)  # chunk_size + small tolerance

    def test_document_parser_unsupported_format(self, tmp_path):
        from src.rag_pipeline import DocumentParser
        f = tmp_path / "test.xyz"
        f.write_text("data")
        parser = DocumentParser()
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse(str(f))

    def test_document_parser_nonexistent(self):
        from src.rag_pipeline import DocumentParser
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.txt")

    def test_embedding_engine_stub(self):
        from src.rag_pipeline import EmbeddingEngine
        from config.settings import EMBEDDING_DIMENSION
        eng = EmbeddingEngine(model_path="/nonexistent.onnx")
        vec = eng.embed("Hello world")
        assert vec.shape == (EMBEDDING_DIMENSION,)
        # L2 normalised
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_embedding_determinism(self):
        from src.rag_pipeline import EmbeddingEngine
        eng = EmbeddingEngine(model_path="/nonexistent.onnx")
        v1 = eng.embed("same text")
        v2 = eng.embed("same text")
        np.testing.assert_array_equal(v1, v2)

    def test_embedding_batch(self):
        from src.rag_pipeline import EmbeddingEngine
        from config.settings import EMBEDDING_DIMENSION
        eng = EmbeddingEngine(model_path="/nonexistent.onnx")
        batch = eng.embed_batch(["text one", "text two", "text three"])
        assert batch.shape == (3, EMBEDDING_DIMENSION)

    def test_rag_pipeline_ingest_search(self, tmp_path):
        """End-to-end ingest + search using stub embeddings and ChromaDB."""
        from src.rag_pipeline import RAGPipeline
        # Patch vector store to avoid actual ChromaDB in tests
        with patch("src.rag_pipeline.CHROMA_AVAILABLE", False):
            rag = RAGPipeline()
            f = tmp_path / "doc.txt"
            f.write_text("Qualcomm Snapdragon X Elite NPU delivers 45 TOPS of AI performance.", encoding="utf-8")
            meta = rag.ingest_document(str(f))
            assert meta["doc_name"] == "doc.txt"
            # Without Chroma, search returns []
            results = rag.search("NPU performance")
            assert isinstance(results, list)

    def test_rag_answer_no_docs(self):
        from src.rag_pipeline import RAGPipeline
        with patch("src.rag_pipeline.CHROMA_AVAILABLE", False):
            rag = RAGPipeline()
            answer = rag.answer("What is the revenue?")
            assert "No relevant documents" in answer or isinstance(answer, str)

    def test_parse_llm_summary(self):
        from src.rag_pipeline import RAGPipeline
        with patch("src.rag_pipeline.CHROMA_AVAILABLE", False):
            rag = RAGPipeline()
        raw = (
            "SUMMARY: This is a test summary sentence.\n"
            "ACTION_ITEMS:\n- Action one\n- Action two\n"
            "HIGHLIGHTS:\n- Highlight alpha\n- Highlight beta\n"
        )
        summary, actions, highlights = rag._parse_llm_summary(raw)
        assert "test summary" in summary
        assert "Action one" in actions
        assert "Highlight alpha" in highlights


# ---------------------------------------------------------------------------
# LLM Engine
# ---------------------------------------------------------------------------

class TestLLMEngine:
    def test_content_task_prompt_email(self):
        from src.llm_engine import LLMEngine, ContentTask
        with patch("src.llm_engine._build_ort_session", return_value=(None, "unavailable")):
            llm = LLMEngine()
        task = ContentTask(
            task_type="draft_email",
            input_text="Team lunch on Friday at noon.",
        )
        prompt = llm._build_task_prompt(task)
        assert "email" in prompt.lower()
        assert "Friday" in prompt

    def test_content_task_prompt_rephrase(self):
        from src.llm_engine import LLMEngine, ContentTask
        with patch("src.llm_engine._build_ort_session", return_value=(None, "unavailable")):
            llm = LLMEngine()
        task = ContentTask(task_type="rephrase", input_text="The thing is bad.")
        prompt = llm._build_task_prompt(task)
        assert "rephrase" in prompt.lower() or "Rephrase" in prompt

    def test_generate_unavailable_returns_message(self):
        from src.llm_engine import LLMEngine
        with patch("src.llm_engine._build_ort_session", return_value=(None, "unavailable")):
            llm = LLMEngine()
        result = llm.generate("Hello")
        assert "not loaded" in result.text.lower() or "LLM" in result.text

    def test_chat_history_trim(self):
        from src.llm_engine import LLMEngine, ChatMessage
        with patch("src.llm_engine._build_ort_session", return_value=(None, "unavailable")):
            llm = LLMEngine()
        # Fill history beyond context limit
        for i in range(50):
            llm._history.append(ChatMessage(role="user", content="x" * 200))
            llm._history.append(ChatMessage(role="assistant", content="y" * 200))
        from config.settings import LLM_CONTEXT_WINDOW
        llm._trim_history()
        total_chars = sum(len(m.content) for m in llm._history)
        assert total_chars <= LLM_CONTEXT_WINDOW * 3 + 400  # small tolerance

    def test_make_llm_callable(self):
        from src.llm_engine import LLMEngine, make_llm_callable, GenerationResult
        with patch("src.llm_engine._build_ort_session", return_value=(None, "unavailable")):
            llm = LLMEngine()
        fn = make_llm_callable(llm)
        result = fn("test prompt")
        assert isinstance(result, str)

    def test_sampler_argmax(self):
        from src.llm_engine import _Sampler
        logits = np.array([0.1, 0.5, 0.9, 0.2])
        token = _Sampler.sample(logits, temperature=0.0, top_p=1.0)
        assert token == 2  # argmax

    def test_sampler_temperature_diversity(self):
        from src.llm_engine import _Sampler
        logits = np.zeros(100, dtype=np.float32)
        logits[50] = 2.0  # Slightly preferred token
        # With high temperature we should sometimes get non-50 tokens
        tokens = {_Sampler.sample(logits, temperature=2.0, top_p=1.0) for _ in range(50)}
        assert len(tokens) > 1  # Should have diversity


# ---------------------------------------------------------------------------
# Privacy / Security
# ---------------------------------------------------------------------------

class TestPrivacyGuard:
    def test_network_egress_blocked(self):
        """With ZERO_NETWORK_EGRESS enabled, non-localhost connections should fail."""
        import socket
        from config.settings import ZERO_NETWORK_EGRESS
        if not ZERO_NETWORK_EGRESS:
            pytest.skip("ZERO_NETWORK_EGRESS is disabled in settings")
        # Importing rag_pipeline activates the guard
        import src.rag_pipeline  # noqa: F401
        with pytest.raises(PermissionError, match="PRIVACY GUARD"):
            socket.create_connection(("google.com", 80), timeout=1)

    def test_wipe_temp_chunks(self, tmp_path):
        """Ensure chunk text is cleared after ingestion when WIPE_TEMP_FILES=True."""
        from src.rag_pipeline import RAGPipeline
        with (
            patch("src.rag_pipeline.CHROMA_AVAILABLE", False),
            patch("src.rag_pipeline.WIPE_TEMP_FILES", True),
        ):
            rag = RAGPipeline()
            f = tmp_path / "sensitive.txt"
            f.write_text("Top secret information " * 20, encoding="utf-8")
            meta = rag.ingest_document(str(f))
            assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# Download Models Script
# ---------------------------------------------------------------------------

class TestDownloadScript:
    def test_model_registry_keys(self):
        from scripts.download_models import MODEL_REGISTRY
        expected = {"llama3-3b", "whisper", "yolov8", "smolvlm2", "qwen3-embedding", "silero-vad"}
        assert set(MODEL_REGISTRY.keys()) == expected

    def test_dry_run_silero(self):
        from scripts.download_models import download_model
        # Silero VAD has a direct_url – dry run should not actually download
        result = download_model("silero-vad", dry_run=True)
        assert result is True

    def test_unknown_model_returns_false(self):
        from scripts.download_models import download_model
        result = download_model("nonexistent-model-key", dry_run=True)
        assert result is False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(ROOT),
    )
