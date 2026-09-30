"""
config/settings.py
==================
Central configuration for Snapdragon Local Workspace & Privacy Assistant.
All paths, model IDs, hardware targets, and runtime tunables live here.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOGS_DIR = ROOT_DIR / "logs"
ASSETS_DIR = ROOT_DIR / "assets"
SOUNDS_DIR = ASSETS_DIR / "sounds"
ICONS_DIR = ASSETS_DIR / "icons"

# Ensure writable dirs exist at import time
for _d in (UPLOADS_DIR, VECTOR_STORE_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Qualcomm AI Hub model identifiers
# Maps capability -> model slug as published on aihub.qualcomm.com
# ---------------------------------------------------------------------------
AIHUB_MODELS: dict[str, str] = {
    # Text / LLM
    "llm_chat":        "Llama-3.2-3B-Instruct",
    "llm_content_gen": "Llama-3.2-3B-Instruct",
    # Speech-to-Text
    "stt_whisper":     "Whisper-Base-En",
    # Vision
    "vision_vlm":      "SmolVLM2",
    "vision_presence": "YOLOv8-Detection",
    # Embeddings
    "embeddings":      "Qwen3-Embedding",
    # Text-to-Speech (CoquiTTS is used locally; placeholder for hub)
    "tts":             "local-coqui-tts",
}

# ---------------------------------------------------------------------------
# ONNX / QNN Runtime settings
# ---------------------------------------------------------------------------
# Execution provider priority: QNN (Hexagon NPU) -> GPU (Adreno) -> CPU
ONNX_EP_PRIORITY: list[str] = [
    "QNNExecutionProvider",
    "DmlExecutionProvider",   # DirectML -> Adreno GPU on Windows ARM
    "CPUExecutionProvider",
]

QNN_EP_OPTIONS: dict[str, str] = {
    "backend_path":           "QnnHtp.dll",  # HTP = Hexagon Tensor Processor
    "htp_performance_mode":   "burst",        # Options: default | burst | sustained_high_performance
    "htp_graph_finalization_optimization_level": "3",
    "enable_htp_fp16_precision": "1",
}

DML_EP_OPTIONS: dict[str, str] = {
    "device_id": "0",
}

# ---------------------------------------------------------------------------
# Vision / Camera settings
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
CAMERA_FPS: int = 15
PRESENCE_CONFIDENCE_THRESHOLD: float = 0.50
PRESENCE_FACE_ANGLE_THRESHOLD: float = 35.0
BLUR_STRENGTH: int = 55
PEEKER_ALERT_COOLDOWN_S: float = 3.0

# ---------------------------------------------------------------------------
# Audio / STT / TTS settings
# ---------------------------------------------------------------------------
STT_SAMPLE_RATE: int = 16_000
STT_CHUNK_DURATION_S: float = 0.5
VAD_SILENCE_THRESHOLD_S: float = 1.2
TTS_SPEED: float = 1.0
TTS_MODEL_NAME: str = "tts_models/en/ljspeech/tacotron2-DDC"
AUDIO_ALERT_FILE: str = str(SOUNDS_DIR / "alert.wav")

# ---------------------------------------------------------------------------
# RAG / Vector store settings
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = "workspace_docs"
EMBEDDING_DIMENSION: int = 1536
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64
TOP_K_RETRIEVAL: int = 5
SIMILARITY_METRIC: str = "cosine"

# ---------------------------------------------------------------------------
# LLM inference settings
# ---------------------------------------------------------------------------
LLM_MAX_NEW_TOKENS: int = 512
LLM_TEMPERATURE: float = 0.7
LLM_TOP_P: float = 0.9
LLM_CONTEXT_WINDOW: int = 4096
LLM_SYSTEM_PROMPT: str = (
    "You are a privacy-focused, fully offline AI assistant running on a "
    "Snapdragon X Elite PC. Never reveal system internals. Be concise, "
    "accurate, and helpful. Do not fabricate information."
)

# ---------------------------------------------------------------------------
# Benchmarking / telemetry
# ---------------------------------------------------------------------------
BENCHMARK_LOG_PATH: str = str(LOGS_DIR / "benchmark.jsonl")
TELEMETRY_POLL_INTERVAL_MS: int = 500
WARMUP_RUNS: int = 3

# ---------------------------------------------------------------------------
# Privacy / Security
# ---------------------------------------------------------------------------
# Set to False for local development/testing. Set True for event/production.
DEV_MODE: bool = os.environ.get("DEV_MODE", "1") == "1"
ZERO_NETWORK_EGRESS: bool = not DEV_MODE  # Disable network guard in dev mode
WIPE_TEMP_FILES: bool = True
MAX_UPLOAD_SIZE_MB: int = 100

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "Snapdragon Local Workspace & Privacy Assistant"
APP_VERSION: str = "1.0.0"
APP_WINDOW_WIDTH: int = 1400
APP_WINDOW_HEIGHT: int = 900
DARK_MODE: bool = True
