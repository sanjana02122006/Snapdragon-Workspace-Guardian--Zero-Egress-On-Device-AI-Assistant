# Snapdragon Local Workspace & Privacy Assistant

**Production-ready, 100% on-device AI desktop application for Windows on Snapdragon X Elite / X Plus / X2 Elite PCs**

> All AI inference runs locally on the **Qualcomm Hexagon NPU** via ONNX Runtime with `QNNExecutionProvider`. Zero cloud dependencies. Zero network egress.

---

## 🏗 Project Structure

```
snapdragon_assistant/
├── app.py                        ← Main entry point (PyQt6 GUI)
├── requirements.txt
│
├── config/
│   └── settings.py               ← All paths, model IDs, HW targets, tunables
│
├── src/
│   ├── vision_engine.py          ← Peeker detection, privacy blur, safety, VLM
│   ├── audio_engine.py           ← VAD, Whisper STT, voice commands, TTS
│   ├── rag_pipeline.py           ← PDF parser, embeddings, ChromaDB, Q&A
│   ├── llm_engine.py             ← Llama-3.2-3B ONNX, streaming, chat
│   └── npu_benchmarker.py        ← Latency/TTFT/RAM benchmarking harness
│
├── scripts/
│   └── download_models.py        ← Qualcomm AI Hub model downloader
│
├── tests/
│   └── test_engines.py           ← pytest suite (40+ unit + integration tests)
│
├── models/                       ← ONNX model files (downloaded by script)
├── data/
│   ├── uploads/                  ← User-uploaded documents
│   └── vector_store/             ← ChromaDB persistent store
├── logs/                         ← Benchmark JSONL + application log
└── assets/
    ├── icons/
    └── sounds/                   ← Alert WAV files
```

---

## 🚀 Quick Start

### 1. Install dependencies

```powershell
cd snapdragon_assistant
pip install -r requirements.txt
```

> **For Snapdragon NPU support**, install the Qualcomm ONNX Runtime wheel:
> ```powershell
> pip install onnxruntime-qnn   # Windows ARM64 wheel from Qualcomm
> ```

### 2. Download AI Hub models

```powershell
# List available models
python scripts/download_models.py --list

# Download all models (requires qai-hub token)
pip install qai-hub qai-hub-models
qai-hub configure --api_token YOUR_TOKEN
python scripts/download_models.py --all

# Download specific model
python scripts/download_models.py --model whisper
python scripts/download_models.py --model yolov8

# Download Silero VAD (no API token needed)
python scripts/download_models.py --model silero-vad
```

### 3. Launch the application

```powershell
python app.py
```

---

## 🧠 Feature Matrix

| # | Feature | Module | Model | Hardware Target |
|---|---------|--------|-------|-----------------|
| 1 | 👁 Peeker & Presence Detection | `vision_engine.py` | YOLOv8-Detection | Hexagon NPU |
| 2 | 🔒 Automatic Privacy Blur | `vision_engine.py` | OpenCV | CPU |
| 3 | 🚨 Safety & Incident Monitor | `vision_engine.py` | Heuristic | CPU |
| 4 | 🖼 Screenshot & Image Analysis | `vision_engine.py` | SmolVLM2 + Tesseract | Hexagon NPU |
| 5 | 📄 PDF Reader & Summariser | `rag_pipeline.py` | Llama-3.2-3B | Hexagon NPU |
| 6 | 📝 Document Note Generator | `rag_pipeline.py` | Llama-3.2-3B | Hexagon NPU |
| 7 | 🔍 In-Document Semantic Search | `rag_pipeline.py` | Qwen3-Embedding | Hexagon NPU |
| 8 | ❓ Document Q&A | `rag_pipeline.py` | Qwen3-Embedding + Llama | Hexagon NPU |
| 9 | 🔒 Sensitive Local Processing | `rag_pipeline.py` | Network egress guard | N/A |
| 10 | 🎤 Voice Command Navigation | `audio_engine.py` | Whisper-Base-En | Hexagon NPU |
| 11 | ✍ Content Generation | `llm_engine.py` | Llama-3.2-3B | Hexagon NPU |
| 12 | 🔊 Local Audio Feedback | `audio_engine.py` | CoquiTTS (local) | CPU |
| 13 | ⚡ Hardware Benchmarking | `npu_benchmarker.py` | ONNX telemetry | All |
| 14 | 🌐 100% Offline Resilience | All modules | — | All |
| 15 | Qualcomm AI Hub Optimisation | `scripts/` | Pre-compiled ONNX | Hexagon NPU |

---

## 🔧 Qualcomm AI Hub Models

| Capability | AI Hub Model Slug | Hardware Target |
|------------|-------------------|-----------------|
| LLM (Chat & Generation) | `Llama-3.2-3B-Instruct` | Hexagon NPU |
| Speech-to-Text | `Whisper-Base-En` | Hexagon NPU |
| Person / Object Detection | `YOLOv8-Detection` | Hexagon NPU |
| Vision-Language Model | `SmolVLM2` | Hexagon NPU |
| Document Embeddings | `Qwen3-Embedding` | Hexagon NPU |
| Voice Activity Detection | `silero-vad` (ONNX) | CPU |

---

## ⚡ ONNX Execution Provider Cascade

The application automatically selects the best available execution provider:

```
QNNExecutionProvider  (Hexagon NPU – burst mode, FP16)
        ↓ fallback
DmlExecutionProvider  (Adreno GPU via DirectML)
        ↓ fallback
CPUExecutionProvider  (x86/ARM64 CPU)
```

Configured in [`config/settings.py`](config/settings.py):
```python
ONNX_EP_PRIORITY = ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
QNN_EP_OPTIONS = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "burst",
    "htp_graph_finalization_optimization_level": "3",
    "enable_htp_fp16_precision": "1",
}
```

---

## 🔒 Privacy Architecture

- **Zero network egress guard** — `socket.create_connection` is patched to block non-localhost connections when `ZERO_NETWORK_EGRESS=True`
- **Memory wipe** — Document chunks are cleared from memory after embedding when `WIPE_TEMP_FILES=True`
- **No cloud APIs** — All model inference calls are local ONNX runtime sessions
- **Local vector store** — ChromaDB uses SQLite (local disk only, `anonymized_telemetry=False`)
- **Camera data** — Frames are processed in-process; never serialised or transmitted

---

## 🧪 Running Tests

```powershell
pytest tests/test_engines.py -v
pytest tests/test_engines.py -v --tb=short -k "not slow"
pytest tests/test_engines.py -v --cov=src --cov-report=term-missing
```

---

## 📊 Benchmarking

```python
from src.npu_benchmarker import NPUBenchmarker
import time

bench = NPUBenchmarker(model_name="MyModel", provider="QNNExecutionProvider")
summary = bench.run(fn=my_inference_fn, fn_args=(inputs,), n_runs=20)
print(summary)
# Output:
# [MyModel] Provider=QNNExecutionProvider  N=20
#   Latency  → mean=12.4ms  min=11.8ms  max=14.3ms  p95=13.9ms
#   TTFT     → 38.2 ms
#   Throughput → 47.3 units/s
#   RAM delta → 12.0 MB
```

Results are appended to `logs/benchmark.jsonl` for offline analysis.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `PyQt6` | Desktop GUI |
| `onnxruntime` / `onnxruntime-qnn` | Model inference on NPU/GPU/CPU |
| `opencv-python` | Camera & image processing |
| `sounddevice` / `librosa` | Microphone capture & audio preprocessing |
| `chromadb` | Local vector store |
| `PyMuPDF` | PDF text extraction |
| `transformers` | Llama / Whisper tokenizers |
| `psutil` | CPU/RAM telemetry |
| `TTS` (optional) | Offline text-to-speech (CoquiTTS) |
| `qai-hub` (optional) | Qualcomm AI Hub model downloads |

---

## 📋 Roadmap

- [ ] Integrate full Whisper decoder ONNX (split encoder/decoder sessions)
- [ ] Wire in SmolVLM2 tokenizer for VLM output decoding
- [ ] Add LanceDB as alternative vector store backend
- [ ] Implement MobileNetV4 for faster presence detection
- [ ] Add real-time NPU utilisation via Qualcomm WinRT API
- [ ] Package as MSIX for Windows Store distribution
- [ ] Integrate `onnxruntime-extensions` for end-to-end Whisper pipeline

---

## ⚖ License

MIT License — See [LICENSE](LICENSE) for details.

---

*Built with ❤ for Qualcomm Snapdragon X Elite · Windows on Snapdragon · Qualcomm AI Hub*
