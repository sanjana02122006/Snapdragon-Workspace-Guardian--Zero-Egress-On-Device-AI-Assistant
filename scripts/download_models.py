"""
scripts/download_models.py
===========================
Qualcomm AI Hub Model Downloader

Downloads pre-compiled ONNX models optimised for Snapdragon X-series NPU
from the Qualcomm AI Hub API and places them in the /models directory.

Requirements:
  pip install qai-hub qai_hub_models

Usage:
  python scripts/download_models.py                # Download all models
  python scripts/download_models.py --model whisper  # Download specific model
  python scripts/download_models.py --list           # List available models
  python scripts/download_models.py --dry-run        # Preview without downloading
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import AIHUB_MODELS, MODELS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Model registry with AI Hub submit params
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "llama3-3b": {
        "hub_name": "Llama-3.2-3B-Instruct",
        "target_dir": MODELS_DIR / "llama-3.2-3b-instruct",
        "device": "Snapdragon X Elite CRD",
        "components": ["model_encoder.onnx", "model_decoder_kv.onnx", "tokenizer.json"],
        "description": "3B parameter LLM for chat and content generation. Hexagon NPU optimised.",
        "size_gb": 3.2,
    },
    "whisper": {
        "hub_name": "Whisper-Base-En",
        "target_dir": MODELS_DIR / "whisper_base_en.onnx",
        "device": "Snapdragon X Elite CRD",
        "components": ["whisper_base_en.onnx"],
        "description": "Whisper Base English STT model, compiled for Hexagon NPU.",
        "size_gb": 0.15,
    },
    "yolov8": {
        "hub_name": "YOLOv8-Detection",
        "target_dir": MODELS_DIR / "yolov8n.onnx",
        "device": "Snapdragon X Elite CRD",
        "components": ["yolov8n.onnx"],
        "description": "YOLOv8 Nano person/object detection for privacy monitoring.",
        "size_gb": 0.006,
        "export_ultralytics": True,  # Export via ultralytics package
    },
    "smolvlm2": {
        "hub_name": "SmolVLM2",
        "target_dir": MODELS_DIR / "smolvlm2.onnx",
        "device": "Snapdragon X Elite CRD",
        "components": ["smolvlm2.onnx"],
        "description": "Small Vision-Language Model for image analysis and OCR understanding.",
        "size_gb": 1.8,
    },
    "qwen3-embedding": {
        "hub_name": "Qwen3-Embedding",
        "target_dir": MODELS_DIR / "qwen3_embedding.onnx",
        "device": "Snapdragon X Elite CRD",
        "components": ["qwen3_embedding.onnx"],
        "description": "Dense embedding model for RAG vector search.",
        "size_gb": 0.5,
    },
    "silero-vad": {
        "hub_name": "silero-vad",
        "target_dir": MODELS_DIR / "silero_vad.onnx",
        "device": "cpu",  # Tiny model; CPU is fine
        "components": ["silero_vad.onnx"],
        "description": "Voice Activity Detection model (Silero VAD ONNX).",
        "size_gb": 0.005,
        "direct_url": "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx",
    },
}


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def list_models() -> None:
    print("\n Available models for Snapdragon X Elite NPU:\n")
    print(f"  {'Key':<20} {'Hub Name':<30} {'Size':<10} Description")
    print("  " + "─" * 90)
    for key, cfg in MODEL_REGISTRY.items():
        size = f"{cfg['size_gb']:.2f} GB"
        print(f"  {key:<20} {cfg['hub_name']:<30} {size:<10} {cfg['description']}")
    print()


def download_via_hub(key: str, cfg: dict, dry_run: bool = False) -> bool:
    """
    Submit a compile job to Qualcomm AI Hub and download the resulting ONNX model.
    Requires: pip install qai-hub && qai-hub configure (set API token)
    """
    try:
        import qai_hub as hub  # type: ignore
    except ImportError:
        logger.error(
            "qai_hub not installed. Run: pip install qai-hub\n"
            "Then authenticate: qai-hub configure --api_token YOUR_TOKEN"
        )
        return False

    target_dir = Path(str(cfg["target_dir"])).parent
    target_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("[DRY-RUN] Would submit AI Hub job for: %s → %s", cfg["hub_name"], target_dir)
        return True

    logger.info("Submitting AI Hub compile job: %s", cfg["hub_name"])
    try:
        # Get the device
        device = hub.Device(cfg["device"])

        # Load model from AI Hub model zoo
        from qai_hub_models.models import get_model  # type: ignore
        model_cls = get_model(cfg["hub_name"])
        model = model_cls.from_pretrained()

        # Export as ONNX
        sample_inputs = model.sample_inputs()

        job = hub.submit_compile_job(
            model=model.convert_to_onnx_and_aimet_config()[0],
            device=device,
            input_specs=sample_inputs,
            options="--target_runtime onnx --output_names logits",
        )
        logger.info("Job submitted: %s | Waiting for completion…", job.job_id)

        # Poll until done
        while True:
            status = job.get_status()
            logger.info("  Status: %s", status.state)
            if status.state in ("PASSED", "FAILED"):
                break
            time.sleep(10)

        if status.state == "PASSED":
            output_dir = target_dir
            job.download_target_model(str(output_dir))
            logger.info("✓ Downloaded %s to %s", cfg["hub_name"], output_dir)
            return True
        else:
            logger.error("Job failed for %s: %s", cfg["hub_name"], status.message)
            return False

    except Exception as exc:
        logger.error("AI Hub download failed for %s: %s", key, exc)
        return False


def download_direct(key: str, cfg: dict, dry_run: bool = False) -> bool:
    """Download model directly from a URL (for non-Hub models like Silero VAD)."""
    import urllib.request
    url = cfg.get("direct_url", "")
    if not url:
        return False
    target = Path(str(cfg["target_dir"]))
    if target.is_dir():
        target = target / Path(url).name
    target.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("[DRY-RUN] Would download: %s → %s", url, target)
        return True
    if target.exists():
        logger.info("Already exists: %s — skipping.", target)
        return True

    logger.info("Downloading %s …", url)
    try:
        urllib.request.urlretrieve(url, str(target))
        logger.info("✓ Saved to %s  (%.2f MB)", target, target.stat().st_size / 1_048_576)
        return True
    except Exception as exc:
        logger.error("Direct download failed: %s", exc)
        return False


def download_model(key: str, dry_run: bool = False) -> bool:
    if key not in MODEL_REGISTRY:
        logger.error("Unknown model key: %s. Use --list to see available models.", key)
        return False
    cfg = MODEL_REGISTRY[key]
    logger.info("─── %s (%s) ───", key, cfg["hub_name"])

    if cfg.get("export_ultralytics"):
        return export_via_ultralytics(key, cfg, dry_run)
    elif cfg.get("direct_url"):
        return download_direct(key, cfg, dry_run)
    else:
        return download_via_hub(key, cfg, dry_run)


def export_via_ultralytics(key: str, cfg: dict, dry_run: bool = False) -> bool:
    """Export YOLOv8 model to ONNX format using the ultralytics package."""
    target = Path(str(cfg["target_dir"]))
    if target.exists():
        logger.info("Already exists: %s — skipping.", target)
        return True
    if dry_run:
        logger.info("[DRY-RUN] Would export via ultralytics → %s", target)
        return True
    try:
        from ultralytics import YOLO  # type: ignore
        logger.info("Exporting yolov8n to ONNX via ultralytics…")
        model = YOLO("yolov8n.pt")
        export_path = model.export(format="onnx", opset=12, dynamic=False)
        import shutil
        shutil.move(str(export_path), str(target))
        logger.info("✓ Exported and saved to %s", target)
        return True
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as exc:
        logger.error("ultralytics export failed: %s", exc)
        return False


def download_all(dry_run: bool = False) -> None:
    results = {}
    for key in MODEL_REGISTRY:
        ok = download_model(key, dry_run)
        results[key] = "✓" if ok else "✗"
        print()

    print("\n Download Summary:")
    for key, status in results.items():
        print(f"  {status}  {key}")


# ---------------------------------------------------------------------------
# Manifest dump
# ---------------------------------------------------------------------------

def dump_manifest() -> None:
    """Write a manifest.json listing all expected model files."""
    manifest = {}
    for key, cfg in MODEL_REGISTRY.items():
        target = str(cfg["target_dir"])
        manifest[key] = {
            "hub_name": cfg["hub_name"],
            "local_path": target,
            "exists": Path(target).exists(),
            "size_gb": cfg["size_gb"],
        }
    out = ROOT / "models" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Qualcomm AI Hub models for Snapdragon Local Assistant"
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Download a specific model by key")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    parser.add_argument("--manifest", action="store_true", help="Dump model manifest JSON")
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.manifest:
        dump_manifest()
    elif args.model:
        download_model(args.model, dry_run=args.dry_run)
    elif args.all:
        download_all(dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
