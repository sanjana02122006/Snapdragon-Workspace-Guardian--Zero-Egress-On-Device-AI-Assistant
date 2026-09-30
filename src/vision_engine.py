"""
src/vision_engine.py
====================
On-Device Vision, Safety & Privacy Module

Handles:
  1. Peeker / Shoulder-Surfer Detection (YOLO person detection + face angle)
  2. Automatic Privacy Blur (blurs screen regions / document windows)
  3. Local Safety & Incident Monitoring (fall / distress / unattended screen)
  4. Screenshot & Image Analysis (OCR + SmolVLM2 multimodal understanding)

All inference runs locally via ONNX Runtime with QNN EP (Hexagon NPU).
Camera frames are NEVER sent over the network.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ONNX Runtime with QNN/DML/CPU EP cascade
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logging.warning("onnxruntime not installed. Vision engine will run in stub mode.")

from config.settings import (
    BLUR_STRENGTH,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DML_EP_OPTIONS,
    MODELS_DIR,
    ONNX_EP_PRIORITY,
    PEEKER_ALERT_COOLDOWN_S,
    PRESENCE_CONFIDENCE_THRESHOLD,
    QNN_EP_OPTIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result from a single YOLO inference pass."""
    num_persons: int
    bboxes: list[list[float]]          # [[x1,y1,x2,y2], ...]
    confidences: list[float]
    peeker_detected: bool = False
    lateral_angles: list[float] = None # deg off-axis per person
    inference_ms: float = 0.0


@dataclass
class VisionAlert:
    alert_type: str          # "peeker" | "fall" | "unattended" | "privacy"
    message: str
    severity: str            # "low" | "medium" | "high"
    frame_snapshot: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class ImageAnalysisResult:
    ocr_text: str
    vlm_description: str
    detected_objects: list[str]
    inference_ms: float


# ---------------------------------------------------------------------------
# ONNX Session builder (NPU -> GPU -> CPU cascade)
# ---------------------------------------------------------------------------

def _build_ort_session(model_path: str) -> Optional["ort.InferenceSession"]:
    """
    Attempts to create an ONNX InferenceSession using the EP priority list.
    Falls back gracefully from QNN -> DML -> CPU.
    """
    if not ORT_AVAILABLE:
        logger.error("onnxruntime not available; cannot build session.")
        return None
    if not Path(model_path).exists():
        logger.warning("Model not found at %s; skipping session creation.", model_path)
        return None

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Build EP list with options
    eps: list[tuple | str] = []
    for ep in ONNX_EP_PRIORITY:
        if ep == "QNNExecutionProvider":
            eps.append((ep, QNN_EP_OPTIONS))
        elif ep == "DmlExecutionProvider":
            eps.append((ep, DML_EP_OPTIONS))
        else:
            eps.append(ep)

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=eps,
        )
        active_ep = session.get_providers()[0]
        logger.info("ORT session created: model=%s  EP=%s", Path(model_path).name, active_ep)
        return session
    except Exception as exc:
        logger.error("Failed to create ORT session for %s: %s", model_path, exc)
        return None


# ---------------------------------------------------------------------------
# Peeker & Presence Detector
# ---------------------------------------------------------------------------

class PresenceDetector:
    """
    Uses YOLOv8 (ONNX) to detect persons in camera frames.
    Flags 'peeker' events when a second person is detected or when
    a detected face appears at an oblique viewing angle.
    """

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(self, model_path: Optional[str] = None) -> None:
        # Prefer MobileNetV4 (Snapdragon NPU optimised), fall back to YOLOv8n
        mobilenet_path = str(MODELS_DIR / "mobilenetv4.onnx")
        yolo_path = str(MODELS_DIR / "yolov8n.onnx")
        if model_path:
            self.model_path = model_path
        elif Path(mobilenet_path).exists():
            self.model_path = mobilenet_path
        else:
            self.model_path = yolo_path
        self.session = _build_ort_session(self.model_path)
        self._last_alert_ts: float = 0.0
        self._input_name: Optional[str] = None
        if self.session:
            self._input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Run YOLO detection on a BGR frame. Returns DetectionResult."""
        t0 = time.perf_counter()

        if self.session is None:
            return DetectionResult(num_persons=0, bboxes=[], confidences=[],
                                   inference_ms=0.0)

        # Pre-process: resize -> RGB -> NCHW float32 normalised [0,1]
        resized = cv2.resize(frame_bgr, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # NCHW

        try:
            outputs = self.session.run(None, {self._input_name: tensor})
            raw = outputs[0]  # (1, num_classes+4, num_anchors)
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return DetectionResult(num_persons=0, bboxes=[], confidences=[],
                                   inference_ms=(time.perf_counter() - t0) * 1000)

        bboxes, confs = self._parse_yolo_output(raw, frame_bgr.shape)
        peeker = self._classify_peeker(bboxes, frame_bgr.shape)
        inf_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            num_persons=len(bboxes),
            bboxes=bboxes,
            confidences=confs,
            peeker_detected=peeker,
            inference_ms=inf_ms,
        )

    def should_alert(self, result: DetectionResult) -> bool:
        """Rate-limit peeker alerts."""
        if not result.peeker_detected:
            return False
        now = time.time()
        if (now - self._last_alert_ts) >= PEEKER_ALERT_COOLDOWN_S:
            self._last_alert_ts = now
            return True
        return False

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _parse_yolo_output(
        self, raw: np.ndarray, frame_shape: tuple
    ) -> tuple[list[list[float]], list[float]]:
        """Parse YOLOv8 output tensor into bboxes and confidences."""
        h, w = frame_shape[:2]
        # raw shape: (1, 84, 8400) for YOLOv8n COCO
        predictions = raw[0]            # (84, 8400)
        predictions = predictions.T     # (8400, 84)

        bboxes, confs = [], []
        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if class_id != self.PERSON_CLASS_ID:
                continue
            if confidence < PRESENCE_CONFIDENCE_THRESHOLD:
                continue
            # cx, cy, cw, ch -> x1,y1,x2,y2
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x1 = int((cx - bw / 2) / 640 * w)
            y1 = int((cy - bh / 2) / 640 * h)
            x2 = int((cx + bw / 2) / 640 * w)
            y2 = int((cy + bh / 2) / 640 * h)
            bboxes.append([x1, y1, x2, y2])
            confs.append(confidence)

        return bboxes, confs

    def _classify_peeker(
        self, bboxes: list[list[float]], frame_shape: tuple
    ) -> bool:
        """Peeker = >1 person OR a person whose bbox centroid is off-axis."""
        if len(bboxes) > 1:
            return True
        if len(bboxes) == 1:
            w = frame_shape[1]
            cx = (bboxes[0][0] + bboxes[0][2]) / 2
            # If person centre is strongly to the side, flag as peeker
            off_axis_ratio = abs(cx - w / 2) / (w / 2)
            if off_axis_ratio > 0.6:
                return True
        return False


# ---------------------------------------------------------------------------
# Privacy Screen Blur
# ---------------------------------------------------------------------------

class PrivacyBlurEngine:
    """
    Blurs or blacks out portions of the screen / frame when a peeker
    is detected, protecting sensitive documents visible on display.
    """

    def apply_blur(self, frame: np.ndarray, regions: list[list[int]] | None = None) -> np.ndarray:
        """
        Apply heavy Gaussian blur to specified screen regions.
        If regions is None, blur the entire frame.
        """
        blurred = frame.copy()
        ksize = BLUR_STRENGTH if BLUR_STRENGTH % 2 == 1 else BLUR_STRENGTH + 1
        if regions:
            for x1, y1, x2, y2 in regions:
                roi = blurred[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        else:
            blurred = cv2.GaussianBlur(blurred, (ksize, ksize), 0)
        return blurred

    def apply_blackout(self, frame: np.ndarray) -> np.ndarray:
        """Return a fully black frame (maximum privacy)."""
        return np.zeros_like(frame)

    def overlay_privacy_banner(self, frame: np.ndarray) -> np.ndarray:
        """Add a red 'PRIVACY PROTECTED' banner to the blurred frame."""
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.putText(
            out, "  PRIVACY PROTECTED - UNAUTHORIZED VIEWER DETECTED",
            (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )
        return out


# ---------------------------------------------------------------------------
# Safety & Incident Monitor
# ---------------------------------------------------------------------------

class SafetyMonitor:
    """
    Detects physical distress signals or prolonged unattended screens.
    Operates entirely offline; no frame data is uploaded.
    """

    UNATTENDED_THRESHOLD_S: float = 300.0  # 5 minutes no person -> alert

    def __init__(self) -> None:
        self._last_person_ts: float = time.time()
        self._fall_frame_buffer: list[np.ndarray] = []

    def check_unattended(self, num_persons: int) -> Optional[VisionAlert]:
        now = time.time()
        if num_persons > 0:
            self._last_person_ts = now
            return None
        elapsed = now - self._last_person_ts
        if elapsed > self.UNATTENDED_THRESHOLD_S:
            return VisionAlert(
                alert_type="unattended",
                message=f"Screen unattended for {elapsed / 60:.1f} minutes.",
                severity="medium",
                timestamp=now,
            )
        return None

    def detect_fall(self, bboxes: list[list[float]], frame_shape: tuple) -> Optional[VisionAlert]:
        """
        Heuristic fall detection: a person bounding box that is significantly
        wider than tall (aspect ratio > 2.0) suggests a horizontal/prone posture.
        """
        for x1, y1, x2, y2 in bboxes:
            bw, bh = x2 - x1, y2 - y1
            if bh == 0:
                continue
            aspect = bw / bh
            if aspect > 2.0:
                return VisionAlert(
                    alert_type="fall",
                    message="Possible fall or distress detected.",
                    severity="high",
                    timestamp=time.time(),
                )
        return None


# ---------------------------------------------------------------------------
# Screenshot & Image Analyser (OCR + VLM)
# ---------------------------------------------------------------------------

class ImageAnalyser:
    """
    Multimodal image analysis pipeline:
      1. Tesseract OCR for text extraction
      2. SmolVLM2 (ONNX) for visual understanding and description
    """

    def __init__(self, vlm_model_path: Optional[str] = None) -> None:
        default_vlm = str(MODELS_DIR / "smolvlm2.onnx")
        self.vlm_model_path = vlm_model_path or default_vlm
        self.vlm_session = _build_ort_session(self.vlm_model_path)
        self._ocr_available = self._check_tesseract()

    def analyse(self, image_bgr: np.ndarray, prompt: str = "Describe this image in detail.") -> ImageAnalysisResult:
        t0 = time.perf_counter()
        ocr_text = self._run_ocr(image_bgr)
        vlm_desc = self._run_vlm(image_bgr, prompt)
        inf_ms = (time.perf_counter() - t0) * 1000.0
        return ImageAnalysisResult(
            ocr_text=ocr_text,
            vlm_description=vlm_desc,
            detected_objects=[],
            inference_ms=inf_ms,
        )

    def analyse_file(self, image_path: str, prompt: str = "Describe this image in detail.") -> ImageAnalysisResult:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return self.analyse(img, prompt)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _check_tesseract(self) -> bool:
        try:
            import pytesseract  # type: ignore
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.warning("Tesseract not available; OCR disabled.")
            return False

    def _run_ocr(self, image_bgr: np.ndarray) -> str:
        if not self._ocr_available:
            return "[OCR unavailable]"
        try:
            import pytesseract  # type: ignore
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            return pytesseract.image_to_string(gray).strip()
        except Exception as exc:
            logger.error("OCR error: %s", exc)
            return "[OCR error]"

    # ------------------------------------------------------------------
    # SmolVLM2 inference (placeholder until model ONNX is ready)
    # ------------------------------------------------------------------

    def _run_vlm(self, image_bgr: np.ndarray, prompt: str) -> str:
        if self.vlm_session is None:
            return (
                "[SmolVLM2 model not loaded – place smolvlm2.onnx in models/ dir. "
                "Download via: python scripts/download_models.py --model smolvlm2]"
            )
        try:
            # Resize to 384x384 (SmolVLM2 default)
            resized = cv2.resize(image_bgr, (384, 384))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Normalise with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb = (rgb - mean) / std
            tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # NCHW

            inp_name = self.vlm_session.get_inputs()[0].name
            output = self.vlm_session.run(None, {inp_name: tensor})
            # Decode output tokens -> text (model-specific post-processing)
            return self._decode_vlm_output(output)
        except Exception as exc:
            logger.error("VLM inference error: %s", exc)
            return "[VLM inference failed]"

    def _decode_vlm_output(self, output) -> str:
        """Decode VLM output tokens into text using the SmolVLM2 tokenizer."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-Instruct")
            # Usually output[0] is logits with shape (batch, seq_len, vocab_size)
            token_ids = np.argmax(output[0], axis=-1)
            # Assuming batch size 1
            text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
            return text.strip()
        except Exception as exc:
            logger.error("VLM decode error: %s", exc)
            return f"[VLM decode error: {exc}]"


# ---------------------------------------------------------------------------
# Vision Pipeline Orchestrator
# ---------------------------------------------------------------------------

class VisionEngine:
    """
    Top-level orchestrator that manages the camera loop and routes
    frames to sub-engines in background threads.

    Usage
    -----
    engine = VisionEngine()
    engine.start()
    # frames / alerts arrive via engine.alert_queue
    engine.stop()
    """

    def __init__(self) -> None:
        self.presence = PresenceDetector()
        self.blur = PrivacyBlurEngine()
        self.safety = SafetyMonitor()
        self.analyser = ImageAnalyser()

        self.alert_queue: queue.Queue[VisionAlert] = queue.Queue(maxsize=50)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_result: Optional[DetectionResult] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()
        logger.info("VisionEngine started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("VisionEngine stopped.")

    def get_privacy_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame with privacy blur applied if peeker detected."""
        frame = self.latest_frame
        result = self.latest_result
        if frame is None:
            return None
        if result and result.peeker_detected:
            frame = self.blur.apply_blur(frame)
            frame = self.blur.overlay_privacy_banner(frame)
        return frame

    # ------------------------------------------------------------------
    # Camera loop (background thread)
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not cap.isOpened():
            logger.error("Cannot open camera index %d.", CAMERA_INDEX)
            self._running = False
            return

        frame_interval = 1.0 / CAMERA_FPS

        while self._running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera frame read failed; retrying…")
                time.sleep(0.1)
                continue

            self.latest_frame = frame

            # Run presence detection
            result = self.presence.detect(frame)
            self.latest_result = result

            # Peeker alert
            if self.presence.should_alert(result):
                alert = VisionAlert(
                    alert_type="peeker",
                    message=f"Shoulder-surfer detected! {result.num_persons} person(s) in frame.",
                    severity="high",
                    timestamp=time.time(),
                )
                self._push_alert(alert)

            # Fall detection
            fall_alert = self.safety.detect_fall(result.bboxes, frame.shape)
            if fall_alert:
                self._push_alert(fall_alert)

            # Unattended screen check
            unattended = self.safety.check_unattended(result.num_persons)
            if unattended:
                self._push_alert(unattended)

            # Throttle to target FPS
            elapsed = time.perf_counter() - t0
            sleep_s = max(0, frame_interval - elapsed)
            time.sleep(sleep_s)

        cap.release()
        gc.collect()

    def _push_alert(self, alert: VisionAlert) -> None:
        try:
            self.alert_queue.put_nowait(alert)
        except queue.Full:
            pass  # Drop oldest-style: UI can drain the queue


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = VisionEngine()
    engine.start()
    logger.info("Running for 5 seconds…")
    time.sleep(5)
    engine.stop()
    while not engine.alert_queue.empty():
        print("ALERT:", engine.alert_queue.get())
