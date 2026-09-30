"""
src/audio_engine.py
===================
On-Device Voice & Audio Module

Handles:
  1. Voice Activity Detection (VAD) via Silero-VAD ONNX
  2. Local Speech-To-Text via Whisper-Base (ONNX, QNN EP)
  3. Voice Command Parser – maps transcribed phrases to app actions
  4. Local Text-To-Speech via CoquiTTS (fully offline)
  5. Audio Alert playback (local WAV files)

All processing is 100% offline. No audio is streamed to any cloud endpoint.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Audio I/O
try:
    import sounddevice as sd  # type: ignore
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    logging.warning("sounddevice not installed; microphone input disabled.")

try:
    import soundfile as sf  # type: ignore
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False

# ONNX Runtime
try:
    import onnxruntime as ort  # type: ignore
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import onnxruntime_extensions as ort_ext
    ORT_EXT_AVAILABLE = True
except ImportError:
    ORT_EXT_AVAILABLE = False

# CoquiTTS
try:
    from TTS.api import TTS as CoquiTTS  # type: ignore
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    logging.warning("CoquiTTS not installed; TTS disabled.")

from config.settings import (
    AUDIO_ALERT_FILE,
    DML_EP_OPTIONS,
    MODELS_DIR,
    ONNX_EP_PRIORITY,
    QNN_EP_OPTIONS,
    STT_CHUNK_DURATION_S,
    STT_SAMPLE_RATE,
    TTS_MODEL_NAME,
    VAD_SILENCE_THRESHOLD_S,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language: str
    inference_ms: float
    is_final: bool = True


@dataclass
class VoiceCommand:
    raw_text: str
    intent: str            # Mapped intent string
    args: dict             # Extracted arguments
    confidence: float


# ---------------------------------------------------------------------------
# ONNX Session helper (shared with vision_engine pattern)
# ---------------------------------------------------------------------------

def _build_ort_session(model_path: str) -> Optional["ort.InferenceSession"]:
    if not ORT_AVAILABLE or not Path(model_path).exists():
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    eps: list = []
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
        logger.error("ORT session creation failed for %s: %s", model_path, exc)
        return None


# ---------------------------------------------------------------------------
# Voice Activity Detector (Silero-VAD ONNX)
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """
    Lightweight ONNX-based VAD to segment microphone stream into speech chunks.
    Uses Silero-VAD model (silero_vad.onnx).
    """

    VAD_THRESHOLD = 0.5
    WINDOW_SIZE_SAMPLES = 512  # Must match Silero-VAD input

    def __init__(self) -> None:
        model_path = str(MODELS_DIR / "silero_vad.onnx")
        self.session = _build_ort_session(model_path)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Returns True if the chunk contains speech."""
        if self.session is None:
            # Fallback: energy-based VAD
            return float(np.abs(audio_chunk).mean()) > 0.01

        # Silero-VAD expects (1, chunk_size) float32 at 16 kHz
        chunk = audio_chunk.astype(np.float32)
        if len(chunk) != self.WINDOW_SIZE_SAMPLES:
            chunk = np.resize(chunk, (self.WINDOW_SIZE_SAMPLES,))
        inp = chunk[np.newaxis]  # (1, 512)

        try:
            outs = self.session.run(
                None,
                {"input": inp, "sr": np.array(STT_SAMPLE_RATE, dtype=np.int64),
                 "h": self._h, "c": self._c},
            )
            speech_prob = float(outs[0])
            self._h = outs[1]
            self._c = outs[2]
            return speech_prob > self.VAD_THRESHOLD
        except Exception as exc:
            logger.debug("VAD inference error: %s", exc)
            return False

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)


# ---------------------------------------------------------------------------
# Whisper STT Engine
# ---------------------------------------------------------------------------

class WhisperSTT:
    """
    Offline speech-to-text using Whisper-Base ONNX model.
    Runs on Hexagon NPU via QNN Execution Provider.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        default = str(MODELS_DIR / "whisper_base_en.onnx")
        self.model_path = model_path or default
        self.encoder_session = _build_ort_session(self.model_path.replace(".onnx", "_encoder.onnx"))
        self.decoder_session = _build_ort_session(self.model_path.replace(".onnx", "_decoder.onnx"))
        self.vad = VoiceActivityDetector()
        # Simple token map stub – replace with full Whisper tokenizer
        self._language = "en"

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a float32 audio array (16kHz mono) to text.

        Parameters
        ----------
        audio : np.ndarray
            Float32 audio samples in range [-1, 1] at 16 kHz mono.
        """
        t0 = time.perf_counter()

        if self.encoder_session is None:
            return TranscriptionResult(
                text="[Whisper model not loaded – place whisper_base_en.onnx in models/]",
                confidence=0.0, language="en",
                inference_ms=(time.perf_counter() - t0) * 1000,
            )

        # Compute log-Mel spectrogram (80 mel bins, 3000 frames)
        mel = self._compute_log_mel(audio)

        try:
            inp_name = self.encoder_session.get_inputs()[0].name
            encoder_out = self.encoder_session.run(None, {inp_name: mel})
            # Decoder pass (autoregressive) – stub for ONNX split model
            text = self._greedy_decode(encoder_out[0])
        except Exception as exc:
            logger.error("Whisper inference failed: %s", exc)
            text = "[STT error]"

        inf_ms = (time.perf_counter() - t0) * 1000.0
        return TranscriptionResult(
            text=text.strip(),
            confidence=0.9,
            language=self._language,
            inference_ms=inf_ms,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_log_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute 80-band log-Mel spectrogram matching Whisper preprocessing.
        Returns shape (1, 80, 3000) float32.
        """
        import librosa  # type: ignore (optional dep)
        n_fft = 400
        hop = 160
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=STT_SAMPLE_RATE,
            n_fft=n_fft, hop_length=hop, n_mels=80, fmax=8000,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        # Pad / truncate to 3000 frames
        if log_mel.shape[1] < 3000:
            pad = 3000 - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), constant_values=-80.0)
        else:
            log_mel = log_mel[:, :3000]
        return log_mel[np.newaxis].astype(np.float32)  # (1, 80, 3000)

    def _greedy_decode(self, encoder_hidden: np.ndarray) -> str:
        """
        Greedy decoding loop using the Whisper decoder ONNX session and onnxruntime_extensions.
        """
        if self.decoder_session is None:
            return "[Decoder model missing - please ensure whisper_base_en_decoder.onnx is downloaded]"
        
        try:
            from transformers import WhisperTokenizer
            tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base.en")
            
            # Start token for English transcription
            decoder_input_ids = np.array([[tokenizer.sot_token_id, tokenizer.encode("<|en|>")[0], tokenizer.encode("<|transcribe|>")[0]]], dtype=np.int32)
            
            # Simplified generation loop
            max_tokens = 50
            for _ in range(max_tokens):
                dec_out = self.decoder_session.run(None, {
                    "encoder_hidden_states": encoder_hidden,
                    "input_ids": decoder_input_ids
                })
                logits = dec_out[0]
                next_token = np.argmax(logits[:, -1, :], axis=-1)[0]
                if next_token == tokenizer.eot_token_id:
                    break
                decoder_input_ids = np.append(decoder_input_ids, [[next_token]], axis=1)
                
            return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error("Decoder error: %s", e)
            return "[Decoder error]"


# ---------------------------------------------------------------------------
# Microphone Listener (background thread)
# ---------------------------------------------------------------------------

class MicrophoneListener:
    """
    Continuously reads from the default microphone, runs VAD, and
    pushes complete utterances to `utterance_queue`.
    """

    def __init__(self) -> None:
        self.vad = VoiceActivityDetector()
        self.utterance_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer: list[np.ndarray] = []
        self._silence_frames = 0
        self._max_silence = int(VAD_SILENCE_THRESHOLD_S / STT_CHUNK_DURATION_S)

    def start(self) -> None:
        if not SD_AVAILABLE:
            logger.error("sounddevice not installed; mic listener cannot start.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("MicrophoneListener started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _listen_loop(self) -> None:
        chunk_samples = int(STT_SAMPLE_RATE * STT_CHUNK_DURATION_S)
        with sd.InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:
            while self._running:
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk[:, 0]  # mono

                if self.vad.is_speech(chunk):
                    self._buffer.append(chunk)
                    self._silence_frames = 0
                else:
                    if self._buffer:
                        self._silence_frames += 1
                        self._buffer.append(chunk)
                        if self._silence_frames >= self._max_silence:
                            utterance = np.concatenate(self._buffer)
                            try:
                                self.utterance_queue.put_nowait(utterance)
                            except queue.Full:
                                pass
                            self._buffer = []
                            self._silence_frames = 0
                            self.vad.reset()


# ---------------------------------------------------------------------------
# Voice Command Parser
# ---------------------------------------------------------------------------

INTENT_PATTERNS: list[tuple[str, str, dict]] = [
    # (regex, intent, default_args)
    (r"open\s+(?P<tab>document|chat|vision|settings|benchmark)", "open_tab", {}),
    (r"(blur|hide)\s+screen", "privacy_blur", {}),
    (r"(stop|disable)\s+blur", "privacy_unblur", {}),
    (r"summarize\s+(?P<doc>.+)", "summarize_doc", {}),
    (r"search\s+for\s+(?P<query>.+)", "rag_search", {}),
    (r"(ask|question)\s+(?P<question>.+)", "llm_qa", {}),
    (r"run\s+benchmark", "benchmark", {}),
    (r"(help|what can you do)", "help", {}),
    (r"(exit|quit|close)", "quit", {}),
]


class VoiceCommandParser:
    """Maps raw transcribed text to structured VoiceCommand intents."""

    def parse(self, text: str) -> Optional[VoiceCommand]:
        normalized = text.lower().strip()
        for pattern, intent, defaults in INTENT_PATTERNS:
            m = re.search(pattern, normalized)
            if m:
                args = {**defaults, **m.groupdict()}
                return VoiceCommand(
                    raw_text=text,
                    intent=intent,
                    args={k: v for k, v in args.items() if v is not None},
                    confidence=0.85,
                )
        return VoiceCommand(raw_text=text, intent="unknown", args={}, confidence=0.0)


# ---------------------------------------------------------------------------
# Text-To-Speech Engine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Offline TTS using CoquiTTS (Tacotron2-DDC + HiFiGAN vocoder).
    Synthesises speech locally and plays it through the default audio device.
    """

    def __init__(self) -> None:
        self._tts: Optional[CoquiTTS] = None
        self._ready = False
        if COQUI_AVAILABLE:
            self._load_thread = threading.Thread(target=self._load_model, daemon=True)
            self._load_thread.start()
        else:
            logger.warning("CoquiTTS not installed; TTS disabled.")

    def _load_model(self) -> None:
        try:
            self._tts = CoquiTTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=False)
            self._ready = True
            logger.info("CoquiTTS model loaded: %s", TTS_MODEL_NAME)
        except Exception as exc:
            logger.error("CoquiTTS load failed: %s", exc)

    def speak(self, text: str, blocking: bool = False) -> None:
        """Convert `text` to speech and play on default output device."""
        if not self._ready or self._tts is None:
            logger.warning("TTS not ready; cannot speak: %s", text[:80])
            return
        def _do():
            try:
                audio_array = self._tts.tts(text=text)
                audio_np = np.array(audio_array, dtype=np.float32)
                if SD_AVAILABLE:
                    sd.play(audio_np, samplerate=22050, blocking=True)
                elif SF_AVAILABLE:
                    tmp = Path("tts_output.wav")
                    self._tts.tts_to_file(text=text, file_path=str(tmp))
                    logger.info("TTS written to %s", tmp)
            except Exception as exc:
                logger.error("TTS speak error: %s", exc)

        if blocking:
            _do()
        else:
            threading.Thread(target=_do, daemon=True).start()

    @staticmethod
    def play_alert(wav_path: str = AUDIO_ALERT_FILE) -> None:
        """Play a local WAV alert sound asynchronously."""
        def _play():
            if not SF_AVAILABLE or not SD_AVAILABLE:
                logger.warning("soundfile/sounddevice not available for alert playback.")
                return
            if not Path(wav_path).exists():
                logger.warning("Alert WAV not found: %s", wav_path)
                return
            try:
                data, sr = sf.read(wav_path, dtype="float32")
                sd.play(data, samplerate=sr, blocking=True)
            except Exception as exc:
                logger.error("Alert playback failed: %s", exc)
        threading.Thread(target=_play, daemon=True).start()


# ---------------------------------------------------------------------------
# Audio Engine Orchestrator
# ---------------------------------------------------------------------------

class AudioEngine:
    """
    Top-level audio pipeline orchestrator.
    Continuously listens for voice commands and routes them to callbacks.

    Usage
    -----
    def handle_cmd(cmd: VoiceCommand):
        print("Intent:", cmd.intent, cmd.args)

    engine = AudioEngine(on_command=handle_cmd)
    engine.start()
    """

    def __init__(self, on_command: Optional[Callable[[VoiceCommand], None]] = None) -> None:
        self.stt = WhisperSTT()
        self.mic = MicrophoneListener()
        self.parser = VoiceCommandParser()
        self.tts = TTSEngine()
        self.on_command = on_command

        self._running = False
        self._process_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.mic.start()
        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        logger.info("AudioEngine started.")

    def stop(self) -> None:
        self._running = False
        self.mic.stop()
        if self._process_thread:
            self._process_thread.join(timeout=3.0)
        logger.info("AudioEngine stopped.")

    def speak(self, text: str) -> None:
        """Non-blocking TTS playback."""
        self.tts.speak(text, blocking=False)

    def _process_loop(self) -> None:
        while self._running:
            try:
                utterance = self.mic.utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            result = self.stt.transcribe(utterance)
            logger.info("[STT] %.0f ms | %s", result.inference_ms, result.text)

            if result.text and not result.text.startswith("["):
                cmd = self.parser.parse(result.text)
                if self.on_command and cmd:
                    try:
                        self.on_command(cmd)
                    except Exception as exc:
                        logger.error("Command handler error: %s", exc)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _on_cmd(cmd: VoiceCommand):
        print(f"  → intent={cmd.intent}  args={cmd.args}  conf={cmd.confidence:.2f}")

    parser = VoiceCommandParser()
    test_phrases = [
        "open document tab",
        "summarize quarterly_report.pdf",
        "blur screen",
        "search for privacy settings",
        "ask what is the revenue forecast",
        "run benchmark",
        "exit",
    ]
    for phrase in test_phrases:
        cmd = parser.parse(phrase)
        _on_cmd(cmd)
