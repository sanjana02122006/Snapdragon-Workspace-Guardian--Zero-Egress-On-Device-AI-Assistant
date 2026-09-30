"""
src/npu_benchmarker.py
======================
Hardware Benchmarking Harness for Snapdragon X-Series NPU / GPU / CPU.

Measures:
  - Inference latency (ms) per model run
  - Time-To-First-Token (TTFT) for autoregressive LLMs
  - RAM footprint (MB) before and after inference
  - Active ONNX Execution Provider (NPU / GPU / CPU)
  - Throughput (tokens/s or frames/s)

All results are emitted as structured JSON-Lines to BENCHMARK_LOG_PATH
and exposed as a live-query API for the PyQt6 telemetry panel.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

# Optional: qai_hub SDK telemetry (graceful fallback if not installed)
try:
    import qai_hub as hub  # type: ignore
    QAI_HUB_AVAILABLE = True
except ImportError:
    QAI_HUB_AVAILABLE = False

from config.settings import (
    BENCHMARK_LOG_PATH,
    ONNX_EP_PRIORITY,
    WARMUP_RUNS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single inference benchmark record."""
    model_name: str
    provider: str                        # Active EP: QNN | DML | CPU
    run_index: int
    latency_ms: float
    ttft_ms: Optional[float] = None      # Only for autoregressive models
    ram_before_mb: float = 0.0
    ram_after_mb: float = 0.0
    ram_delta_mb: float = 0.0
    throughput: Optional[float] = None   # tokens/s or frames/s
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Aggregated statistics across N runs."""
    model_name: str
    provider: str
    num_runs: int
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    mean_ttft_ms: Optional[float] = None
    mean_ram_delta_mb: float = 0.0
    mean_throughput: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            f"[{self.model_name}] Provider={self.provider}  N={self.num_runs}",
            f"  Latency  → mean={self.mean_latency_ms:.1f}ms  "
            f"min={self.min_latency_ms:.1f}ms  max={self.max_latency_ms:.1f}ms  "
            f"p95={self.p95_latency_ms:.1f}ms",
        ]
        if self.mean_ttft_ms is not None:
            lines.append(f"  TTFT     → {self.mean_ttft_ms:.1f} ms")
        if self.mean_throughput is not None:
            lines.append(f"  Throughput → {self.mean_throughput:.1f} units/s")
        lines.append(f"  RAM delta → {self.mean_ram_delta_mb:.1f} MB")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_ram_mb() -> float:
    """Returns current process RSS in MB."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1_048_576


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    f, c = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


@contextmanager
def _ram_delta():
    """Context manager yielding a dict populated with before/after/delta RAM."""
    info: dict[str, float] = {}
    info["before"] = _process_ram_mb()
    try:
        yield info
    finally:
        info["after"] = _process_ram_mb()
        info["delta"] = info["after"] - info["before"]


# ---------------------------------------------------------------------------
# Core benchmarker
# ---------------------------------------------------------------------------

class NPUBenchmarker:
    """
    Wraps any callable inference function and collects timing + hardware stats.

    Usage
    -----
    bench = NPUBenchmarker(model_name="Whisper-Base-En", provider="QNNExecutionProvider")

    # Benchmark a generic callable
    summary = bench.run(
        fn=my_transcribe_fn,
        fn_args=(audio_array,),
        n_runs=10,
    )
    print(summary)
    """

    def __init__(
        self,
        model_name: str,
        provider: str = "QNNExecutionProvider",
        log_path: str = BENCHMARK_LOG_PATH,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._results: list[BenchmarkResult] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        fn: Callable[..., Any],
        fn_args: tuple = (),
        fn_kwargs: dict | None = None,
        n_runs: int = 10,
        warmup: int = WARMUP_RUNS,
        ttft_sentinel: Optional[Callable[[], float]] = None,
    ) -> BenchmarkSummary:
        """
        Execute `fn` n_runs times after `warmup` discard runs.

        Parameters
        ----------
        fn            : The inference callable to benchmark.
        fn_args       : Positional arguments forwarded to fn.
        fn_kwargs     : Keyword arguments forwarded to fn.
        n_runs        : Measured iterations.
        warmup        : Throw-away warm-up iterations.
        ttft_sentinel : Optional callable that returns the TTFT (ms)
                        immediately after fn completes (LLM streaming hook).
        """
        fn_kwargs = fn_kwargs or {}
        results: list[BenchmarkResult] = []

        logger.info(
            "Benchmarking '%s' on %s: %d warmup + %d measured runs",
            self.model_name, self.provider, warmup, n_runs,
        )

        # Warm-up
        for _ in range(warmup):
            try:
                fn(*fn_args, **fn_kwargs)
            except Exception as exc:
                logger.warning("Warmup run failed: %s", exc)
            gc.collect()

        # Measured runs
        for i in range(n_runs):
            with _ram_delta() as ram_info:
                t0 = time.perf_counter()
                try:
                    fn(*fn_args, **fn_kwargs)
                except Exception as exc:
                    logger.error("Benchmark run %d failed: %s", i, exc)
                    continue
                t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            ttft_ms = ttft_sentinel() if ttft_sentinel else None

            result = BenchmarkResult(
                model_name=self.model_name,
                provider=self.provider,
                run_index=i,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                ram_before_mb=ram_info["before"],
                ram_after_mb=ram_info["after"],
                ram_delta_mb=ram_info["delta"],
            )
            results.append(result)
            self._append_log(result)
            gc.collect()

        summary = self._summarize(results)
        logger.info("\n%s", summary)
        with self._lock:
            self._results.extend(results)
        return summary

    def measure_ttft(
        self,
        generate_fn: Callable[..., Any],
        fn_args: tuple = (),
        fn_kwargs: dict | None = None,
    ) -> float:
        """
        Measure Time-To-First-Token for streaming / generator-based LLMs.
        `generate_fn` must be a generator yielding token strings.
        Returns TTFT in milliseconds.
        """
        fn_kwargs = fn_kwargs or {}
        t0 = time.perf_counter()
        gen = generate_fn(*fn_args, **fn_kwargs)
        try:
            next(gen)  # block until first token
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        except StopIteration:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        return ttft_ms

    def latest_results(self) -> list[dict[str, Any]]:
        """Return all collected BenchmarkResult dicts (thread-safe)."""
        with self._lock:
            return [r.to_dict() for r in self._results]

    def live_stats(self) -> dict[str, Any]:
        """
        Snapshot of current hardware utilisation for the telemetry panel.
        Returns NPU load % (via qai_hub if available), CPU %, RAM MB.
        """
        stats: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": _process_ram_mb(),
            "ram_total_mb": psutil.virtual_memory().total / 1_048_576,
        }

        if QAI_HUB_AVAILABLE:
            try:
                device_info = hub.get_device()
                stats["npu_load_percent"] = getattr(device_info, "npu_utilization", "N/A")
            except Exception:
                stats["npu_load_percent"] = "N/A"
        else:
            stats["npu_load_percent"] = "N/A (qai_hub not installed)"

        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _summarize(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        if not results:
            return BenchmarkSummary(
                model_name=self.model_name, provider=self.provider,
                num_runs=0, mean_latency_ms=0, min_latency_ms=0,
                max_latency_ms=0, p95_latency_ms=0,
            )
        latencies = [r.latency_ms for r in results]
        ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
        rams = [r.ram_delta_mb for r in results]
        throughs = [r.throughput for r in results if r.throughput is not None]

        return BenchmarkSummary(
            model_name=self.model_name,
            provider=self.provider,
            num_runs=len(results),
            mean_latency_ms=sum(latencies) / len(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p95_latency_ms=_percentile(latencies, 95),
            mean_ttft_ms=(sum(ttfts) / len(ttfts)) if ttfts else None,
            mean_ram_delta_mb=sum(rams) / len(rams),
            mean_throughput=(sum(throughs) / len(throughs)) if throughs else None,
        )

    def _append_log(self, result: BenchmarkResult) -> None:
        """Append a JSON-Lines record to the benchmark log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Failed to write benchmark log: %s", exc)


# ---------------------------------------------------------------------------
# Convenience: detect active ONNX execution provider
# ---------------------------------------------------------------------------

def detect_active_provider(session) -> str:
    """
    Given an ort.InferenceSession, return the first active EP name.
    Falls back to 'CPUExecutionProvider' if detection fails.
    """
    try:
        providers = session.get_providers()
        for preferred in ONNX_EP_PRIORITY:
            if preferred in providers:
                return preferred
    except Exception:
        pass
    return "CPUExecutionProvider"


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def _dummy_inference():
        """Simulate 10 ms of work."""
        time.sleep(0.010)
        return [math.sqrt(i) for i in range(10_000)]

    bench = NPUBenchmarker(model_name="dummy-model", provider="CPUExecutionProvider")
    summary = bench.run(fn=_dummy_inference, n_runs=5, warmup=2)
    print(summary)
    print("\nLive stats:", bench.live_stats())
