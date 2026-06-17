"""
k2/src/python/adaptive_optimizer.py

Adaptive Optimization Layer — K2
----------------------------------
Monitors compression performance (ratio, speed) over a sliding window
of recent chunks and dynamically adjusts:

  1. Neural mix weight (alpha) — how much neural vs. statistical prediction
  2. Chunk size — trade latency vs. context quality
  3. Transform selection — which OpenZL graph path to use per segment
  4. Compression level — Zstd level when arithmetic coding is skipped
  5. Fallback triggers — revert to pure Zstd when neural overhead > benefit

Design:
  - Bandit-style selection (UCB1) among a discrete set of "strategy configs"
  - Each strategy is scored by: compression_ratio - latency_penalty
  - Exploration vs. exploitation controlled by a single 'exploration' param
  - Thread-safe: uses a lock around shared state for concurrent chunk workers
  - Plugs into OpenZL via its compressor-graph control points

The optimizer is stateful and improves with more data.  On first run it
explores; over time it exploits the best-performing strategy.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from structure_discovery import StructureHint, DataClass
from hybrid_predictor import HybridConfig, PredictorMode


# ---------------------------------------------------------------------------
# Strategy definition
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    predictor_mode: PredictorMode
    alpha: float           # neural mix weight
    chunk_size: int        # bytes per chunk
    zstd_level: int        # Zstd fallback level (1–22)
    transforms: list[str]  # ordered transform chain (OpenZL graph path)

    # UCB1 bookkeeping
    _n_plays: int = field(default=0, repr=False)
    _total_score: float = field(default=0.0, repr=False)

    @property
    def mean_score(self) -> float:
        return self._total_score / self._n_plays if self._n_plays else 0.0

    def ucb1_score(self, total_plays: int, exploration: float) -> float:
        if self._n_plays == 0:
            return float("inf")
        return self.mean_score + exploration * math.sqrt(
            math.log(total_plays + 1) / self._n_plays
        )

    def record(self, score: float) -> None:
        self._n_plays += 1
        self._total_score += score


# ---------------------------------------------------------------------------
# Built-in strategy set (expanded per data class)
# ---------------------------------------------------------------------------

def _default_strategies_for(hint: StructureHint) -> list[Strategy]:
    """
    Generate a relevant strategy set tailored to the structure hint.
    The AdaptiveOptimizer will also add generic fallbacks.
    """
    strategies: list[Strategy] = []
    cls = hint.data_class
    w = hint.element_size

    if cls == DataClass.TIMESERIES:
        strategies += [
            Strategy("ts_cma_light",  PredictorMode.LINEAR_CMA, 0.3, 4096,  3,
                     [f"bytedelta-{w}", "zigzag", "zstd"]),
            Strategy("ts_cma_heavy",  PredictorMode.LINEAR_CMA, 0.5, 2048,  6,
                     [f"bytedelta-{w}", "zigzag", "zstd"]),
            Strategy("ts_lstm",       PredictorMode.LINEAR_CMA, 0.6, 1024,  3,
                     [f"bytedelta-{w}", "zstd"]),
        ]
    elif cls == DataClass.FLOAT_ARRAY:
        strategies += [
            Strategy("fp_split_delta",       PredictorMode.LINEAR_CMA, 0.25, 8192, 3,
                     [f"bytesplit-{w}", f"bytedelta-{w}", "zstd"]),
            Strategy("fp_delta_only",       PredictorMode.LINEAR_CMA, 0.30, 4096, 3,
                     [f"bytedelta-{w}", "zstd"]),
            Strategy("fp_pure_zstd",        PredictorMode.PASSTHROUGH, 0.0, 4096, 9,
                     ["zstd"]),
        ]
    elif cls == DataClass.COLUMNAR:
        s = hint.column_stride or 8
        strategies += [
            Strategy("col_cma",       PredictorMode.LINEAR_CMA, 0.35, 4096, 3,
                     [f"columnsplit-{s}", "bytedelta-4", "zstd"]),
            Strategy("col_pure_zstd", PredictorMode.PASSTHROUGH, 0.0, 4096, 9,
                     [f"columnsplit-{s}", "zstd"]),
        ]
    elif cls == DataClass.TEXT:
        strategies += [
            Strategy("text_cma",      PredictorMode.LINEAR_CMA, 0.45, 2048, 3,
                     ["bwt", "mtf", "zstd"]),
            Strategy("text_zstd",     PredictorMode.PASSTHROUGH, 0.0, 4096, 9,
                     ["zstd"]),
        ]

    # Always include a pure Zstd fallback
    strategies.append(
        Strategy("zstd_fallback",     PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                 ["zstd"])
    )
    return strategies


# ---------------------------------------------------------------------------
# Performance window
# ---------------------------------------------------------------------------

@dataclass
class _ChunkResult:
    strategy_name: str
    input_size: int
    output_size: int
    elapsed_ms: float

    @property
    def ratio(self) -> float:
        return self.input_size / max(self.output_size, 1)

    @property
    def throughput_mbs(self) -> float:
        return (self.input_size / 1e6) / max(self.elapsed_ms / 1000, 1e-9)

    def score(self, latency_weight: float = 0.1) -> float:
        """
        Composite score: high ratio is good; low throughput is penalized.
        latency_weight in [0, 1]: 0 = ratio only; 1 = speed only.
        """
        ratio_score = math.log2(max(self.ratio, 1.0))
        speed_score = math.log2(max(self.throughput_mbs, 0.1))
        return (1.0 - latency_weight) * ratio_score + latency_weight * speed_score


# ---------------------------------------------------------------------------
# Adaptive Optimizer
# ---------------------------------------------------------------------------

class AdaptiveOptimizer:
    """
    Selects and tunes compression strategies using UCB1 bandit.

    Parameters
    ----------
    hint : StructureHint
        Output from StructureDiscovery.analyze().
    exploration : float
        UCB1 exploration constant.  Higher → more exploration.
        Typical range: 0.5–2.0.  Default: 1.0.
    latency_weight : float
        How much to penalize slow strategies (0=ignore speed, 1=speed only).
    window : int
        Sliding window of recent results used for adaptive re-weighting.
    budget_ms : float or None
        If set, strategies taking longer than this per chunk are soft-penalized.
    """

    def __init__(
        self,
        hint: Optional[StructureHint] = None,
        exploration: float = 1.0,
        latency_weight: float = 0.15,
        window: int = 50,
        budget_ms: Optional[float] = None,
        extra_strategies: Optional[list[Strategy]] = None,
    ):
        self._hint = hint
        self._exploration = exploration
        self._latency_weight = latency_weight
        self._window: deque[_ChunkResult] = deque(maxlen=window)
        self._budget_ms = budget_ms
        self._lock = threading.Lock()
        self._total_plays = 0

        # Build strategy pool
        base = _default_strategies_for(hint) if hint else []
        extra = extra_strategies or []
        self._strategies: dict[str, Strategy] = {
            s.name: s for s in base + extra
        }
        if not self._strategies:
            # Minimal fallback
            self._strategies["zstd_fallback"] = Strategy(
                "zstd_fallback", PredictorMode.PASSTHROUGH, 0.0, 4096, 3, ["zstd"]
            )

        # Current selection
        self._current: Optional[Strategy] = None

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def select_strategy(self) -> Strategy:
        """UCB1 strategy selection.  Call before compressing each chunk."""
        with self._lock:
            best = max(
                self._strategies.values(),
                key=lambda s: s.ucb1_score(self._total_plays, self._exploration),
            )
            self._current = best
            return best

    def record_result(
        self,
        strategy_name: str,
        input_size: int,
        output_size: int,
        elapsed_ms: float,
    ) -> None:
        """Call after each chunk is compressed to update bandit statistics."""
        result = _ChunkResult(strategy_name, input_size, output_size, elapsed_ms)
        score = result.score(self._latency_weight)

        # Budget penalty
        if self._budget_ms and elapsed_ms > self._budget_ms:
            penalty = (elapsed_ms - self._budget_ms) / self._budget_ms * 0.5
            score = max(0.0, score - penalty)

        with self._lock:
            self._total_plays += 1
            self._window.append(result)
            if strategy_name in self._strategies:
                self._strategies[strategy_name].record(score)

    # ------------------------------------------------------------------
    # HybridConfig factory
    # ------------------------------------------------------------------

    def get_config(self, strategy: Strategy) -> HybridConfig:
        """Convert a Strategy to a HybridConfig for the HybridPredictor."""
        return HybridConfig(
            mode=strategy.predictor_mode,
            alpha=strategy.alpha,
            chunk_size=strategy.chunk_size,
            use_arithmetic=(strategy.predictor_mode != PredictorMode.PASSTHROUGH),
        )

    # ------------------------------------------------------------------
    # Monitoring / reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return current bandit statistics for all strategies."""
        with self._lock:
            return {
                name: {
                    "plays":      s._n_plays,
                    "mean_score": round(s.mean_score, 4),
                    "ucb1":       round(
                        s.ucb1_score(self._total_plays, self._exploration), 4
                    ),
                }
                for name, s in self._strategies.items()
            }

    def best_strategy(self) -> Strategy:
        """Return the strategy with the highest mean score (exploitation only)."""
        with self._lock:
            played = [s for s in self._strategies.values() if s._n_plays > 0]
            if not played:
                return next(iter(self._strategies.values()))
            return max(played, key=lambda s: s.mean_score)

    def recent_ratio(self) -> float:
        """Average compression ratio over the sliding window."""
        with self._lock:
            if not self._window:
                return 1.0
            return sum(r.ratio for r in self._window) / len(self._window)

    def recent_throughput_mbs(self) -> float:
        """Average throughput (MB/s) over the sliding window."""
        with self._lock:
            if not self._window:
                return 0.0
            return sum(r.throughput_mbs for r in self._window) / len(self._window)

    # ------------------------------------------------------------------
    # Dynamic alpha tuning (gradient-free)
    # ------------------------------------------------------------------

    def tune_alpha(self, strategy: Strategy, step: float = 0.02) -> float:
        """
        Perturb alpha ±step based on recent performance trend.
        Returns the new alpha.  Call periodically (e.g., every 10 chunks).
        """
        with self._lock:
            if len(self._window) < 4:
                return strategy.alpha

            recent = list(self._window)[-4:]
            scores = [r.score(self._latency_weight) for r in recent]
            trend = scores[-1] - scores[0]  # positive → improving

            if trend > 0.05:
                new_alpha = min(1.0, strategy.alpha + step)
            elif trend < -0.05:
                new_alpha = max(0.0, strategy.alpha - step)
            else:
                new_alpha = strategy.alpha

            strategy.alpha = new_alpha
            return new_alpha

    # ------------------------------------------------------------------
    # Integration hook: timed compress wrapper
    # ------------------------------------------------------------------

    def compress_with_timing(
        self,
        data: bytes,
        compress_fn: Callable[[bytes, Strategy], bytes],
    ) -> tuple[bytes, str]:
        """
        Select a strategy, run compress_fn under timing, record result.
        Returns (compressed_bytes, strategy_name).

        Example:
            optimizer = AdaptiveOptimizer(hint)
            compressed, strat_name = optimizer.compress_with_timing(
                chunk,
                lambda data, strat: predictor.compress_chunk(data),
            )
        """
        strategy = self.select_strategy()
        t0 = time.perf_counter()
        compressed = compress_fn(data, strategy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.record_result(
            strategy.name,
            len(data),
            len(compressed),
            elapsed_ms,
        )
        return compressed, strategy.name


# ---------------------------------------------------------------------------
# Integrated pipeline: Discovery → Optimizer → Predictor
# ---------------------------------------------------------------------------

class K2Pipeline:
    """
    K2 — End-to-end compression pipeline.

    1. Calls StructureDiscovery on input data header
    2. Initialises AdaptiveOptimizer with appropriate strategies
    3. Compresses each chunk using HybridPredictor under bandit control
    4. Returns compressed stream + metadata for decompressor

    This is the main class to instantiate when integrating with
    the OpenZL C++ back-end via ctypes / cffi / pybind11.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        exploration: float = 1.0,
        latency_weight: float = 0.15,
        probe_bytes: int = 128 * 1024,
    ):
        from structure_discovery import StructureDiscovery
        from hybrid_predictor import HybridPredictor

        self._discovery = StructureDiscovery(onnx_model_path, probe_bytes)
        self._exploration = exploration
        self._latency_weight = latency_weight
        self._optimizer: Optional[AdaptiveOptimizer] = None
        self._predictor: Optional[HybridPredictor] = None
        self._hint: Optional[StructureHint] = None

    def prepare(self, sample: bytes) -> StructureHint:
        """
        Analyse a data sample to set up the strategy pool.
        Call once before compressing a new stream.
        """
        from hybrid_predictor import HybridPredictor

        self._hint = self._discovery.analyze(sample)
        self._optimizer = AdaptiveOptimizer(
            hint=self._hint,
            exploration=self._exploration,
            latency_weight=self._latency_weight,
        )
        # Warm up all strategies with one synthetic play each so the
        # bandit has a baseline to exploit from the very first chunk
        for strat in self._optimizer._strategies.values():
            self._optimizer.record_result(strat.name, 4096, 4096, 1.0)

        init_strat = self._optimizer.select_strategy()
        cfg = self._optimizer.get_config(init_strat)
        self._predictor = HybridPredictor(cfg)
        self._predictor.train(sample[:min(len(sample), 32768)])
        return self._hint

    def compress(self, data: bytes) -> bytes:
        """Compress data using the full neural pipeline."""
        if self._optimizer is None or self._predictor is None:
            self.prepare(data[:min(len(data), 65536)])

        strategy = self._optimizer.select_strategy()
        cfg = self._optimizer.get_config(strategy)
        self._predictor._cfg = cfg

        t0 = time.perf_counter()
        compressed = self._predictor.compress(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._optimizer.record_result(
            strategy.name, len(data), len(compressed), elapsed_ms
        )

        # Periodic alpha tuning every 10 plays
        if strategy._n_plays % 10 == 0:
            self._optimizer.tune_alpha(strategy)

        return compressed

    def stats(self) -> dict:
        if self._optimizer is None:
            return {}
        return {
            "hint":          str(self._hint.data_class.name) if self._hint else None,
            "recent_ratio":  round(self._optimizer.recent_ratio(), 3),
            "throughput_mbs": round(self._optimizer.recent_throughput_mbs(), 1),
            "strategies":    self._optimizer.stats(),
            "best_strategy": self._optimizer.best_strategy().name,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from structure_discovery import StructureDiscovery

    rng = np.random.default_rng(99)

    # Simulate a time-series workload
    ts_data = np.cumsum(rng.integers(0, 200, size=16384)).astype("<u8").tobytes()

    pipeline = K2Pipeline(exploration=1.5, latency_weight=0.1)
    hint = pipeline.prepare(ts_data[:8192])
    print(f"Detected: {hint.data_class.name}, "
          f"elem={hint.element_size}, "
          f"Δgain={hint.delta_gain_db:.1f}dB, "
          f"transforms={hint.suggested_transforms}")

    # Simulate 20 chunks arriving in a stream
    chunk_size = 4096
    total_in = total_out = 0
    for i in range(0, len(ts_data), chunk_size):
        chunk = ts_data[i: i + chunk_size]
        compressed = pipeline.compress(chunk)
        total_in += len(chunk)
        total_out += len(compressed)

    print(f"\nOverall ratio: {total_in/total_out:.2f}x "
          f"({total_in} → {total_out} bytes)")
    import json
    print("\nOptimizer stats:")
    print(json.dumps(pipeline.stats(), indent=2))
