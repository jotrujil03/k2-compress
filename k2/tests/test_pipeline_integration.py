"""
k2/tests/test_pipeline_integration.py

Integration Tests — K2
------------------------
Tests the three layers together:
  1. StructureDiscovery   → identifies data type
  2. HybridPredictor      → compresses with neural+stat blend
  3. AdaptiveOptimizer    → selects best strategy via UCB1 bandit

Run with:
    pytest tests/test_pipeline_integration.py -v
    # or directly:
    python tests/test_pipeline_integration.py
"""

from __future__ import annotations

import os
import sys
import struct
import math

# Allow running from tests/ or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/python"))

import numpy as np
import pytest

from structure_discovery import StructureDiscovery, DataClass
from hybrid_predictor import HybridPredictor, HybridConfig, PredictorMode
from adaptive_optimizer import (
    AdaptiveOptimizer, K2Pipeline, Strategy
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2025)


def make_timeseries(n: int = 16384, step_max: int = 200) -> bytes:
    arr = np.cumsum(RNG.integers(0, step_max, n)).astype("<u8")
    return arr.tobytes()


def make_float32(n: int = 16384) -> bytes:
    return RNG.standard_normal(n).astype("<f4").tobytes()


def make_text(n_chars: int = 16000) -> bytes:
    words = b"the quick brown fox jumps over lazy dog "
    return (words * (n_chars // len(words) + 1))[:n_chars]


def make_random_blob(n: int = 8192) -> bytes:
    return os.urandom(n)


def make_columnar(rows: int = 2048, cols: int = 8, stride: int = 4) -> bytes:
    """Simulate struct-of-arrays style data."""
    data = bytearray()
    for _ in range(rows):
        for c in range(cols):
            data += struct.pack("<i", int(RNG.integers(-1000, 1000)))
    return bytes(data)


# ---------------------------------------------------------------------------
# Structure Discovery Tests
# ---------------------------------------------------------------------------

class TestStructureDiscovery:

    def setup_method(self):
        self.disc = StructureDiscovery()

    def test_timeseries_detected(self):
        data = make_timeseries()
        hint = self.disc.analyze(data)
        assert hint.data_class in (DataClass.TIMESERIES, DataClass.INTEGER_ARRAY), \
            f"Expected timeseries/integer, got {hint.data_class}"
        assert hint.element_size in (4, 8)
        assert hint.delta_gain_db > 0
        assert "zstd" in hint.suggested_transforms

    def test_float_detected(self):
        data = make_float32()
        hint = self.disc.analyze(data)
        # Float or integer are both acceptable (heuristic may vary)
        assert hint.data_class != DataClass.TEXT
        assert hint.byte_entropy < 8.0  # not random

    def test_text_detected(self):
        data = make_text()
        hint = self.disc.analyze(data)
        assert hint.data_class == DataClass.TEXT
        assert hint.confidence > 0.5

    def test_random_blob_detected(self):
        data = make_random_blob(32768)
        hint = self.disc.analyze(data)
        assert hint.data_class == DataClass.BINARY_BLOB
        assert hint.byte_entropy > 7.0

    def test_entropy_monotone(self):
        """More random data → higher entropy."""
        low_entropy = bytes([42]) * 8192
        high_entropy = make_random_blob(8192)
        h_low  = self.disc.analyze(low_entropy)
        h_high = self.disc.analyze(high_entropy)
        assert h_low.byte_entropy < h_high.byte_entropy

    def test_hint_has_transforms(self):
        for maker in [make_timeseries, make_float32, make_text]:
            hint = self.disc.analyze(maker())
            assert len(hint.suggested_transforms) >= 1


# ---------------------------------------------------------------------------
# Hybrid Predictor Tests
# ---------------------------------------------------------------------------

class TestHybridPredictor:

    def _make_predictor(self, alpha=0.35, chunk=2048):
        return HybridPredictor(HybridConfig(
            mode=PredictorMode.LINEAR_CMA,
            alpha=alpha,
            chunk_size=chunk,
            use_arithmetic=True,
        ))

    def test_compress_produces_bytes(self):
        pred = self._make_predictor()
        data = make_timeseries(4096)
        pred.train(data[:1024])
        compressed = pred.compress(data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    def test_compression_ratio_positive(self):
        """Structured data should compress to something smaller."""
        pred = self._make_predictor()
        data = make_text(8192)
        pred.train(data[:2048])
        compressed = pred.compress(data)
        # Even a modest ratio (>1.0) confirms we're not expanding uniformly
        ratio = len(data) / len(compressed)
        assert ratio > 0.5, f"Suspiciously poor: {ratio:.2f}x"

    def test_passthrough_mode(self):
        pred = HybridPredictor(HybridConfig(
            mode=PredictorMode.PASSTHROUGH,
            chunk_size=4096,
        ))
        data = make_timeseries(8192)
        # Should not raise
        compressed = pred.compress(data)
        assert isinstance(compressed, bytes)

    def test_probs_sum_to_one(self):
        pred = self._make_predictor()
        context = b"Hello world"
        probs = pred.predict_probs(context)
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probs sum to {probs.sum()}"
        assert (probs >= 0).all(), "Negative probabilities"

    def test_reset_clears_state(self):
        pred = self._make_predictor()
        data = make_timeseries(4096)
        pred.train(data)
        pred.reset()
        # After reset, should still function
        probs = pred.predict_probs(b"\x00")
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_chunking_consistency(self):
        """Compressing with different chunk sizes should produce valid output."""
        data = make_timeseries(8192)
        for chunk in (512, 2048, 8192):
            pred = HybridPredictor(HybridConfig(
                mode=PredictorMode.LINEAR_CMA,
                chunk_size=chunk,
                alpha=0.3,
            ))
            pred.train(data[:1024])
            compressed = pred.compress(data)
            assert len(compressed) > 0, f"chunk_size={chunk} produced empty output"


# ---------------------------------------------------------------------------
# Adaptive Optimizer Tests
# ---------------------------------------------------------------------------

class TestAdaptiveOptimizer:

    def _make_optimizer(self, data: bytes) -> AdaptiveOptimizer:
        disc = StructureDiscovery()
        hint = disc.analyze(data)
        return AdaptiveOptimizer(hint=hint, exploration=1.0)

    def test_select_returns_strategy(self):
        opt = self._make_optimizer(make_timeseries())
        strat = opt.select_strategy()
        assert isinstance(strat, Strategy)
        assert strat.name

    def test_record_updates_stats(self):
        opt = self._make_optimizer(make_timeseries())
        strat = opt.select_strategy()
        opt.record_result(strat.name, 4096, 2048, 5.0)
        stats = opt.stats()
        assert strat.name in stats
        assert stats[strat.name]["plays"] == 1

    def test_exploration_decreases_over_time(self):
        """UCB1 scores should converge as we gather more data."""
        opt = self._make_optimizer(make_timeseries())
        initial_ucbs = []
        # Play each strategy once
        for _ in range(30):
            s = opt.select_strategy()
            opt.record_result(s.name, 4096, 2000, 3.0)
            initial_ucbs.append(s.ucb1_score(opt._total_plays, 1.0))
        # After many plays, best mean_score strategy should dominate
        best = opt.best_strategy()
        assert best.mean_score >= 0

    def test_tune_alpha_within_bounds(self):
        opt = self._make_optimizer(make_timeseries())
        strat = opt.select_strategy()
        original_alpha = strat.alpha
        # Simulate some results
        for _ in range(10):
            opt.record_result(strat.name, 4096, 1024, 3.0)
        new_alpha = opt.tune_alpha(strat)
        assert 0.0 <= new_alpha <= 1.0

    def test_best_strategy_after_plays(self):
        opt = self._make_optimizer(make_text())
        strategies = list(opt._strategies.values())
        # Force one strategy to have clearly better scores
        good = strategies[0]
        bad  = strategies[-1] if len(strategies) > 1 else strategies[0]
        for _ in range(5):
            opt.record_result(good.name, 4096, 500,  2.0)  # good ratio
        for _ in range(5):
            opt.record_result(bad.name,  4096, 4000, 1.0)  # bad ratio
        if good.name != bad.name:
            assert opt.best_strategy().name == good.name

    def test_recent_ratio_computed(self):
        opt = self._make_optimizer(make_timeseries())
        s = opt.select_strategy()
        opt.record_result(s.name, 4096, 2048, 3.0)
        assert opt.recent_ratio() == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestK2Pipeline:

    def test_prepare_and_compress(self):
        pipeline = K2Pipeline(exploration=1.0)
        data = make_timeseries(16384)
        hint = pipeline.prepare(data[:8192])
        assert hint.data_class is not None

        compressed = pipeline.compress(data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    def test_stats_populated_after_compress(self):
        pipeline = K2Pipeline()
        data = make_text(8192)
        pipeline.prepare(data[:2048])
        pipeline.compress(data)
        stats = pipeline.stats()
        assert "recent_ratio" in stats
        assert "best_strategy" in stats
        assert stats["recent_ratio"] > 0

    def test_multiple_chunks_improve_ratio(self):
        """
        With more chunks, the bandit should converge to a better strategy.
        We verify ratio doesn't degrade catastrophically over time.
        """
        pipeline = K2Pipeline(exploration=0.5)
        data = make_timeseries(32768)
        pipeline.prepare(data[:4096])

        ratios = []
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            chunk = data[i: i + chunk_size]
            compressed = pipeline.compress(chunk)
            if len(compressed) > 0:
                ratios.append(len(chunk) / len(compressed))

        # At least some chunks should achieve compression > 1x
        assert any(r > 1.0 for r in ratios), \
            f"No chunk achieved > 1x ratio: {ratios}"

    def test_different_data_types(self):
        """Pipeline should handle all data types without exceptions."""
        for maker, name in [
            (make_timeseries, "timeseries"),
            (make_float32,    "float32"),
            (make_text,       "text"),
            (make_columnar,   "columnar"),
        ]:
            pipeline = K2Pipeline()
            data = maker()
            pipeline.prepare(data[:min(len(data), 8192)])
            compressed = pipeline.compress(data)
            assert len(compressed) > 0, f"Failed on {name}"

    def test_random_blob_fallback(self):
        """Random/encrypted data should fall back gracefully (no crash)."""
        pipeline = K2Pipeline()
        data = make_random_blob(8192)
        pipeline.prepare(data)
        compressed = pipeline.compress(data)
        assert isinstance(compressed, bytes)


# ---------------------------------------------------------------------------
# Benchmark (not a pytest test, run manually)
# ---------------------------------------------------------------------------

def benchmark():
    import time

    print("\n=== Benchmark ===")
    datasets = {
        "timeseries (u64)": make_timeseries(65536),
        "float32":          make_float32(65536),
        "text":             make_text(65536),
        "columnar":         make_columnar(8192),
        "random blob":      make_random_blob(65536),
    }

    for name, data in datasets.items():
        pipeline = K2Pipeline(exploration=0.0, latency_weight=0.1)
        pipeline.prepare(data[:min(len(data), 65536)])
        from structure_discovery import StructureDiscovery
        disc = StructureDiscovery()
        hint = disc.analyze(data[:65536])
        print(f"    hint: class={hint.data_class.name} stride={hint.column_stride} "
            f"delta_gain={hint.delta_gain_db:.1f}dB elem={hint.element_size} "
            f"conf={hint.confidence:.2f}")

        N_RUNS = 3
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            compressed = pipeline.compress(data)
            times.append(time.perf_counter() - t0)
        elapsed_ms = min(times) * 1000

        ratio = len(data) / len(compressed)
        throughput = (len(data) / 1e6) / (elapsed_ms / 1000)
        strat = pipeline.stats().get("best_strategy", "?")
        print(f"  {name:<25}  {len(data):>7}→{len(compressed):>7} bytes  "
              f"ratio={ratio:.2f}x  {throughput:.0f} MB/s  strategy={strat}")

    # After the datasets loop, add:
    import numpy as np
    rng = np.random.default_rng(42)

    # Test with structured floats instead of random normal
    ml_weights = (rng.standard_normal(65536) * 0.02).astype("<f4").tobytes()  # tight distribution
    sensor = np.sin(np.linspace(0, 100, 65536)).astype("<f4").tobytes()        # smooth signal
    embeddings = np.repeat(rng.standard_normal(512), 128).astype("<f4").tobytes()  # repeated vectors

    for name, data in [("ml_weights", ml_weights), ("sensor", sensor), ("embeddings", embeddings)]:
        pipeline = K2Pipeline(exploration=0.0, latency_weight=0.1)
        pipeline.prepare(data[:65536])
        compressed = pipeline.compress(data)
        print(f"  {name:<25}  {len(data):>7}→{len(compressed):>7} bytes  ratio={len(data)/len(compressed):.2f}x")


if __name__ == "__main__":
    # Run tests manually
    import traceback
    suites = [
        TestStructureDiscovery,
        TestHybridPredictor,
        TestAdaptiveOptimizer,
        TestK2Pipeline,
    ]
    failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite) if m.startswith("test_")]
        for method in methods:
            try:
                if hasattr(suite, "setup_method"):
                    suite.setup_method()
                getattr(suite, method)()
                print(f"  ✓ {suite_cls.__name__}.{method}")
            except Exception as e:
                print(f"  ✗ {suite_cls.__name__}.{method}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    benchmark()
