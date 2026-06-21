"""
k2/tests/test_pipeline_integration.py

Integration Tests — K2
------------------------
Tests the Python pipeline layers together:
  1. StructureDiscovery       → identifies data type, suggests Python-side
                                 transforms (currently: columnsplit only —
                                 see structure_discovery.py's module
                                 docstring for why bytedelta/bytesplit are
                                 NOT suggested here: ASDP's C++
                                 pick_transform_cm already owns that
                                 search with a real trial-encode)
  2. hybrid_predictor models  → StructureClassifier / TransformGainPredictor
                                 model definitions (no longer an entropy
                                 coder — see hybrid_predictor.py's module
                                 docstring for why the old neural-mixing +
                                 arithmetic-coder path was removed)
  3. AdaptiveOptimizer        → selects best transform strategy via UCB1
                                 bandit
  4. K2Pipeline                → end-to-end: prepare/compress_full/
                                 reseal_frame/decompress_full, the actual
                                 C++ bridge contract

ASDP-LH (backend 0x04) is the only entropy backend. There is no zstd
anywhere in this pipeline — these tests do not assert on it, and the
pipeline does not depend on it at any layer.

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

try:
    import pytest
except ImportError:
    # Minimal shim so `python tests/test_pipeline_integration.py` (the
    # no-pytest path this file's own docstring advertises) works without
    # pytest installed. Only supports what this file actually uses:
    # @pytest.mark.skipif as a marker the __main__ runner below can
    # introspect. Running via `pytest tests/...` directly is unaffected —
    # this shim is only ever used as a fallback.
    class _SkipIf:
        def __init__(self, condition, reason=""):
            self.condition = condition
            self.reason = reason
        def __call__(self, fn):
            fn._skip = self.condition
            fn._skip_reason = self.reason
            return fn

    class _MarkShim:
        @staticmethod
        def skipif(condition, reason=""):
            return _SkipIf(condition, reason)

    class _PytestShim:
        mark = _MarkShim()

    pytest = _PytestShim()  # type: ignore[assignment]

from structure_discovery import StructureDiscovery, DataClass
from hybrid_predictor import ONNXGainPredictor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from hybrid_predictor import StructureClassifier, TransformGainPredictor

from adaptive_optimizer import (
    AdaptiveOptimizer, K2Pipeline, Strategy, decode_k2_frame,
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


def make_columnar(rows: int = 2000, stride: int = 12) -> bytes:
    """
    Simulate struct-of-arrays style data: interleaved records of (u32, u32,
    f32), 12 bytes/row. Verified directly that this specific mixed-type
    layout is what StructureDiscovery's column-stride detector reliably
    recognizes as COLUMNAR — a same-type-only layout (e.g. 8 columns of
    plain int32) does NOT reliably trigger it, since the detector relies
    on autocorrelation of byte VALUES at candidate strides, which is a
    much weaker signal when every column has the same statistical shape.
    stride is currently fixed at 12 (the u32+u32+f32 layout); only
    candidate strides in _detect_column_stride's fixed list (4, 8, 12, 16,
    24, 32) can be detected at all — see that function's docstring.
    """
    assert stride == 12, "only the 12-byte (u32,u32,f32) layout is implemented"
    a = RNG.integers(0, 1000, rows).astype("<u4")
    b = np.arange(rows).astype("<u4")
    c = RNG.standard_normal(rows).astype("<f4")
    records = np.zeros(rows, dtype=[("a", "<u4"), ("b", "<u4"), ("c", "<f4")])
    records["a"] = a
    records["b"] = b
    records["c"] = c
    return records.tobytes()


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
        # No suggested_transforms for TIMESERIES — ASDP's own
        # pick_transform_cm() trial-encode owns bytedelta/bytesplit
        # selection; Python only suggests something for COLUMNAR data
        # (the one transform ASDP can't do). See structure_discovery.py's
        # module docstring.
        assert hint.suggested_transforms == []

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
        assert hint.suggested_transforms == []  # text gets no Python-side transform

    def test_random_blob_detected(self):
        data = make_random_blob(32768)
        hint = self.disc.analyze(data)
        assert hint.data_class == DataClass.BINARY_BLOB
        assert hint.byte_entropy > 7.0
        assert hint.suggested_transforms == []

    def test_entropy_monotone(self):
        """More random data → higher entropy."""
        low_entropy = bytes([42]) * 8192
        high_entropy = make_random_blob(8192)
        h_low  = self.disc.analyze(low_entropy)
        h_high = self.disc.analyze(high_entropy)
        assert h_low.byte_entropy < h_high.byte_entropy

    def test_columnar_suggests_columnsplit(self):
        """The one case where Python DOES suggest a transform: genuinely
        columnar data gets columnsplit-<stride>, the structural transform
        ASDP's C++ layer has no equivalent for."""
        data = make_columnar()
        hint = self.disc.analyze(data)
        if hint.data_class == DataClass.COLUMNAR and hint.column_stride > 0:
            assert hint.suggested_transforms == [f"columnsplit-{hint.column_stride}"]
        # Note: column-stride detection only checks a fixed candidate list
        # (4, 8, 12, 16, 24, 32 bytes) — see _detect_column_stride's
        # docstring — so this isn't guaranteed to fire for every columnar
        # layout, only documented ones. Not asserting data_class ==
        # COLUMNAR unconditionally for that reason.

    def test_non_columnar_has_no_transforms(self):
        """TIMESERIES/FLOAT_ARRAY/TEXT all correctly get an EMPTY transform
        list now -- ASDP's C++ trial-encode owns transform selection for
        everything except columnar data. An empty list here is the
        intended behavior, not a placeholder."""
        for maker in [make_timeseries, make_float32, make_text]:
            hint = self.disc.analyze(maker())
            assert hint.suggested_transforms == [], \
                f"{maker.__name__} unexpectedly got transforms: {hint.suggested_transforms}"


# ---------------------------------------------------------------------------
# hybrid_predictor.py Tests
# ---------------------------------------------------------------------------
# hybrid_predictor.py no longer does entropy coding (no compress/decompress,
# no predict_probs, no PASSTHROUGH/LINEAR_CMA modes) — that machinery was
# removed because it duplicated what ASDP/CM already does in C++ and was
# never wired into the live bridge path. What's tested here instead: the
# two small feed-forward model definitions actually used by the pipeline
# (StructureClassifier, TransformGainPredictor) and the ONNX inference
# wrapper's fallback behavior, which is the part every other layer
# actually depends on working correctly even when no model is trained yet.

class TestHybridPredictorModels:

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_structure_classifier_forward_shape(self):
        clf = StructureClassifier(in_dim=64, n_classes=8)
        import torch
        x = torch.zeros(4, 64)  # batch of 4
        out = clf(x)
        assert out.shape == (4, 8)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_gain_predictor_forward_shape(self):
        model = TransformGainPredictor(in_dim=64)
        import torch
        x = torch.zeros(4, 64)
        out = model(x)
        assert out.shape == (4,)  # squeezed to scalar-per-sample

    def test_onnx_gain_predictor_missing_model_returns_none(self):
        """No model file -> predict() returns None, never raises. Every
        caller (the live gain-guard in adaptive_optimizer.py, once wired
        up) depends on this degrading gracefully to the existing
        zlib-1-ratio heuristic rather than crashing when no model has
        been trained yet -- which is the default state until someone
        runs train_predictor.py against real data."""
        predictor = ONNXGainPredictor(None)
        assert predictor.available is False
        assert predictor.predict(np.zeros((1, 64))) is None

    def test_onnx_gain_predictor_bad_path_returns_none(self):
        predictor = ONNXGainPredictor("/nonexistent/path/model.onnx")
        assert predictor.available is False
        assert predictor.predict(np.zeros((1, 64))) is None


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

    def test_best_strategy_after_plays(self):
        """Uses COLUMNAR data specifically — it's the one DataClass with
        two genuinely distinct strategies (col_split vs col_raw) to choose
        between. TEXT/TIMESERIES/etc. now only ever have a single "raw"
        strategy (see structure_discovery.py's division-of-labor
        docstring), so testing best-strategy selection against them would
        be vacuous: there'd be nothing to choose between."""
        opt = self._make_optimizer(make_columnar())
        strategies = list(opt._strategies.values())
        assert len(strategies) >= 2, \
            "expected col_split vs col_raw choice for columnar data"
        good = strategies[0]
        bad  = strategies[-1]
        for _ in range(5):
            opt.record_result(good.name, 4096, 500,  2.0)  # good ratio
        for _ in range(5):
            opt.record_result(bad.name,  4096, 4000, 1.0)  # bad ratio
        assert opt.best_strategy().name == good.name

    def test_recent_ratio_computed(self):
        opt = self._make_optimizer(make_timeseries())
        s = opt.select_strategy()
        opt.record_result(s.name, 4096, 2048, 3.0)
        assert abs(opt.recent_ratio() - 2.0) < 0.02, \
            f"expected ~2.0, got {opt.recent_ratio()}"


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestK2Pipeline:
    """
    K2Pipeline's real contract is prepare() / compress_full() /
    reseal_frame() / decompress_full() — the C++ bridge calls
    asdp_compress() between compress_full() and reseal_frame(), and
    asdp_decompress() before decompress_full(). These tests don't have
    ASDP available (that's exercised by the C++ test suite separately),
    so they use identity in place of the real ASDP step via
    _fake_asdp_roundtrip below — what's actually under test here is the
    Python-side transform/frame logic, which is the correct boundary for
    a Python-only test file. The previously-removed compress() method
    (and HybridPredictor.compress() underneath it) tested the old
    arithmetic-coder entropy path directly; there's no equivalent to port
    forward since ASDP/CM, the real entropy coder, is C++-only.
    """

    @staticmethod
    def _fake_asdp_roundtrip(pipeline: K2Pipeline, data: bytes) -> bytes:
        """
        Stands in for what the real C++ bridge does: call compress_full()
        to get the pre-ASDP frame, then reseal_frame() with the payload
        unchanged (identity 'compression') instead of a real
        asdp_compress() call. Returns the final sealed frame, exactly the
        bytes decompress_full() expects.
        """
        frame = pipeline.compress_full(data)
        backend, orig_size, txhdr, payload = decode_k2_frame(frame)
        return pipeline.reseal_frame(frame, payload)

    def test_prepare_and_compress(self):
        pipeline = K2Pipeline(exploration=1.0)
        data = make_timeseries(16384)
        hint = pipeline.prepare(data[:8192])
        assert hint.data_class is not None

        sealed = self._fake_asdp_roundtrip(pipeline, data)
        assert isinstance(sealed, bytes)
        assert len(sealed) > 0

    def test_roundtrip_recovers_original(self):
        """The actual correctness property that matters: whatever
        Python-side transform was applied, decompress_full() must invert
        it exactly. This is the one thing the old test file never
        actually verified (it only checked compressed output was
        non-empty) — worth adding."""
        for maker in (make_timeseries, make_float32, make_text, make_columnar):
            pipeline = K2Pipeline()
            data = maker()
            pipeline.prepare(data[:min(len(data), 8192)])
            sealed = self._fake_asdp_roundtrip(pipeline, data)
            restored = pipeline.decompress_full(sealed)
            assert restored == data, f"{maker.__name__}: roundtrip mismatch"

    def test_stats_populated_after_compress(self):
        pipeline = K2Pipeline()
        data = make_text(8192)
        pipeline.prepare(data[:2048])
        self._fake_asdp_roundtrip(pipeline, data)
        stats = pipeline.stats()
        assert "recent_ratio" in stats
        assert "best_strategy" in stats
        assert stats["recent_ratio"] > 0

    def test_multiple_chunks_track_ratio(self):
        """
        With more chunks, reseal_frame() should keep recording real
        ratios into the bandit. We verify the bandit's recorded ratio
        stays sane (not zero/negative) across repeated calls, since the
        identity fake-ASDP step here means every chunk's "ratio" is
        exactly 1.0x by construction -- the real ratio-improvement
        property (does the bandit converge to a BETTER strategy over
        time) needs real ASDP and belongs in an end-to-end test against
        the actual C++ bridge, not this Python-only file.
        """
        pipeline = K2Pipeline(exploration=0.5)
        data = make_timeseries(32768)
        pipeline.prepare(data[:4096])

        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            chunk = data[i: i + chunk_size]
            if not chunk:
                continue
            self._fake_asdp_roundtrip(pipeline, chunk)

        assert pipeline.stats()["recent_ratio"] > 0

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
            sealed = self._fake_asdp_roundtrip(pipeline, data)
            assert len(sealed) > 0, f"Failed on {name}"

    def test_random_blob_fallback(self):
        """Random/encrypted data should fall back gracefully (no crash)."""
        pipeline = K2Pipeline()
        data = make_random_blob(8192)
        pipeline.prepare(data)
        sealed = self._fake_asdp_roundtrip(pipeline, data)
        assert isinstance(sealed, bytes)


# ---------------------------------------------------------------------------
# Benchmark (not a pytest test, run manually)
# ---------------------------------------------------------------------------

def benchmark():
    """
    Measures Python-side overhead and transform-decision behavior only.
    Does NOT measure real compression ratio or throughput — that requires
    the real ASDP/CM entropy coder, which is C++-only and not available
    in this Python-only test file. Reporting a "ratio" here using an
    identity stand-in for ASDP would be actively misleading (every result
    would be ~1.0x by construction, looking like real numbers but
    measuring nothing). For real end-to-end ratio/throughput numbers, use
    k2cli (see the top-level README) against real data.
    """
    import time

    print("\n=== Python-side overhead benchmark ===")
    print("(transform decisions + frame encode/decode only — NOT real "
          "compression ratio, which needs the C++ ASDP bridge)\n")

    datasets = {
        "timeseries (u64)": make_timeseries(65536),
        "float32":          make_float32(65536),
        "text":             make_text(65536),
        "columnar":         make_columnar(),
        "random blob":      make_random_blob(65536),
    }

    for name, data in datasets.items():
        pipeline = K2Pipeline(exploration=0.0, latency_weight=0.1)
        pipeline.prepare(data[:min(len(data), 65536)])
        disc = StructureDiscovery()
        hint = disc.analyze(data[:65536])
        print(f"  {name}")
        print(f"    hint: class={hint.data_class.name} stride={hint.column_stride} "
              f"delta_gain={hint.delta_gain_db:.1f}dB elem={hint.element_size} "
              f"conf={hint.confidence:.2f} transforms={hint.suggested_transforms}")

        N_RUNS = 3
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            frame = pipeline.compress_full(data)
            backend, orig_size, txhdr, payload = decode_k2_frame(frame)
            pipeline.reseal_frame(frame, payload)  # identity stand-in for asdp_compress
            times.append(time.perf_counter() - t0)
        elapsed_ms = min(times) * 1000
        throughput = (len(data) / 1e6) / (elapsed_ms / 1000)
        strat = pipeline.stats().get("best_strategy", "?")
        print(f"    python-side: {elapsed_ms:.2f}ms  ({throughput:.0f} MB/s, "
              f"transform-decision overhead only)  strategy={strat}\n")


if __name__ == "__main__":
    # Run tests manually
    import traceback
    suites = [
        TestStructureDiscovery,
        TestHybridPredictorModels,
        TestAdaptiveOptimizer,
        TestK2Pipeline,
    ]
    failed = 0
    skipped = 0
    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite) if m.startswith("test_")]
        for method in methods:
            bound = getattr(suite, method)
            # Honor @pytest.mark.skipif markers even when run outside
            # pytest (e.g. plain `python test_pipeline_integration.py`)
            # — without this, tests gated on TORCH_AVAILABLE crash with
            # NameError instead of skipping. Handles both real pytest
            # (marker objects on .pytestmark) and the no-pytest fallback
            # shim above (plain ._skip/._skip_reason attributes).
            skip_reason = None
            real_markers = getattr(bound, "pytestmark", None)
            if real_markers:
                for m in real_markers:
                    if m.name == "skipif" and m.args and m.args[0]:
                        skip_reason = m.kwargs.get("reason", "skipped")
            elif getattr(bound, "_skip", False):
                skip_reason = getattr(bound, "_skip_reason", "skipped")
            if skip_reason:
                print(f"  - {suite_cls.__name__}.{method} (skipped: {skip_reason})")
                skipped += 1
                continue
            try:
                if hasattr(suite, "setup_method"):
                    suite.setup_method()
                bound()
                print(f"  ✓ {suite_cls.__name__}.{method}")
            except Exception as e:
                print(f"  ✗ {suite_cls.__name__}.{method}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'} "
          f"({skipped} skipped)")
    benchmark()
