"""
k2/src/python/structure_discovery.py

Auto Structure Discovery — K2
------------------------------
Lightweight ML probe that analyzes raw byte/chunk data to:
  1. Compute entropy and detect regularity signals
  2. Identify common patterns: delta sequences, run-length candidates,
     columnar repeats, floating-point arrays, text-like data
  3. Classify into a StructureHint describing the data's likely layout
  4. Optionally use a tiny ONNX classifier for richer type detection

Division of labor with the C++ ASDP/CM backend
------------------------------------------------
ASDP's own pick_transform_cm() (asdp.cpp) already does a real, empirically
validated trial-encode search over {none, bytedelta_1/2/4/8, bytesplit_4/8}
on every block, before context-mixing entropy coding. That search is ground
truth: it measures actual CM-encoded size, not a heuristic proxy, and it
always includes "none" as a candidate, so it can never make a block worse
than skipping a transform.

This module therefore does NOT try to re-pick bytedelta/bytesplit — Python
duplicating that search would just be a slower, less accurate copy of work
C++ already does correctly. Instead, StructureDiscovery focuses on the one
real gap: transforms ASDP's C++ layer cannot do at all today, primarily
columnsplit (true struct-of-arrays reordering for interleaved/columnar
records). The suggested transform chain runs ONCE in Python, upstream of
ASDP; C++'s own trial-encode then runs independently on whatever bytes
Python hands it, exactly as it does for any other input. There is no zstd
fallback anywhere in this pipeline (backend 0x04 / ASDP-LH is the only
entropy backend) — any reference to zstd/bwt/mtf in older versions of this
file predates that decision and has been removed.

The output is a StructureHint that AdaptiveOptimizer/K2Pipeline consume to
select the upstream transform chain.
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

# Optional: ONNX Runtime for the tiny classifier model
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data-type taxonomy
# ---------------------------------------------------------------------------

class DataClass(Enum):
    UNKNOWN        = auto()
    INTEGER_ARRAY  = auto()   # le-u32, le-u64, le-i32, etc.
    FLOAT_ARRAY    = auto()   # le-f32, le-f64
    COLUMNAR       = auto()   # struct-of-arrays / Protobuf-like
    TIMESERIES     = auto()   # monotone / delta-coded integers
    TEXT           = auto()   # UTF-8 prose / CSV text
    BINARY_BLOB    = auto()   # opaque / encrypted / already compressed
    MIXED          = auto()   # heterogeneous; use generic Zstd fallback


@dataclass
class StructureHint:
    data_class: DataClass = DataClass.UNKNOWN
    element_size: int = 1          # detected primitive width (1/2/4/8 bytes)
    is_little_endian: bool = True
    byte_entropy: float = 0.0      # Shannon entropy on raw bytes
    delta_gain_db: float = 0.0     # estimated gain from delta transform (dB)
    repeat_density: float = 0.0    # fraction of 8-byte runs that are exact repeats
    column_stride: int = 0         # detected column stride (0 = not columnar)
    suggested_transforms: list[str] = field(default_factory=list)
    confidence: float = 0.0        # [0, 1]


# ---------------------------------------------------------------------------
# Core heuristics
# ---------------------------------------------------------------------------

def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of the raw byte stream, in bits/byte."""
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _detect_element_size(data: bytes, max_probe: int = 65536) -> int:
    """
    Try element widths 2, 4, 8.  Choose the one whose delta-encoded stream
    has the lowest entropy (higher compressibility signal).
    """
    probe = data[:max_probe]
    n = len(probe)
    best_w, best_ent = 1, float("inf")
    for w in (2, 4, 8):
        if n < w * 16:
            continue
        arr = np.frombuffer(probe[: (n // w) * w], dtype=f"<u{w}")
        deltas = np.diff(arr.astype(np.int64))
        # Compress delta bytes
        delta_bytes = deltas.astype(f"<i{w}").tobytes()
        ent = _byte_entropy(delta_bytes)
        if ent < best_ent:
            best_ent = ent
            best_w = w
    return best_w


def _delta_gain_db(data: bytes, element_size: int) -> float:
    """Estimate compression gain (in dB) from applying delta transform."""
    n = len(data)
    if n < element_size * 4:
        return 0.0
    w = element_size
    arr = np.frombuffer(data[: (n // w) * w], dtype=f"<u{w}")
    raw_ent = _byte_entropy(data[: (n // w) * w])
    delta_bytes = np.diff(arr.astype(np.int64)).astype(f"<i{w}").tobytes()
    delta_ent = _byte_entropy(delta_bytes)
    if raw_ent == 0:
        return 0.0
    return 20 * math.log10(max(raw_ent / max(delta_ent, 1e-9), 1.0))


def _repeat_density(data: bytes, stride: int = 8) -> float:
    """Fraction of non-overlapping stride-byte windows that are exact repeats."""
    chunks = [data[i:i + stride] for i in range(0, len(data) - stride, stride)]
    if not chunks:
        return 0.0
    counts = Counter(chunks)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(chunks)


def _detect_float_array(data: bytes, element_size: int) -> bool:
    """Heuristic: check if data looks like IEEE 754 floats (exponent distribution)."""
    if element_size not in (4, 8):
        return False
    n = len(data)
    w = element_size
    arr = np.frombuffer(data[: (n // w) * w], dtype=f"<f{w}")
    finite = arr[np.isfinite(arr)]
    if len(finite) < 4:
        return False
    # Float arrays have characteristic exponent clustering
    if w == 4:
        bits = arr.view(np.uint32)
        exps = (bits >> 23) & 0xFF
    else:
        bits = arr.view(np.uint64)
        exps = (bits >> 52) & 0x7FF
    unique_exp_ratio = len(np.unique(exps)) / max(len(exps), 1)
    return 0.02 < unique_exp_ratio < 0.9


def _is_text_like(data: bytes) -> bool:
    """True if >85 % of bytes are printable ASCII or common UTF-8 continuation."""
    if not data:
        return False
    printable = sum(1 for b in data[:4096] if 0x09 <= b <= 0x0D or 0x20 <= b <= 0x7E)
    return printable / min(len(data), 4096) > 0.85


def _detect_column_stride(data: bytes, candidate_sizes=(4, 8, 12, 16, 24, 32)) -> int:
    """
    Columnar data often has periodic structure.  Check autocorrelation of
    byte values at candidate strides.

    Note: this only tests a fixed list of candidate strides. A record
    layout whose true byte width isn't in candidate_sizes (e.g. a 14-byte
    record) won't be detected even if it's genuinely columnar — caller
    falls through to whatever other DataClass heuristics fit instead, not
    a crash, but a real detection gap worth knowing about.
    """
    probe = np.frombuffer(data[:8192], dtype=np.uint8).astype(np.float32)
    if len(probe) < 64:
        return 0
    best_stride, best_corr = 0, 0.0
    for stride in candidate_sizes:
        if stride >= len(probe) // 4:
            continue
        # Mean absolute correlation at multiples of stride
        shifted = probe[stride:]
        base = probe[: len(shifted)]
        # np.corrcoef divides by each array's stddev; a zero-variance
        # window (e.g. a constant byte run landing in the probe) produces
        # NaN, which silently loses the "corr > best_corr" comparison
        # rather than crashing -- but that means a real stride could be
        # missed without any signal that something was skipped. Guard
        # explicitly instead of relying on NaN comparison semantics.
        if base.std() == 0.0 or shifted.std() == 0.0:
            continue
        corr = float(np.corrcoef(base, shifted)[0, 1])
        if not np.isfinite(corr):
            continue
        if corr > best_corr and corr > 0.15:
            best_corr = corr
            best_stride = stride
    return best_stride


# ---------------------------------------------------------------------------
# Optional ONNX classifier
# ---------------------------------------------------------------------------

class ONNXStructureClassifier:
    """
    Wraps a tiny ONNX model (e.g. 64-dim feature → 8-class softmax).
    Train offline on labelled byte chunks; export to ONNX.
    Falls back gracefully if model file is missing.
    """

    def __init__(self, model_path: str):
        self._session: Optional["ort.InferenceSession"] = None
        if ONNX_AVAILABLE:
            try:
                self._session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                pass

    def predict(self, features: np.ndarray) -> tuple[DataClass, float]:
        """Returns (predicted_class, confidence)."""
        if self._session is None:
            return DataClass.UNKNOWN, 0.0
        inp_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {inp_name: features.astype(np.float32)})[0]
        probs = _softmax(logits[0])
        idx = int(np.argmax(probs))
        classes = list(DataClass)
        return classes[idx % len(classes)], float(probs[idx])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _extract_features(data: bytes, element_size: int) -> np.ndarray:
    """
    Build a fixed-length 64-dim feature vector for the ONNX classifier.
    Features: byte histogram (32 bins), entropy, delta entropy,
              repeat density, column stride hint, is_float, is_text.
    """
    probe = data[:4096]
    arr = np.frombuffer(probe, dtype=np.uint8).astype(np.float32)
    hist, _ = np.histogram(arr, bins=32, range=(0, 256), density=True)
    raw_ent = _byte_entropy(probe)
    w = element_size
    n = len(probe)
    delta_ent = 0.0
    if n >= w * 4:
        iarr = np.frombuffer(probe[:(n // w) * w], dtype=f"<u{w}")
        delta_bytes = np.diff(iarr.astype(np.int64)).astype(f"<i{w}").tobytes()
        delta_ent = _byte_entropy(delta_bytes)
    features = np.concatenate([
        hist,                              # 32
        [raw_ent / 8.0],                   # 1
        [delta_ent / 8.0],                 # 1
        [_repeat_density(probe)],          # 1
        [_detect_column_stride(probe) / 32.0],  # 1
        [float(_detect_float_array(probe, element_size))],  # 1
        [float(_is_text_like(probe))],     # 1
        np.zeros(26),                      # pad to 64
    ])
    return features[:64].reshape(1, -1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StructureDiscovery:
    """
    Main entry point.  Call `analyze(data)` to get a StructureHint.

    Parameters
    ----------
    onnx_model_path : str or None
        Path to optional ONNX type-classifier model.  If None or the file
        doesn't exist, falls back entirely to heuristics.
    probe_bytes : int
        How many leading bytes to use for analysis (default: 512 KB).
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        probe_bytes: int = 512 * 1024,
    ):
        self._probe_bytes = probe_bytes
        self._classifier: Optional[ONNXStructureClassifier] = None
        if onnx_model_path:
            self._classifier = ONNXStructureClassifier(onnx_model_path)

    def analyze(self, data: bytes) -> StructureHint:
        probe = data[: self._probe_bytes]
        hint = StructureHint()

        # --- Basic entropy ---
        hint.byte_entropy = _byte_entropy(probe)

        # High entropy → likely already compressed/encrypted, or otherwise
        # incompressible by structure alone. Threshold matches ASDP's own
        # cm_precheck_incompressible() entropy gate (cm.cpp) for
        # consistency, though Python only checks entropy here (cheap,
        # avoids a wasted probe) — ASDP's C++ precheck additionally checks
        # match-density before truly giving up, so it correctly still
        # finds wins on duplicated-but-high-entropy data (e.g. repeated
        # texture mips) that this Python-side gate alone would miss. That
        # is fine: skipping a Python transform here does not prevent C++
        # from finding that win independently afterward, since ASDP runs
        # regardless of what Python decided.
        if hint.byte_entropy > 7.9:
            hint.data_class = DataClass.BINARY_BLOB
            hint.confidence = 0.85
            hint.suggested_transforms = []
            return hint

        # --- Text check (fast path) ---
        if _is_text_like(probe):
            hint.data_class = DataClass.TEXT
            hint.element_size = 1
            hint.suggested_transforms = []
            hint.confidence = 0.80
            return hint

        # --- Numeric structure ---
        hint.element_size = _detect_element_size(probe)
        hint.delta_gain_db = _delta_gain_db(probe, hint.element_size)
        hint.repeat_density = _repeat_density(probe)
        hint.column_stride = _detect_column_stride(probe)

        # Optional ONNX override
        onnx_class, onnx_conf = DataClass.UNKNOWN, 0.0
        if self._classifier:
            feats = _extract_features(probe, hint.element_size)
            onnx_class, onnx_conf = self._classifier.predict(feats)

        # --- Classify ---
        if onnx_conf > 0.70:
            hint.data_class = onnx_class
            hint.confidence = onnx_conf
        else:
            hint.data_class, hint.confidence = self._heuristic_classify(hint)

        # --- Suggest transforms ---
        hint.suggested_transforms = self._suggest_transforms(hint)
        return hint

    def _heuristic_classify(
        self, hint: StructureHint
    ) -> tuple[DataClass, float]:
        score: dict[DataClass, float] = {c: 0.0 for c in DataClass}

        if hint.delta_gain_db > 6.0:
            score[DataClass.TIMESERIES] += 0.5
            score[DataClass.INTEGER_ARRAY] += 0.3

        if hint.element_size in (4, 8) and hint.delta_gain_db < 3.0:
            score[DataClass.FLOAT_ARRAY] += 0.4

        if hint.column_stride > 0 and hint.delta_gain_db < 3.0:
            score[DataClass.COLUMNAR] += 0.6


        if hint.repeat_density > 0.3:
            score[DataClass.INTEGER_ARRAY] += 0.3
            score[DataClass.COLUMNAR] += 0.2

        if hint.byte_entropy < 4.0:
            score[DataClass.INTEGER_ARRAY] += 0.3
            score[DataClass.TIMESERIES] += 0.2

        # Weak COLUMNAR prior for 4-byte, low-delta data that no other signal
        # has clearly claimed: FLOAT_ARRAY still wins when detected (0.4 > 0.3).
        # Previously guarded by `hint.confidence < 0.5`, but hint.confidence is
        # always 0.0 here (it is set by the caller *after* this method returns),
        # making that guard a permanent no-op. Removed so the actual behavior is
        # explicit rather than hidden behind a dead condition.
        if hint.element_size == 4 and hint.delta_gain_db < 2.0:
            score[DataClass.COLUMNAR] += 0.3

        best = max(score, key=lambda c: score[c])
        conf = min(score[best], 1.0)
        if conf < 0.2:
            best = DataClass.UNKNOWN
        return best, conf

    def _suggest_transforms(self, hint: StructureHint) -> list[str]:
        """
        Suggest the upstream (Python-side) transform chain, applied once
        before data reaches ASDP. Deliberately does NOT suggest
        bytedelta/bytesplit variants: ASDP's pick_transform_cm() already
        searches that space empirically (real CM-encoded trial size, not a
        heuristic), every time, for every block — Python re-suggesting
        them would either duplicate that work for no gain, or double-apply
        a transform ASDP would have chosen anyway. The one real gap is
        columnsplit (true struct-of-arrays reordering across whole
        records), which ASDP cannot do at all today.
        """
        if hint.data_class == DataClass.COLUMNAR and hint.column_stride > 0:
            return [f"columnsplit-{hint.column_stride}"]

        # Every other class: hand bytes to ASDP as-is and let
        # pick_transform_cm's trial-encode make the real call.
        return []


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    disc = StructureDiscovery()

    # Simulate a uint64 time-series
    rng = np.random.default_rng(42)
    ts = np.cumsum(rng.integers(0, 100, size=8192)).astype("<u8")
    hint = disc.analyze(ts.tobytes())
    print(f"[timeseries]  class={hint.data_class.name!r}  "
          f"elem={hint.element_size}  Δgain={hint.delta_gain_db:.1f}dB  "
          f"transforms={hint.suggested_transforms}  conf={hint.confidence:.2f}")

    # Simulate float32 array
    floats = rng.standard_normal(8192).astype("<f4")
    hint2 = disc.analyze(floats.tobytes())
    print(f"[float32]     class={hint2.data_class.name!r}  "
          f"elem={hint2.element_size}  transforms={hint2.suggested_transforms}  "
          f"conf={hint2.confidence:.2f}")

    # Simulate text
    text = (b"The quick brown fox jumps over the lazy dog. " * 512)
    hint3 = disc.analyze(text)
    print(f"[text]        class={hint3.data_class.name!r}  "
          f"transforms={hint3.suggested_transforms}  conf={hint3.confidence:.2f}")

    # Simulate random (compressed/encrypted)
    blob = os.urandom(32768)
    hint4 = disc.analyze(blob)
    print(f"[random blob] class={hint4.data_class.name!r}  "
          f"entropy={hint4.byte_entropy:.2f}  transforms={hint4.suggested_transforms}")
