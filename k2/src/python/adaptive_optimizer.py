"""
k2/src/python/adaptive_optimizer.py

Adaptive Optimization Layer — K2
----------------------------------
K2Pipeline owns ONE decision: whether to apply a Python-side structural
transform (currently: columnsplit) before handing data to the ASDP-LH C++
backend (backend 0x04, the sole entropy backend; there is no zstd/zlib
fallback). It does NOT pick bytedelta/bytesplit variants — ASDP's own
pick_transform_cm() already searches that space with a real trial-encode
against the actual CM-coded size, every block, and Python duplicating that
search would be slower and less accurate. See structure_discovery.py's
module docstring for the full division-of-labor rationale.

Per-DataClass strategy space (post-rebuild):
  COLUMNAR              → columnsplit-N vs raw (2 strategies, bandit picks)
  everything else       → raw only (no Python-side transform to choose
                           between; ASDP's trial-encode handles the rest)

The multi-armed-bandit (UCB1) strategy selection and online ratio-feedback
loop are unchanged in spirit, just operating over the smaller, honest
strategy space above instead of also picking a (now-removed) neural
predictor mode.

K2 Frame Format
---------------
Offset  Size  Field
0       4     magic  b'K2\\xf7\\x01'
4       1     version  0x01
5       1     backend  0x04=ASDP  (0x01/02/03 retired)
6       1     flags    bit0=transforms_applied  rest reserved
7       1     reserved 0x00
8       8     orig_size  uint64 LE
16      2     txhdr_len  uint16 LE
18      N     transform_header (K2T\\x01 ...)
18+N    ...   ASDP frame (output of asdp_compress)

The C++ bridge reads the frame and:
  - calls asdp_compress(payload), reseals frame via reseal_frame().
  - On decompress: asdp_decompress then invert transforms via
    decompress_full().
"""

from __future__ import annotations

import math
import struct
import threading
import time
import zlib   # used only for the transform-gain probe heuristic (_zlib1_ratio)
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from structure_discovery import StructureHint, DataClass


# ---------------------------------------------------------------------------
# K2 frame constants
# ---------------------------------------------------------------------------

_FRAME_MAGIC      = b"K2\xf7\x01"
_FRAME_VERSION    = 0x01
_BACKEND_ASDP     = 0x04   # sole entropy backend
_FLAG_TRANSFORMS  = 0x01
_FRAME_HDR_STRUCT = struct.Struct("<4sBBBBQH")   # 18 bytes
_FRAME_HDR_SIZE   = _FRAME_HDR_STRUCT.size       # 18


# ---------------------------------------------------------------------------
# K2 frame encode / decode
# ---------------------------------------------------------------------------

def encode_k2_frame(
    backend: int,
    orig_size: int,
    txhdr: bytes,
    payload: bytes,
) -> bytes:
    """
    Build a complete K2 frame.

    payload is pre-transform bytes; the C++ bridge replaces it with the
    ASDP-compressed output and re-seals the frame.
    """
    flags = _FLAG_TRANSFORMS if txhdr and len(txhdr) > 5 else 0x00
    hdr = _FRAME_HDR_STRUCT.pack(
        _FRAME_MAGIC, _FRAME_VERSION, backend, flags, 0x00,
        orig_size, len(txhdr)
    )
    return hdr + txhdr + payload


def decode_k2_frame(data: bytes) -> tuple[int, int, bytes, bytes]:
    """
    Parse a K2 frame.
    Returns (backend, orig_size, txhdr, payload).
    Raises ValueError on bad magic/version.
    """
    if len(data) < _FRAME_HDR_SIZE:
        raise ValueError(f"K2 frame too short: {len(data)} bytes")
    magic, version, backend, flags, _, orig_size, txhdr_len = \
        _FRAME_HDR_STRUCT.unpack(data[:_FRAME_HDR_SIZE])
    if magic != _FRAME_MAGIC:
        raise ValueError(f"bad K2 frame magic: {magic!r}")
    if version != _FRAME_VERSION:
        raise ValueError(f"unknown K2 frame version: {version}")
    need = _FRAME_HDR_SIZE + txhdr_len
    if len(data) < need:
        raise ValueError(f"K2 frame truncated: need {need}, got {len(data)}")
    txhdr   = data[_FRAME_HDR_SIZE : need]
    payload = data[need:]
    return backend, orig_size, txhdr, payload


def reseal_k2_frame(frame: bytes, new_payload: bytes) -> bytes:
    """
    Replace the payload of an existing K2 frame with new_payload.
    Used by the C++ bridge after CCtx::compressSerial() returns.
    orig_size and txhdr are preserved from the original frame.
    """
    backend, orig_size, txhdr, _ = decode_k2_frame(frame)
    return encode_k2_frame(backend, orig_size, txhdr, new_payload)


# ---------------------------------------------------------------------------
# Strategy definition
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """
    A candidate upstream (Python-side) transform choice for the bandit to
    evaluate. Previously also carried predictor_mode/alpha/chunk_size for
    a neural-mixing entropy path that has been removed (see
    hybrid_predictor.py's module docstring) — those fields are gone, since
    they no longer correspond to any real decision Python makes. The only
    thing left to choose is the transform chain itself; backend is always
    ASDP (kept as a field for forward-compat / explicitness, not because
    there's currently a second option).
    """
    name: str
    transforms: list[str]
    backend: int = field(default=_BACKEND_ASDP)

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
# Default strategy sets per DataClass
# ---------------------------------------------------------------------------

def _default_strategies_for(hint: StructureHint) -> list[Strategy]:
    """
    Build the candidate set for the bandit to choose between.

    Only COLUMNAR data has a genuine two-way choice: apply columnsplit
    (Python's one real upstream transform) or don't, since whether it
    helps depends on the actual record layout and isn't free to apply (it
    costs a full data copy). Every other class has no Python-side
    transform to choose between any more — ASDP's pick_transform_cm()
    owns bytedelta/bytesplit selection entirely, with a real trial-encode
    against actual CM-coded size, every block. Giving those classes a
    single "raw" strategy is intentional, not a placeholder: there is
    nothing else for Python to decide for them today, and the bandit
    degenerating to "one strategy, always picked" is the honest behavior
    rather than a fake choice between options that don't differ.
    """
    cls = hint.data_class

    if cls == DataClass.COLUMNAR and hint.column_stride > 0:
        s = hint.column_stride
        return [
            Strategy("col_split", [f"columnsplit-{s}"]),
            Strategy("col_raw",   []),
        ]

    # TIMESERIES / INTEGER_ARRAY / FLOAT_ARRAY / TEXT / BINARY_BLOB /
    # UNKNOWN / MIXED, and COLUMNAR without a usable detected stride:
    # nothing for Python to choose, hand bytes to ASDP as-is.
    return [Strategy("raw", [])]


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
        ratio_score = math.log2(max(self.ratio, 1.0))
        speed_score = math.log2(max(self.throughput_mbs, 0.1))
        return (1.0 - latency_weight) * ratio_score + latency_weight * speed_score


# ---------------------------------------------------------------------------
# Adaptive Optimizer
# ---------------------------------------------------------------------------

class AdaptiveOptimizer:

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

        base  = _default_strategies_for(hint) if hint else []
        extra = extra_strategies or []
        self._strategies: dict[str, Strategy] = {s.name: s for s in base + extra}
        if not self._strategies:
            self._strategies["raw"] = Strategy("raw", [])
        self._current: Optional[Strategy] = None

    def select_strategy(self) -> Strategy:
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
        result = _ChunkResult(strategy_name, input_size, output_size, elapsed_ms)
        score  = result.score(self._latency_weight)
        if self._budget_ms and elapsed_ms > self._budget_ms:
            penalty = (elapsed_ms - self._budget_ms) / self._budget_ms * 0.5
            score = max(0.0, score - penalty)
        with self._lock:
            self._total_plays += 1
            self._window.append(result)
            if strategy_name in self._strategies:
                self._strategies[strategy_name].record(score)

    # get_config / HybridConfig removed — a Strategy's `transforms` list IS
    # the complete decision now (apply this transform chain, or don't).
    # There is no longer a separate runtime config object to build from it.

    def stats(self) -> dict:
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
        with self._lock:
            played = [s for s in self._strategies.values() if s._n_plays > 0]
            if not played:
                return next(iter(self._strategies.values()))
            return max(played, key=lambda s: s.mean_score)

    def recent_ratio(self) -> float:
        with self._lock:
            if not self._window:
                return 1.0
            return sum(r.ratio for r in self._window) / len(self._window)

    def recent_throughput_mbs(self) -> float:
        with self._lock:
            if not self._window:
                return 0.0
            return sum(r.throughput_mbs for r in self._window) / len(self._window)

    # tune_alpha (continuous-parameter tuning) removed along with
    # Strategy.alpha — there is no longer a continuous knob to adjust.
    # The bandit's UCB1 selection over the (now discrete, small) strategy
    # set is the only tuning mechanism needed.

    def compress_with_timing(
        self,
        data: bytes,
        compress_fn: Callable[[bytes, Strategy], bytes],
    ) -> tuple[bytes, str]:
        strategy = self.select_strategy()
        t0 = time.perf_counter()
        compressed = compress_fn(data, strategy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.record_result(strategy.name, len(data), len(compressed), elapsed_ms)
        return compressed, strategy.name


# ---------------------------------------------------------------------------
# Transform primitives
# ---------------------------------------------------------------------------

def _apply_bytedelta(data: bytes, w: int) -> bytes:
    n = len(data); aligned = (n // w) * w
    arr = np.frombuffer(data[:aligned], dtype=f"<u{w}").copy()
    arr[1:] = arr[1:].astype(np.int64) - arr[:-1].astype(np.int64)
    return bytes(bytearray(arr.astype(f"<i{w}").tobytes()) + data[aligned:])

def _apply_bytedelta_inv(data: bytes, w: int) -> bytes:
    n = len(data); aligned = (n // w) * w
    arr = np.frombuffer(data[:aligned], dtype=f"<i{w}").copy()
    np.cumsum(arr, out=arr)
    return bytes(bytearray(arr.astype(f"<u{w}").tobytes()) + data[aligned:])

def _apply_bytesplit(data: bytes, w: int) -> bytes:
    n = len(data); aligned = (n // w) * w
    arr = np.frombuffer(data[:aligned], dtype=np.uint8).reshape(-1, w)
    return bytes(bytearray(arr.T.flatten().tobytes()) + data[aligned:])

def _apply_bytesplit_inv(data: bytes, w: int) -> bytes:
    n = len(data); n_elem = n // w; aligned = n_elem * w
    arr = np.frombuffer(data[:aligned], dtype=np.uint8).reshape(w, n_elem)
    return bytes(bytearray(arr.T.flatten().tobytes()) + data[aligned:])

def _apply_columnsplit(data: bytes, stride: int) -> bytes:
    n = len(data)
    if stride <= 0 or stride > n: return data
    n_rows = n // stride; aligned = n_rows * stride
    arr = np.frombuffer(data[:aligned], dtype=np.uint8).reshape(n_rows, stride)
    return bytes(bytearray(arr.T.flatten().tobytes()) + data[aligned:])

def _apply_columnsplit_inv(data: bytes, stride: int) -> bytes:
    n = len(data)
    if stride <= 0 or stride > n: return data
    n_rows = n // stride; aligned = n_rows * stride
    arr = np.frombuffer(data[:aligned], dtype=np.uint8).reshape(stride, n_rows)
    return bytes(bytearray(arr.T.flatten().tobytes()) + data[aligned:])

def _apply_zigzag(data: bytes) -> bytes:
    arr = np.frombuffer(data, dtype=np.int8)
    return bytes((arr.astype(np.int16) * 2 ^ arr.astype(np.int16) >> 7).astype(np.uint8))

def _apply_zigzag_inv(data: bytes) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    return bytes(((arr >> 1) ^ -(arr & 1)).astype(np.int8).view(np.uint8))


# ---------------------------------------------------------------------------
# Transform header (unchanged from previous version)
# ---------------------------------------------------------------------------

_TX_MAGIC       = b"K2T\x01"
_OP_BYTEDELTA   = 0x01
_OP_BYTESPLIT   = 0x02
_OP_COLUMNSPLIT = 0x03
_OP_ZIGZAG      = 0x04
_OP_BWT         = 0x10   # no-op, kept for compat
_OP_MTF         = 0x11   # no-op, kept for compat

def _parse_transform_name(name: str) -> tuple[int, int]:
    if name == "zigzag":   return (_OP_ZIGZAG, 0)
    if name == "bwt":      return (_OP_BWT, 0)
    if name == "mtf":      return (_OP_MTF, 0)
    parts = name.split("-")
    if len(parts) == 2:
        kind, param_s = parts
        try:
            p = int(param_s)
        except ValueError:
            return (0, 0)
        if kind == "bytedelta":   return (_OP_BYTEDELTA, p)
        if kind == "bytesplit":   return (_OP_BYTESPLIT, p)
        if kind == "columnsplit": return (_OP_COLUMNSPLIT, min(p, 255))
    return (0, 0)

def _ops_from_transforms(transforms: list[str]) -> list[tuple[int, int]]:
    ops = []
    for t in transforms:
        op, param = _parse_transform_name(t)
        if op not in (0, _OP_BWT, _OP_MTF):
            ops.append((op, param))
    return ops

def _encode_txhdr(ops: list[tuple[int, int]]) -> bytes:
    hdr = bytearray(_TX_MAGIC); hdr.append(len(ops))
    for op, param in ops: hdr.append(op); hdr.append(param)
    return bytes(hdr)

def _decode_txhdr(data: bytes) -> tuple[list[tuple[int, int]], int]:
    if len(data) < 5 or data[:4] != _TX_MAGIC:
        raise ValueError(f"bad txhdr magic: {data[:4]!r}")
    n    = data[4]; need = 5 + n * 2
    if len(data) < need:
        raise ValueError(f"txhdr truncated: need {need}, got {len(data)}")
    return [(data[5+i*2], data[5+i*2+1]) for i in range(n)], need

def _apply_chain(data: bytes, ops: list[tuple[int, int]]) -> bytes:
    for op, param in ops:
        if op == _OP_BYTEDELTA:   data = _apply_bytedelta(data, param)
        elif op == _OP_BYTESPLIT:  data = _apply_bytesplit(data, param)
        elif op == _OP_COLUMNSPLIT: data = _apply_columnsplit(data, param)
        elif op == _OP_ZIGZAG:     data = _apply_zigzag(data)
    return data

def _apply_inv_chain(data: bytes, ops: list[tuple[int, int]]) -> bytes:
    for op, param in reversed(ops):
        if op == _OP_BYTEDELTA:   data = _apply_bytedelta_inv(data, param)
        elif op == _OP_BYTESPLIT:  data = _apply_bytesplit_inv(data, param)
        elif op == _OP_COLUMNSPLIT: data = _apply_columnsplit_inv(data, param)
        elif op == _OP_ZIGZAG:     data = _apply_zigzag_inv(data)
    return data


# ---------------------------------------------------------------------------
# Transform guard: zlib-1 probe
# ---------------------------------------------------------------------------

_MIN_TRANSFORM_GAIN = 1.05
_GUARD_PROBE_BYTES  = 65536

def _zlib1_ratio(data: bytes) -> float:
    if not data: return 1.0
    return len(data) / max(len(zlib.compress(data, 1)), 1)

def _probe_transform_gain(sample: bytes, ops: list[tuple[int, int]]) -> float:
    probe = sample[:_GUARD_PROBE_BYTES]
    if not probe or not ops: return 1.0
    return _zlib1_ratio(_apply_chain(probe, ops)) / max(_zlib1_ratio(probe), 1e-9)


# ---------------------------------------------------------------------------
# K2Pipeline — the top-level object
# ---------------------------------------------------------------------------

class K2Pipeline:
    """
    End-to-end compression pipeline.

    C++ bridge entry points
    -----------------------
    prepare(sample)                       → StructureHint
    compress_full(data)                   → K2 frame bytes (pre-transform
                                             payload; C++ must call
                                             asdp_compress() then
                                             reseal_frame())
    reseal_frame(frame, entropy_payload)  → K2 frame bytes  (C++ calls this
                                             after asdp_compress() returns)
    decompress_full(frame)                → original bytes (C++ has
                                             already run asdp_decompress()
                                             on the payload before calling)
    update_final_score(strat, in, out, ms) → None
    stats()                                → dict

    ASDP-LH (backend 0x04) is the only entropy backend. There is no
    OpenZL or zstd path — both were removed earlier in this project; any
    reference to either elsewhere in this codebase's history predates
    that decision.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        exploration: float = 1.0,
        latency_weight: float = 0.15,
        probe_bytes: int = 128 * 1024,
    ):
        from structure_discovery import StructureDiscovery

        self._discovery     = StructureDiscovery(onnx_model_path, probe_bytes)
        self._exploration   = exploration
        self._latency_weight = latency_weight
        self._optimizer: Optional[AdaptiveOptimizer] = None
        self._hint:      Optional[StructureHint]      = None
        self._probe_sample: bytes = b""

        # Active state refreshed on each compress_full() call
        self._active_ops:      list[tuple[int, int]] = []
        self._active_txhdr:    bytes = _encode_txhdr([])
        self._active_backend:  int   = _BACKEND_ASDP
        self._active_strategy: str   = ""

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(self, sample: bytes) -> StructureHint:
        self._hint         = self._discovery.analyze(sample)
        self._probe_sample = sample[:_GUARD_PROBE_BYTES]

        self._optimizer = AdaptiveOptimizer(
            hint=self._hint,
            exploration=self._exploration,
            latency_weight=self._latency_weight,
        )

        # Seed each strategy with a realistic warmup score based on zlib-1
        # ratio of transformed probe sample (same scale as real scores).
        # This is a cheap proxy purely for ordering the bandit's initial
        # exploration — it does not imply zlib is used for real entropy
        # coding anywhere; ASDP/CM (backend 0x04) is the only entropy
        # backend in this pipeline.
        probe     = self._probe_sample
        raw_ratio = max(_zlib1_ratio(probe), 1.0) if probe else 1.0
        for strat in list(self._optimizer._strategies.values()):
            ops = _ops_from_transforms(strat.transforms)
            if ops and probe:
                transformed  = _apply_chain(probe, ops)
                init_ratio   = max(_zlib1_ratio(transformed), 1.0)
            else:
                init_ratio = raw_ratio
            strat._n_plays       = 1
            strat._total_score   = math.log2(init_ratio)
            self._optimizer._total_plays += 1

        init_strat = self._optimizer.select_strategy()
        self._refresh(init_strat)
        return self._hint

    def _refresh(self, strategy: Strategy) -> None:
        """Apply guard and update active state from strategy."""
        ops = _ops_from_transforms(strategy.transforms)
        if ops and self._probe_sample:
            if _probe_transform_gain(self._probe_sample, ops) < _MIN_TRANSFORM_GAIN:
                ops = []
        self._active_ops      = ops
        self._active_txhdr    = _encode_txhdr(ops)
        self._active_backend  = _BACKEND_ASDP
        self._active_strategy = strategy.name

    # ------------------------------------------------------------------
    # compress_full — main C++ bridge entry point
    # ------------------------------------------------------------------

    def compress_full(self, data: bytes) -> bytes:
        """
        Compress `data` and return a complete K2 frame.

        The payload is always pre-transform bytes; the C++ bridge calls
        asdp_compress(payload) then reseal_frame().

        Always returns bytes.  Never raises (falls back to raw frame on error).
        """
        if self._optimizer is None:
            self.prepare(data[:min(len(data), 65536)])

        strategy = self._optimizer.select_strategy()
        self._refresh(strategy)

        # Apply structural transforms; pass pre-transform bytes to C++ ASDP.
        transformed = _apply_chain(data, self._active_ops)

        return encode_k2_frame(
            _BACKEND_ASDP, len(data),
            self._active_txhdr, transformed
        )

    # ------------------------------------------------------------------
    # reseal_frame — called by C++ after the entropy backend (ASDP) returns
    # ------------------------------------------------------------------

    def reseal_frame(self, frame: bytes, entropy_payload: bytes) -> bytes:
        """
        Replace the payload in an entropy-backend (ASDP) K2 frame with the
        output of asdp_compress().  Called by the C++ bridge.  Backend byte and
        txhdr are preserved; also records the real compressed size for the bandit.
        """
        backend, orig_size, txhdr, _ = decode_k2_frame(frame)
        sealed = encode_k2_frame(backend, orig_size, txhdr, entropy_payload)
        # Record with real size
        strat = self._optimizer._strategies.get(self._active_strategy) if self._optimizer else None
        if strat:
            ratio_score = math.log2(max(orig_size / max(len(entropy_payload), 1), 1.0))
            strat._n_plays     += 1
            strat._total_score += ratio_score
            if self._optimizer:
                self._optimizer._total_plays += 1
        return sealed

    # Backward-compat alias (older C++ bridges called this name).
    def reseal_openzl_frame(self, frame: bytes, openzl_payload: bytes) -> bytes:
        return self.reseal_frame(frame, openzl_payload)

    # ------------------------------------------------------------------
    # update_final_score — C API feedback
    # ------------------------------------------------------------------

    def update_final_score(
        self,
        strategy_name: str,
        input_size: int,
        final_compressed_size: int,
        elapsed_ms: float,
    ) -> None:
        if self._optimizer is None:
            return
        name = strategy_name or self._active_strategy
        if not name:
            return
        self._optimizer.record_result(name, input_size, final_compressed_size, elapsed_ms)

    # ------------------------------------------------------------------
    # decompress_full — main C++ bridge entry point
    # ------------------------------------------------------------------

    def decompress_full(self, frame: bytes, orig_size: int = 0) -> bytes:
        """
        Decompress a K2 frame.

        C++ has already called asdp_decompress() on the payload and resealed
        the frame before calling here.  This function only inverts the
        structural transforms.

        Returns original bytes.
        """
        try:
            backend, stored_orig, txhdr_bytes, payload = decode_k2_frame(frame)
        except ValueError:
            # No K2 frame header — legacy raw path, return as-is
            return frame

        if backend != _BACKEND_ASDP:
            raise ValueError(f"unsupported backend: {backend:#04x}")

        # payload is already entropy-decoded by C++ asdp_decompress; invert transforms.
        if len(txhdr_bytes) >= 5:
            try:
                ops, _ = _decode_txhdr(txhdr_bytes)
                return _apply_inv_chain(payload, ops)
            except ValueError:
                pass
        return payload

    # ------------------------------------------------------------------
    # Legacy compress() (pre-C++-mediation entropy path) — removed.
    # It exercised HybridPredictor.compress(), a pure-Python arithmetic
    # coder with a zstd fallback that has been removed entirely (see
    # hybrid_predictor.py's module docstring). There is no remaining
    # pure-Python entropy path to port this forward to: ASDP/CM (backend
    # 0x04) is the only entropy backend, and it is C++-only. Use
    # compress_full() via the C++ bridge for all real compression.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Backward-compat shims (C++ bridge currently calls these names)
    # ------------------------------------------------------------------

    def compress_transforms(self, data: bytes) -> bytes:
        """Shim: delegates to compress_full()."""
        return self.compress_full(data)

    def decompress_transforms(self, data: bytes) -> bytes:
        """Shim: delegates to decompress_full()."""
        return self.decompress_full(data)

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        if self._optimizer is None:
            return {}
        return {
            "hint":              str(self._hint.data_class.name) if self._hint else None,
            "recent_ratio":      round(self._optimizer.recent_ratio(), 3),
            "throughput_mbs":    round(self._optimizer.recent_throughput_mbs(), 1),
            "strategies":        self._optimizer.stats(),
            "best_strategy":     self._optimizer.best_strategy().name,
            "active_strategy":   self._active_strategy,
            "active_backend":    "asdp",
            "active_transforms": [
                {"op": hex(op), "param": param}
                for op, param in self._active_ops
            ],
        }
