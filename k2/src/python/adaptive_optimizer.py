"""
k2/src/python/adaptive_optimizer.py

Adaptive Optimization Layer — K2
----------------------------------
K2Pipeline owns the full compression decision: backend selection, transforms,
and entropy coding.  The C++ bridge is a thin shim that executes the OpenZL
call when K2Pipeline requests it.

Backend selection per DataClass:
  TIMESERIES / INTEGER  → K2 transforms (delta/zigzag) + OpenZL LZ
  FLOAT                 → K2 transforms (bytesplit)    + zstd
  TEXT                  → zstd directly (no transforms)
  COLUMNAR              → K2 transforms (columnsplit)  + zstd
  BINARY_BLOB / UNKNOWN → zstd directly

K2 Frame Format
---------------
Offset  Size  Field
0       4     magic  b'K2\\xf7\\x01'
4       1     version  0x01
5       1     backend  0x01=OpenZL  0x02=zstd  0x03=zlib-fallback
6       1     flags    bit0=transforms_applied  rest reserved
7       1     reserved 0x00
8       8     orig_size  uint64 LE
16      2     txhdr_len  uint16 LE
18      N     transform_header (K2T\\x01 ... same format as before)
18+N    ...   backend payload

The C++ bridge reads the frame, and:
  - backend=OpenZL → calls CCtx::compressSerial(payload), replaces payload,
                     re-seals frame.  On decompress: DCtx then K2 inv.
  - backend=zstd   → payload is already a complete zstd frame.
                     No CCtx call.  On decompress: zstd_decompress then K2 inv.
  - backend=zlib   → same as zstd but zlib (fallback when zstandard absent).
"""

from __future__ import annotations

import math
import struct
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from structure_discovery import StructureHint, DataClass
from hybrid_predictor import HybridConfig, PredictorMode

try:
    import zstandard as zstd_mod
    _ZSTD_AVAILABLE = True
    _zstd_cctx = zstd_mod.ZstdCompressor(level=3)
    _zstd_dctx = zstd_mod.ZstdDecompressor()
except ImportError:
    _ZSTD_AVAILABLE = False
    _zstd_cctx = None
    _zstd_dctx = None


# ---------------------------------------------------------------------------
# K2 frame constants
# ---------------------------------------------------------------------------

_FRAME_MAGIC      = b"K2\xf7\x01"
_FRAME_VERSION    = 0x01
_BACKEND_OPENZL   = 0x01
_BACKEND_ZSTD     = 0x02
_BACKEND_ZLIB     = 0x03   # fallback when zstandard not installed
_FLAG_TRANSFORMS  = 0x01
_FRAME_HDR_STRUCT = struct.Struct("<4sBBBBQH")   # 18 bytes
_FRAME_HDR_SIZE   = _FRAME_HDR_STRUCT.size       # 18


def _best_entropy_backend() -> int:
    """Return the backend ID to use for zstd-class data."""
    return _BACKEND_ZSTD if _ZSTD_AVAILABLE else _BACKEND_ZLIB


def _entropy_compress(data: bytes) -> bytes:
    """Compress with zstd (or zlib fallback)."""
    if _ZSTD_AVAILABLE:
        return _zstd_cctx.compress(data)
    return zlib.compress(data, 3)


def _entropy_decompress(data: bytes, max_length: int) -> bytes:
    """Decompress zstd or zlib frame."""
    if _ZSTD_AVAILABLE:
        return _zstd_dctx.decompress(data, max_length=max_length)
    return zlib.decompress(data)


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

    For backend=OPENZL the payload is pre-transform bytes; the C++ bridge
    replaces it with the OpenZL-compressed output and re-seals the frame.
    For backend=ZSTD/ZLIB the payload is already the final compressed bytes.
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
    name: str
    predictor_mode: PredictorMode
    alpha: float
    chunk_size: int
    zstd_level: int
    transforms: list[str]
    backend: int = field(default=0)  # 0 = use class default

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
    strategies: list[Strategy] = []
    cls = hint.data_class
    w   = hint.element_size

    if cls == DataClass.TIMESERIES:
        strategies += [
            Strategy("ts_delta_zigzag", PredictorMode.LINEAR_CMA, 0.3, 4096, 3,
                     [f"bytedelta-{w}", "zigzag"], _BACKEND_OPENZL),
            Strategy("ts_delta_only",   PredictorMode.LINEAR_CMA, 0.3, 4096, 3,
                     [f"bytedelta-{w}"], _BACKEND_OPENZL),
            Strategy("ts_raw",          PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                     [], _BACKEND_OPENZL),
        ]
    elif cls == DataClass.INTEGER_ARRAY:
        strategies += [
            Strategy("int_delta",        PredictorMode.LINEAR_CMA, 0.3, 4096, 3,
                     [f"bytedelta-{w}"], _BACKEND_OPENZL),
            Strategy("int_delta_zigzag", PredictorMode.LINEAR_CMA, 0.3, 4096, 3,
                     [f"bytedelta-{w}", "zigzag"], _BACKEND_OPENZL),
            Strategy("int_raw",          PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                     [], _BACKEND_OPENZL),
        ]
    elif cls == DataClass.FLOAT_ARRAY:
        eb = _best_entropy_backend()
        strategies += [
            Strategy("fp_split_delta",  PredictorMode.LINEAR_CMA, 0.25, 8192, 3,
                     [f"bytesplit-{w}", f"bytedelta-{w}"], eb),
            Strategy("fp_split_only",   PredictorMode.LINEAR_CMA, 0.25, 4096, 3,
                     [f"bytesplit-{w}"], eb),
            Strategy("fp_raw",          PredictorMode.PASSTHROUGH, 0.0,  4096, 3,
                     [], eb),
        ]
    elif cls == DataClass.COLUMNAR:
        s  = hint.column_stride or 8
        eb = _best_entropy_backend()
        strategies += [
            Strategy("col_split_delta", PredictorMode.LINEAR_CMA, 0.35, 4096, 3,
                     [f"columnsplit-{s}", "bytedelta-4"], eb),
            Strategy("col_split_only",  PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                     [f"columnsplit-{s}"], eb),
            Strategy("col_raw",         PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                     [], eb),
        ]
    elif cls == DataClass.TEXT:
        eb = _best_entropy_backend()
        strategies += [
            Strategy("text_cma",         PredictorMode.LINEAR_CMA, 0.45, 2048, 3,
                     [], eb),
            Strategy("text_passthrough", PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                     [], eb),
        ]
    else:
        # BINARY_BLOB / UNKNOWN / MIXED
        eb = _best_entropy_backend()
        strategies += [
            Strategy("blob_raw", PredictorMode.PASSTHROUGH, 0.0, 4096, 3, [], eb),
        ]

    strategies.append(
        Strategy("raw_fallback", PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                 [], _best_entropy_backend())
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
            self._strategies["raw_fallback"] = Strategy(
                "raw_fallback", PredictorMode.PASSTHROUGH, 0.0, 4096, 3,
                [], _best_entropy_backend()
            )
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

    def get_config(self, strategy: Strategy) -> HybridConfig:
        return HybridConfig(
            mode=strategy.predictor_mode,
            alpha=strategy.alpha,
            chunk_size=strategy.chunk_size,
            use_arithmetic=(strategy.predictor_mode != PredictorMode.PASSTHROUGH),
        )

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

    def tune_alpha(self, strategy: Strategy, step: float = 0.02) -> float:
        with self._lock:
            if len(self._window) < 4:
                return strategy.alpha
            recent = list(self._window)[-4:]
            scores = [r.score(self._latency_weight) for r in recent]
            trend  = scores[-1] - scores[0]
            if trend > 0.05:
                new_alpha = min(1.0, strategy.alpha + step)
            elif trend < -0.05:
                new_alpha = max(0.0, strategy.alpha - step)
            else:
                new_alpha = strategy.alpha
            strategy.alpha = new_alpha
            return new_alpha

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
_OP_ZSTD        = 0xFF   # never stored

def _parse_transform_name(name: str) -> tuple[int, int]:
    if name == "zigzag":   return (_OP_ZIGZAG, 0)
    if name == "bwt":      return (_OP_BWT, 0)
    if name == "mtf":      return (_OP_MTF, 0)
    if name == "zstd":     return (_OP_ZSTD, 0)
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
        if op not in (0, _OP_ZSTD, _OP_BWT, _OP_MTF):
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
    prepare(sample)                             → StructureHint
    compress_full(data)                         → K2 frame bytes
      For OpenZL-backend data: frame contains pre-transform bytes as payload;
      C++ must call reseal_openzl_frame() after running CCtx.
      For zstd-backend data: frame is complete, C++ passes it through unchanged.
    reseal_openzl_frame(frame, openzl_payload)  → K2 frame bytes  (C++ calls this)
    decompress_full(frame)                      → original bytes
      For OpenZL-backend: C++ runs DCtx first, passes result to decompress_full.
      For zstd-backend: C++ passes raw frame bytes, Python handles everything.
    update_final_score(strat, in, out, ms)      → None
    stats()                                     → dict
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

        self._discovery     = StructureDiscovery(onnx_model_path, probe_bytes)
        self._exploration   = exploration
        self._latency_weight = latency_weight
        self._optimizer: Optional[AdaptiveOptimizer] = None
        self._predictor: Optional[HybridPredictor]   = None
        self._hint:      Optional[StructureHint]      = None
        self._probe_sample: bytes = b""

        # Active state refreshed on each compress_full() call
        self._active_ops:      list[tuple[int, int]] = []
        self._active_txhdr:    bytes = _encode_txhdr([])
        self._active_backend:  int   = _best_entropy_backend()
        self._active_strategy: str   = ""

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(self, sample: bytes) -> StructureHint:
        from hybrid_predictor import HybridPredictor

        self._hint         = self._discovery.analyze(sample)
        self._probe_sample = sample[:_GUARD_PROBE_BYTES]

        self._optimizer = AdaptiveOptimizer(
            hint=self._hint,
            exploration=self._exploration,
            latency_weight=self._latency_weight,
        )

        # Seed each strategy with a realistic warmup score based on zlib-1
        # ratio of transformed probe sample (same scale as real scores).
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
        cfg        = self._optimizer.get_config(init_strat)
        self._predictor = HybridPredictor(cfg)
        self._predictor.train(sample[:min(len(sample), 32768)])
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
        self._active_backend  = strategy.backend or _best_entropy_backend()
        self._active_strategy = strategy.name

    # ------------------------------------------------------------------
    # compress_full — main C++ bridge entry point
    # ------------------------------------------------------------------

    def compress_full(self, data: bytes) -> bytes:
        """
        Compress `data` and return a complete K2 frame.

        Backend=ZSTD/ZLIB: frame payload is the final compressed bytes.
          C++ bridge passes the frame through unchanged.

        Backend=OPENZL: frame payload is pre-transformed bytes.
          C++ bridge must call CCtx::compressSerial(payload) and then
          call reseal_openzl_frame(frame, openzl_output) before writing.

        Always returns bytes.  Never raises (falls back to raw on error).
        """
        if self._optimizer is None:
            self.prepare(data[:min(len(data), 65536)])

        strategy = self._optimizer.select_strategy()
        self._refresh(strategy)

        # Apply structural transforms
        transformed = _apply_chain(data, self._active_ops)

        if self._active_backend == _BACKEND_OPENZL:
            # C++ will run CCtx; we return the pre-transform payload.
            # The C++ bridge calls reseal_openzl_frame() after CCtx returns.
            payload = transformed
        else:
            # Run entropy compression in Python.
            t0      = time.perf_counter()
            payload = _entropy_compress(transformed)
            elapsed = (time.perf_counter() - t0) * 1000.0
            # Record score now (we have the final size).
            self._optimizer.record_result(
                strategy.name, len(data), len(payload), elapsed
            )
            if strategy._n_plays % 10 == 0:
                self._optimizer.tune_alpha(strategy)

        return encode_k2_frame(
            self._active_backend, len(data),
            self._active_txhdr, payload
        )

    # ------------------------------------------------------------------
    # reseal_openzl_frame — called by C++ after CCtx returns
    # ------------------------------------------------------------------

    def reseal_openzl_frame(self, frame: bytes, openzl_payload: bytes) -> bytes:
        """
        Replace the payload in an OPENZL-backend K2 frame with the output
        of CCtx::compressSerial().  Called by the C++ bridge.
        Also records the real compressed size for the bandit.
        """
        backend, orig_size, txhdr, _ = decode_k2_frame(frame)
        sealed = encode_k2_frame(backend, orig_size, txhdr, openzl_payload)
        # Record with real size
        strat = self._optimizer._strategies.get(self._active_strategy) if self._optimizer else None
        if strat:
            ratio_score = math.log2(max(orig_size / max(len(openzl_payload), 1), 1.0))
            strat._n_plays     += 1
            strat._total_score += ratio_score
            if self._optimizer:
                self._optimizer._total_plays += 1
        return sealed

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
        strat = self._optimizer._strategies.get(name)
        if strat and strat._n_plays % 10 == 0:
            self._optimizer.tune_alpha(strat)

    # ------------------------------------------------------------------
    # decompress_full — main C++ bridge entry point
    # ------------------------------------------------------------------

    def decompress_full(self, frame_or_openzl_payload: bytes, orig_size: int = 0) -> bytes:
        """
        Decompress a K2 frame.

        For ZSTD/ZLIB backends: pass the raw K2 frame bytes.
          Python handles all decompression.

        For OPENZL backend: C++ has already called DCtx::decompressSerial()
          and passes the resulting bytes here (which still contain the K2
          frame header + txhdr + now-decompressed payload).
          `orig_size` must be provided so we can bound zlib decompression.

        Returns original bytes.
        """
        try:
            backend, stored_orig, txhdr_bytes, payload = decode_k2_frame(
                frame_or_openzl_payload
            )
        except ValueError:
            # No K2 frame header — legacy raw path, return as-is
            return frame_or_openzl_payload

        actual_orig = stored_orig or orig_size

        # Entropy decompress
        if backend == _BACKEND_OPENZL:
            # payload is already entropy-decoded by C++ DCtx; use as-is
            transformed = payload
        else:
            transformed = _entropy_decompress(
                payload, max_length=max(actual_orig * 2, 1 << 20)
            )

        # Inverse transforms
        if len(txhdr_bytes) >= 5:
            try:
                ops, _ = _decode_txhdr(txhdr_bytes)
                return _apply_inv_chain(transformed, ops)
            except ValueError:
                pass
        return transformed

    # ------------------------------------------------------------------
    # Legacy compress/decompress (no-OpenZL path, used by direct tests)
    # ------------------------------------------------------------------

    def compress(self, data: bytes) -> bytes:
        """Full compress without C++ mediation (uses entropy backend only)."""
        if self._optimizer is None or self._predictor is None:
            self.prepare(data[:min(len(data), 65536)])
        strategy = self._optimizer.select_strategy()
        cfg = self._optimizer.get_config(strategy)
        self._predictor._cfg = cfg
        t0 = time.perf_counter()
        compressed = self._predictor.compress(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._optimizer.record_result(strategy.name, len(data), len(compressed), elapsed_ms)
        if strategy._n_plays % 10 == 0:
            self._optimizer.tune_alpha(strategy)
        return compressed

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
        backend_name = {
            _BACKEND_OPENZL: "openzl",
            _BACKEND_ZSTD:   "zstd",
            _BACKEND_ZLIB:   "zlib",
        }.get(self._active_backend, "unknown")
        return {
            "hint":             str(self._hint.data_class.name) if self._hint else None,
            "recent_ratio":     round(self._optimizer.recent_ratio(), 3),
            "throughput_mbs":   round(self._optimizer.recent_throughput_mbs(), 1),
            "strategies":       self._optimizer.stats(),
            "best_strategy":    self._optimizer.best_strategy().name,
            "active_strategy":  self._active_strategy,
            "active_backend":   backend_name,
            "zstd_available":   _ZSTD_AVAILABLE,
            "active_transforms": [
                {"op": hex(op), "param": param}
                for op, param in self._active_ops
            ],
        }
