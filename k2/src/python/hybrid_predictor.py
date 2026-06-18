"""
k2/src/python/hybrid_predictor.py

Hybrid Prediction Engine — K2
-------------------------------
Combines classical entropy coding (Zstd-derived symbol stats) with a
small neural context mixer.  The mixer produces a blended probability
distribution over next-byte / next-token that is fed into arithmetic
coding before the Zstd back-end.

Architecture options (selectable at runtime):
  - LSTM:        Fast, low memory, decent context (~256 tokens)
  - MiniTransformer: Better long-range, ~4 layers, 64-dim, causal masking
  - Linear (CMA): Context-Mixing Adapter — weighted blend of N statistical
                   sub-models; zero neural overhead, very fast
  - PassThrough: Disable neural layer entirely (pure Zstd path)

The neural models are trained offline (see train_predictor.py) and
exported to ONNX for inference via ONNX Runtime or loaded as PyTorch
state-dicts via a bundled TorchScript module.

Integration with OpenZL:
  1. Chunk data at control-point boundaries (as returned by the OpenZL
     graph's source nodes).
  2. For each chunk, call `predict_probs(context, chunk)`.
  3. Pass the returned probability array to an arithmetic coder.
  4. The arithmetic coder's output replaces (or supplements) the Zstd
     back-end for that chunk.
"""

from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

# Optional heavy deps
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Probability blending
# ---------------------------------------------------------------------------

def _log_mix(p_stat: np.ndarray, p_neural: np.ndarray, alpha: float) -> np.ndarray:
    """
    Log-linear interpolation (log-domain mixing) of two distributions.
    alpha=0 → pure statistical; alpha=1 → pure neural.
    """
    log_p = (1.0 - alpha) * np.log(p_stat + 1e-10) + alpha * np.log(p_neural + 1e-10)
    p = np.exp(log_p)
    return p / p.sum()


# ---------------------------------------------------------------------------
# Statistical baseline: byte-unigram + order-1 model
# ---------------------------------------------------------------------------

class StatisticalModel:
    """
    Lightweight order-1 Markov model (256×256 count table).
    Acts as the 'Zstd-side' of the hybrid.
    """

    def __init__(self, smoothing: float = 0.5):
        self._smoothing = smoothing
        # Prior: uniform over 256 bytes
        self._counts = np.full((256, 256), smoothing, dtype=np.float64)

    def update(self, context_byte: int, next_byte: int) -> None:
        self._counts[context_byte, next_byte] += 1.0

    def predict(self, context_byte: int) -> np.ndarray:
        row = self._counts[context_byte]
        return row / row.sum()

    def train_on(self, data: bytes) -> None:
        prev = 0
        for b in data:
            self.update(prev, b)
            prev = b

    def reset(self) -> None:
        self._counts = np.full((256, 256), self._smoothing, dtype=np.float64)


# ---------------------------------------------------------------------------
# Neural back-ends
# ---------------------------------------------------------------------------

class NeuralBackend(ABC):
    @abstractmethod
    def predict(self, context: bytes) -> np.ndarray:
        """Return a probability distribution over 256 byte values."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset any recurrent state (call at chunk boundaries)."""


# --- Linear Context-Mixing Adapter (no torch required) ---

class LinearCMA(NeuralBackend):
    """
    Blends K order-N statistical sub-models with learned weights.
    Very fast; equivalent to a single-layer linear network over model outputs.
    """

    ORDERS = (1, 2, 3, 4)

    def __init__(self, smoothing: float = 0.1):
        self._models: list[dict] = [
            {"order": o, "counts": {}, "smoothing": smoothing}
            for o in self.ORDERS
        ]
        self._weights = np.ones(len(self.ORDERS)) / len(self.ORDERS)
        self._losses = np.zeros(len(self.ORDERS))
        self._lr = 0.01
        self._context: bytes = b"\x00" * max(self.ORDERS)

    def _get_counts(self, model, key):
        if key not in model["counts"]:
            model["counts"][key] = np.full(256, model["smoothing"])
        return model["counts"][key]

    def _predict_model(self, model, context: bytes) -> np.ndarray:
        key = context[-model["order"]:]
        counts = self._get_counts(model, key)
        return counts / counts.sum()

    def predict(self, context: bytes) -> np.ndarray:
        ctx = (self._context + context)[-max(self.ORDERS):]
        probs = np.stack([
            self._predict_model(m, ctx) for m in self._models
        ])  # (K, 256)
        w_stable = self._weights - self._weights.max()  # log-sum-exp trick
        w = np.exp(w_stable)
        w_sum = w.sum()
        if w_sum == 0 or not np.isfinite(w_sum):
            w = np.ones(len(self._weights)) / len(self._weights)
        else:
            w /= w_sum
        return (w[:, None] * probs).sum(axis=0)

    def update_weights(self, true_byte: int) -> None:
        """Online weight update via exponentiated gradient."""
        for i, m in enumerate(self._models):
            ctx = self._context[-m["order"]:]
            p = self._predict_model(m, ctx)
            loss = -math.log(p[true_byte] + 1e-10)
            self._losses[i] = 0.9 * self._losses[i] + 0.1 * loss
        # Exponentiated gradient descent on weights
        grad = self._losses - self._losses.mean()
        self._weights -= self._lr * grad
        self._weights = np.clip(self._weights, -20.0, 20.0)  # prevent overflow
        # Also update sub-model counts
        for m in self._models:
            key = self._context[-m["order"]:]
            counts = self._get_counts(m, key)
            counts[true_byte] += 1.0

    def reset_state(self) -> None:
        self._context = b"\x00" * max(self.ORDERS)

    def advance(self, byte_val: int) -> None:
        self._context = (self._context + bytes([byte_val]))[-max(self.ORDERS):]


# --- LSTM backend (requires torch) ---

if TORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        """
        Lightweight LSTM-based next-byte predictor.
        Input:  byte-embedding (256 → embed_dim)
        Hidden: 2-layer LSTM, hidden_size
        Output: linear → 256 logits (softmax for probabilities)

        Parameter count examples:
          embed=32, hidden=128 → ~200K params  (fast, ~2 MB)
          embed=64, hidden=256 → ~800K params  (better quality, ~6 MB)
        """
        def __init__(self, embed_dim: int = 32, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.embed = nn.Embedding(256, embed_dim)
            self.lstm = nn.LSTM(
                embed_dim, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 256)
            self._state: Optional[tuple] = None

        def forward(self, x: "torch.Tensor",
                    state=None) -> tuple["torch.Tensor", tuple]:
            emb = self.embed(x)          # (B, T, embed)
            out, state = self.lstm(emb, state)
            logits = self.head(out)      # (B, T, 256)
            return logits, state

        def reset_state(self) -> None:
            self._state = None

        def predict_next(self, context_ids: list[int]) -> np.ndarray:
            """Return softmax probs for the next byte given context_ids."""
            self.eval()
            with torch.no_grad():
                x = torch.tensor([context_ids], dtype=torch.long)
                logits, self._state = self.forward(x, self._state)
                probs = torch.softmax(logits[0, -1], dim=-1).numpy()
            return probs

    class MiniTransformer(nn.Module):
        """
        Causal transformer, 4 layers, 64-dim, 4 heads.
        Context window: 512 bytes.
        ~1.2M parameters; ~9 MB on disk.
        """
        def __init__(self, d_model: int = 64, nhead: int = 4,
                     num_layers: int = 4, max_len: int = 512):
            super().__init__()
            self.embed = nn.Embedding(256, d_model)
            self.pos   = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 256)
            self._max_len = max_len

        def _causal_mask(self, sz: int) -> "torch.Tensor":
            return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.embed(x) + self.pos(pos)
            mask = self._causal_mask(T).to(x.device)
            h = self.encoder(h, mask=mask)
            return self.head(h)   # (B, T, 256)

        def reset_state(self) -> None:
            pass  # Transformer is stateless (full context each call)

        def predict_next(self, context_ids: list[int]) -> np.ndarray:
            ctx = context_ids[-self._max_len:]
            self.eval()
            with torch.no_grad():
                x = torch.tensor([ctx], dtype=torch.long)
                logits = self.forward(x)
                probs = torch.softmax(logits[0, -1], dim=-1).numpy()
            return probs


# --- ONNX wrapper (for exported LSTM/Transformer) ---

class ONNXPredictor(NeuralBackend):
    def __init__(self, model_path: str, context_len: int = 64):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._ctx_len = context_len
        self._context: list[int] = [0] * context_len

    def predict(self, context: bytes) -> np.ndarray:
        for b in context:
            self._context.append(b)
            self._context = self._context[-self._ctx_len:]
        inp = np.array([self._context], dtype=np.int64)
        inp_name = self._sess.get_inputs()[0].name
        logits = self._sess.run(None, {inp_name: inp})[0]  # (1, ctx, 256)
        probs = _softmax_np(logits[0, -1])
        return probs

    def reset_state(self) -> None:
        self._context = [0] * self._ctx_len


def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Arithmetic coder (pure Python, streaming-capable)
# ---------------------------------------------------------------------------

class ArithmeticEncoder:
    """
    Simple integer arithmetic encoder.
    Precision: 32-bit integer range [0, 2^32).
    """

    FULL  = 1 << 32
    HALF  = 1 << 31
    QRTR  = 1 << 30
    _3QTR = 3 * (1 << 30)

    def __init__(self):
        self._low = 0
        self._high = self.FULL - 1
        self._pending = 0
        self._bits: list[int] = []

    def _emit(self, bit: int) -> None:
        self._bits.append(bit)
        for _ in range(self._pending):
            self._bits.append(1 - bit)
        self._pending = 0

    def encode_symbol(self, probs: np.ndarray, symbol: int) -> None:
        """Encode one symbol given a probability distribution."""
        total = self._high - self._low + 1
        # CDF
        cum = np.zeros(257, dtype=np.float64)
        cum[1:] = np.cumsum(probs)
        cum = np.clip(cum * total, 0, total).astype(np.int64)
        # Ensure strict monotonicity (floating point can collapse adjacent values)
        for i in range(1, 257):
            if cum[i] <= cum[i - 1]:
                cum[i] = cum[i - 1] + 1
        cum = np.clip(cum, 0, total)
        lo = self._low + int(cum[symbol])
        hi = self._low + int(cum[symbol + 1]) - 1

        self._low, self._high = lo, hi

        while True:
            if self._high < self.HALF:
                self._emit(0)
                self._low  *= 2
                self._high  = self._high * 2 + 1
            elif self._low >= self.HALF:
                self._emit(1)
                self._low  = (self._low  - self.HALF) * 2
                self._high = (self._high - self.HALF) * 2 + 1
            elif self._low >= self.QRTR and self._high < self._3QTR:
                self._pending += 1
                self._low  = (self._low  - self.QRTR) * 2
                self._high = (self._high - self.QRTR) * 2 + 1
            else:
                break

    def flush(self) -> bytes:
        self._pending += 1
        self._emit(1 if self._low >= self.QRTR else 0)
        # Pack bits into bytes
        bits = self._bits
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j, b in enumerate(bits[i:i+8]):
                byte |= b << (7 - j)
            out.append(byte)
        return bytes(out)


# ---------------------------------------------------------------------------
# Arithmetic decoder — mirrors ArithmeticEncoder exactly
# ---------------------------------------------------------------------------

class ArithmeticDecoder:
    """
    Integer arithmetic decoder.
    Precision: 32-bit range [0, 2^32).

    Must be used with the same probability model and symbol count as the
    corresponding ArithmeticEncoder.  The model state must be synchronised
    symbol-by-symbol so both sides produce identical CDF tables.
    """

    FULL  = 1 << 32
    HALF  = 1 << 31
    QRTR  = 1 << 30
    _3QTR = 3 * (1 << 30)

    def __init__(self, data: bytes):
        # Unpack all bits MSB-first (mirrors ArithmeticEncoder.flush)
        self._bits: list[int] = []
        for byte in data:
            for i in range(7, -1, -1):
                self._bits.append((byte >> i) & 1)
        self._pos = 0

        # Initialise value register with the first 32 bits
        self._low   = 0
        self._high  = self.FULL - 1
        self._value = 0
        for _ in range(32):
            self._value = (self._value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self._pos < len(self._bits):
            b = self._bits[self._pos]
            self._pos += 1
            return b
        return 0  # pad with zeros beyond end of stream

    def decode_symbol(self, probs: np.ndarray) -> int:
        """
        Decode one symbol given a probability distribution over 256 values.
        The distribution must be identical to the one used during encoding.
        """
        total = self._high - self._low + 1

        # Build integer CDF — must exactly mirror ArithmeticEncoder.encode_symbol
        cum = np.zeros(257, dtype=np.float64)
        cum[1:] = np.cumsum(probs)
        cum_int = np.clip(cum * total, 0, total).astype(np.int64)
        for i in range(1, 257):
            if cum_int[i] <= cum_int[i - 1]:
                cum_int[i] = cum_int[i - 1] + 1
        cum_int = np.clip(cum_int, 0, total)

        # Find the symbol whose range contains (value - low)
        scaled = self._value - self._low
        lo, hi = 0, 255
        while lo < hi:
            mid = (lo + hi) // 2
            if cum_int[mid + 1] <= scaled:
                lo = mid + 1
            else:
                hi = mid
        symbol = lo

        # Update range to the symbol's sub-interval
        self._high = self._low + int(cum_int[symbol + 1]) - 1
        self._low  = self._low + int(cum_int[symbol])

        # Renormalise — mirror encoder, but read bits instead of emitting them
        while True:
            if self._high < self.HALF:
                self._low   *= 2
                self._high   = self._high  * 2 + 1
                self._value  = self._value * 2 + self._read_bit()
            elif self._low >= self.HALF:
                self._low   = (self._low   - self.HALF) * 2
                self._high  = (self._high  - self.HALF) * 2 + 1
                self._value = (self._value - self.HALF) * 2 + self._read_bit()
            elif self._low >= self.QRTR and self._high < self._3QTR:
                self._low   = (self._low   - self.QRTR) * 2
                self._high  = (self._high  - self.QRTR) * 2 + 1
                self._value = (self._value - self.QRTR) * 2 + self._read_bit()
            else:
                break

        return symbol


# ---------------------------------------------------------------------------
# Main HybridPredictor façade
# ---------------------------------------------------------------------------

class PredictorMode(Enum):
    LINEAR_CMA       = auto()
    LSTM             = auto()
    MINI_TRANSFORMER = auto()
    ONNX             = auto()
    PASSTHROUGH      = auto()   # pure Zstd


@dataclass
class HybridConfig:
    mode: PredictorMode = PredictorMode.LINEAR_CMA
    alpha: float = 0.4          # neural weight in log-mix (0=stat only, 1=neural only)
    chunk_size: int = 4096      # bytes per arithmetic-coded chunk
    context_window: int = 64    # bytes of context fed to neural model
    use_arithmetic: bool = False # False → skip arith coding, just return probs
    model_path: Optional[str] = None  # for LSTM/Transformer/ONNX modes


class HybridPredictor:
    """
    Top-level compression predictor.

    Usage:
        predictor = HybridPredictor(config)
        predictor.train(representative_data)      # offline or online
        compressed = predictor.compress(data)
        restored   = predictor.decompress(compressed)
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        self._cfg = config or HybridConfig()
        self._stat = StatisticalModel()
        self._neural: Optional[NeuralBackend] = None
        self._init_neural()

    def _init_neural(self) -> None:
        mode = self._cfg.mode
        if mode == PredictorMode.LINEAR_CMA:
            self._neural = LinearCMA()
        elif mode == PredictorMode.LSTM:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch required for LSTM mode")
            model = LSTMPredictor()
            if self._cfg.model_path:
                model.load_state_dict(
                    torch.load(self._cfg.model_path, map_location="cpu")
                )
            # Wrap as NeuralBackend
            self._neural = _TorchWrapper(model, self._cfg.context_window)
        elif mode == PredictorMode.MINI_TRANSFORMER:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch required for MiniTransformer mode")
            model = MiniTransformer()
            if self._cfg.model_path:
                model.load_state_dict(
                    torch.load(self._cfg.model_path, map_location="cpu")
                )
            self._neural = _TorchWrapper(model, self._cfg.context_window)
        elif mode == PredictorMode.ONNX:
            if not self._cfg.model_path:
                raise ValueError("model_path required for ONNX mode")
            self._neural = ONNXPredictor(
                self._cfg.model_path, self._cfg.context_window
            )
        else:
            self._neural = None  # PASSTHROUGH

    def train(self, data: bytes) -> None:
        """Online-train statistical model on representative data."""
        self._stat.train_on(data)
        # For LinearCMA, also warm up sub-models
        if isinstance(self._neural, LinearCMA):
            prev = 0
            for b in data:
                self._neural.advance(prev)
                self._neural.update_weights(b)
                prev = b

    def predict_probs(self, context: bytes, next_byte: Optional[int] = None
                      ) -> np.ndarray:
        """
        Return blended P(next_byte | context) distribution over [0,255].
        If next_byte provided, also update online models.
        """
        ctx_byte = context[-1] if context else 0
        p_stat = self._stat.predict(ctx_byte)

        if self._neural is None:
            return p_stat

        p_neural = self._neural.predict(context)
        blended = _log_mix(p_stat, p_neural, self._cfg.alpha)

        if next_byte is not None:
            self._stat.update(ctx_byte, next_byte)
            if isinstance(self._neural, LinearCMA):
                self._neural.update_weights(next_byte)
                self._neural.advance(next_byte)

        return blended

    def compress_chunk(self, data: bytes) -> bytes:
        """
        Compress a single chunk using the hybrid model + arithmetic coding.

        Wire format (per chunk):
          4 bytes  chunk_payload_length  uint32 big-endian
          1 byte   flag: 0x01 = arithmetic coded
                         0x02 = zstd fallback
                         0x03 = raw (no compression)
          N bytes  payload

        The flag byte lets decompress_chunk route correctly without
        knowing which path was taken at encode time.
        """
        enc = ArithmeticEncoder()
        context = b"\x00" * self._cfg.context_window

        for byte_val in data:
            probs = self.predict_probs(context, next_byte=byte_val)
            enc.encode_symbol(probs, byte_val)
            context = (context + bytes([byte_val]))[-self._cfg.context_window:]

        encoded = enc.flush()

        # Fallback: if arithmetic coding expanded the data, try zstd then raw
        if len(encoded) >= len(data):
            if ZSTD_AVAILABLE:
                cctx = zstd.ZstdCompressor(level=3)
                fallback = cctx.compress(data)
                if len(fallback) < len(data):
                    payload = b"\x02" + fallback
                    return struct.pack(">I", len(payload)) + payload
            # Last resort: store raw
            payload = b"\x03" + data
            return struct.pack(">I", len(payload)) + payload

        payload = b"\x01" + encoded
        return struct.pack(">I", len(payload)) + payload

    def decompress_chunk(self, data: bytes, original_len: int) -> bytes:
        """
        Decompress one chunk produced by compress_chunk().
        `data` starts immediately after the 4-byte length header.
        `original_len` is the number of bytes to recover.
        """
        if not data:
            raise ValueError("empty chunk payload")

        flag = data[0]
        payload = data[1:]

        if flag == 0x01:
            # Arithmetic-coded: decode symbol by symbol with same model state
            dec = ArithmeticDecoder(payload)
            context = b"\x00" * self._cfg.context_window
            result = bytearray()
            for _ in range(original_len):
                probs = self.predict_probs(context)
                b = dec.decode_symbol(probs)
                # Update models online (mirrors compress_chunk)
                ctx_byte = context[-1] if context else 0
                self._stat.update(ctx_byte, b)
                if isinstance(self._neural, LinearCMA):
                    self._neural.update_weights(b)
                    self._neural.advance(b)
                context = (context + bytes([b]))[-self._cfg.context_window:]
                result.append(b)
            return bytes(result)

        elif flag == 0x02:
            # zstd fallback
            if not ZSTD_AVAILABLE:
                raise RuntimeError("zstd not available but chunk uses zstd flag")
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(payload, max_length=original_len * 2)

        elif flag == 0x03:
            # Raw stored
            return payload

        else:
            raise ValueError(f"unknown chunk flag: {flag:#04x}")

    def compress(self, data: bytes) -> bytes:
        """
        Compress full data stream in chunks.

        Stream format:
          4 bytes  original_size  uint32 big-endian
          4 bytes  n_chunks       uint32 big-endian
          Then n_chunks × (4-byte-length + flag + payload).

        Model state is snapshotted before encoding so decompress()
        can restore the identical starting state.
        """
        self._stat.reset()
        if self._neural:
            self._neural.reset_state()

        # Snapshot state AFTER reset, BEFORE any encoding.
        # decompress() restores this so both sides start identically.
        self._compress_state = self.save_state()

        chunks = [
            data[i:i + self._cfg.chunk_size]
            for i in range(0, len(data), self._cfg.chunk_size)
        ]
        header = struct.pack(">II", len(data), len(chunks))
        parts = [header]
        for chunk in chunks:
            parts.append(self.compress_chunk(chunk))
        return b"".join(parts)

    def decompress(self, data: bytes) -> bytes:
        """
        Decompress a stream produced by compress().
        Restores the exact model state used at the start of compression.
        """
        if len(data) < 8:
            raise ValueError("stream too short for header")

        original_size, n_chunks = struct.unpack(">II", data[:8])
        pos = 8

        # Restore model to the state captured at start of compress()
        if hasattr(self, "_compress_state"):
            self.restore_state(self._compress_state)
        else:
            self._stat.reset()
            if self._neural:
                self._neural.reset_state()

        chunk_size = self._cfg.chunk_size
        chunk_orig_lens = []
        remaining = original_size
        for _ in range(n_chunks):
            cl = min(chunk_size, remaining)
            chunk_orig_lens.append(cl)
            remaining -= cl

        parts = []
        for i in range(n_chunks):
            if pos + 4 > len(data):
                raise ValueError(f"stream truncated at chunk {i}")
            payload_len = struct.unpack(">I", data[pos:pos + 4])[0]
            pos += 4
            if pos + payload_len > len(data):
                raise ValueError(f"chunk {i} payload truncated")
            chunk_payload = data[pos:pos + payload_len]
            pos += payload_len
            parts.append(self.decompress_chunk(chunk_payload, chunk_orig_lens[i]))

        result = b"".join(parts)
        if len(result) != original_size:
            raise ValueError(
                f"size mismatch: got {len(result)}, expected {original_size}"
            )
        return result

    # ------------------------------------------------------------------
    # State snapshot — ensures encoder and decoder start from
    # identical model state even after repeated compress calls.
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """
        Snapshot mutable model state before compression.
        Pass the result to restore_state() before the matching decompress().
        """
        state: dict = {"stat_counts": self._stat._counts.copy()}
        if isinstance(self._neural, LinearCMA):
            state["cma_weights"] = self._neural._weights.copy()
            state["cma_losses"]  = self._neural._losses.copy()
            state["cma_context"] = self._neural._context
            state["cma_model_counts"] = [
                {k: v.copy() for k, v in m["counts"].items()}
                for m in self._neural._models
            ]
        return state

    def restore_state(self, state: dict) -> None:
        """Restore a snapshot produced by save_state()."""
        self._stat._counts = state["stat_counts"].copy()
        if isinstance(self._neural, LinearCMA) and "cma_weights" in state:
            self._neural._weights  = state["cma_weights"].copy()
            self._neural._losses   = state["cma_losses"].copy()
            self._neural._context  = state["cma_context"]
            for m, saved in zip(self._neural._models,
                                state["cma_model_counts"]):
                m["counts"] = {k: v.copy() for k, v in saved.items()}

    def reset(self) -> None:
        self._stat.reset()
        if self._neural:
            self._neural.reset_state()


# ---------------------------------------------------------------------------
# Torch wrapper helper
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class _TorchWrapper(NeuralBackend):
        def __init__(self, model, context_window: int):
            self._model = model
            self._ctx_window = context_window
            self._context: list[int] = []

        def predict(self, context: bytes) -> np.ndarray:
            ctx = list(context[-self._ctx_window:]) or [0]
            if hasattr(self._model, "predict_next"):
                return self._model.predict_next(ctx)
            return np.ones(256) / 256.0

        def reset_state(self) -> None:
            self._context = []
            if hasattr(self._model, "reset_state"):
                self._model.reset_state()
else:
    class _TorchWrapper(NeuralBackend):  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("PyTorch not available")
        def predict(self, context: bytes) -> np.ndarray: ...
        def reset_state(self) -> None: ...


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    predictor = HybridPredictor(HybridConfig(
        mode=PredictorMode.LINEAR_CMA,
        alpha=0.35,
        chunk_size=2048,
    ))

    # Structured data: repeating uint32 counter
    import numpy as np
    rng = np.random.default_rng(0)
    data = np.cumsum(rng.integers(0, 50, size=4096)).astype("<u4").tobytes()

    predictor.train(data[:2048])
    compressed = predictor.compress(data)
    ratio = len(data) / len(compressed)
    print(f"Linear CMA: {len(data)} → {len(compressed)} bytes  "
          f"({ratio:.2f}x ratio)")

    # Text data
    text = b"Hello world! " * 1000
    predictor2 = HybridPredictor(HybridConfig(
        mode=PredictorMode.LINEAR_CMA,
        alpha=0.45,
        chunk_size=2048,
    ))
    predictor2.train(text[:2000])
    comp2 = predictor2.compress(text)
    ratio2 = len(text) / len(comp2)
    print(f"Text CMA:   {len(text)} → {len(comp2)} bytes  "
          f"({ratio2:.2f}x ratio)")
