"""
k2/src/python/hybrid_predictor.py

ML Models — K2
----------------
This file used to host a pure-Python neural-mixing + arithmetic-coding
entropy path (LSTM/MiniTransformer/LinearCMA blended with a Zstd-derived
statistical model). That machinery is removed: it was never wired into the
live C++ bridge path (only a separate, disconnected test entry point used
it), it depended on zstd as a fallback backend, and the underlying idea —
blending several statistical/learned predictors via a mixer — is already
what ASDP's CM codec does in C++ (six order-k context models blended by a
trained logistic mixer, plus a match model), directly against the real
bitstream. Re-deriving a slower, disconnected copy of that idea in Python
added no value.

What lives here now: the two real ML levers Python actually controls,
neither of which touches entropy coding —

  1. StructureClassifier — small feed-forward net, 64-dim engineered
     features -> DataClass logits. Replaces hand-tuned heuristic
     thresholds in StructureDiscovery._heuristic_classify with a trained
     model when one is available (ONNXStructureClassifier in
     structure_discovery.py already has the inference-side wrapper; this
     file is the model definition + training-time companion).

  2. TransformGainPredictor — small feed-forward net, predicts expected
     compression-gain ratio from applying columnsplit (Python's one real
     upstream transform) vs leaving data untouched. Replaces the fixed
     zlib-1-ratio threshold guard (_probe_transform_gain /
     _MIN_TRANSFORM_GAIN in adaptive_optimizer.py) with a learned
     estimate once trained.

Both are intentionally tiny (a few hundred KB at most) since they run
synchronously on every prepare() call, on the CPU, in a process that also
needs to leave headroom for the real compression work.

Training happens offline in train_predictor.py; this file only defines
the model architectures and the runtime-inference wrappers (PyTorch
state-dict or ONNX Runtime), matching the pattern structure_discovery.py
already established for ONNXStructureClassifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

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


# ---------------------------------------------------------------------------
# Model 1: Structure classifier (DataClass)
# ---------------------------------------------------------------------------
# Architecture mirrors train_predictor.py's existing StructureClassifier
# exactly (same shape is required for state-dict / ONNX compatibility) —
# defined here as well so this module is the single source of truth for
# both training and runtime use, rather than train_predictor.py owning a
# model definition that only it imports.

if TORCH_AVAILABLE:
    class StructureClassifier(nn.Module):
        """64-dim engineered feature vector -> DataClass logits."""

        def __init__(self, in_dim: int = 64, n_classes: int = 8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


# ---------------------------------------------------------------------------
# Model 2: Transform-gain predictor
# ---------------------------------------------------------------------------
# Predicts log2(ratio_with_columnsplit / ratio_without) from the same
# engineered feature vector used by the structure classifier (no need for
# a second, separate feature extractor — the signal that predicts "is this
# columnar" and "would columnsplit help" overlaps heavily: column_stride,
# repeat_density, byte histogram shape).
#
# Trained as a regression target (continuous predicted gain), not a
# classifier, since the actual guard decision (_MIN_TRANSFORM_GAIN) is a
# threshold applied to a continuous quantity — replacing the existing
# zlib-1-ratio heuristic with a *learned estimate of the same quantity*,
# evaluated without needing to actually run zlib or columnsplit at
# decision time. Falls back cleanly to the zlib-1 heuristic guard if no
# trained model is available (see adaptive_optimizer.py's
# _probe_transform_gain), so a missing or stale model file degrades to
# today's known-correct behavior rather than breaking compression.

if TORCH_AVAILABLE:
    class TransformGainPredictor(nn.Module):
        """64-dim engineered feature vector -> scalar predicted gain (log2 ratio)."""

        def __init__(self, in_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Runtime inference wrappers (ONNX Runtime, CPU-only)
# ---------------------------------------------------------------------------
# Both wrappers degrade gracefully: if onnxruntime isn't installed, or the
# model file is missing/fails to load, predict() returns None and callers
# fall back to their existing heuristic path. No exception ever propagates
# out of these wrappers for a missing/bad model — only for a genuinely
# malformed call (wrong feature shape), which indicates a real bug.

class ONNXGainPredictor:
    """
    Wraps a tiny ONNX model (64-dim features -> scalar gain estimate).
    Mirrors structure_discovery.ONNXStructureClassifier's pattern.
    """

    def __init__(self, model_path: Optional[str]):
        self._session: Optional["ort.InferenceSession"] = None
        if model_path and ONNX_AVAILABLE:
            try:
                self._session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"],
                )
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._session is not None

    def predict(self, features: np.ndarray) -> Optional[float]:
        """Returns predicted log2(gain ratio), or None if unavailable."""
        if self._session is None:
            return None
        inp_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {inp_name: features.astype(np.float32)})[0]
        return float(out.reshape(-1)[0])


@dataclass
class MLConfig:
    """
    Paths to trained model files, if any. All optional — every consumer of
    these models must work correctly (via heuristic fallback) when these
    are None or point to missing files. No model file ships by default;
    see train_predictor.py for how to produce one from a real, labelled
    corpus (synthetic data is not a substitute — see that file's
    docstring for why).
    """
    structure_classifier_path: Optional[str] = None
    gain_predictor_path: Optional[str] = None
