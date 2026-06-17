"""
k2/src/python/train_predictor.py

Offline Training Script — K2
------------------------------
Trains the LSTMPredictor or MiniTransformer on representative data files.
Exports to:
  1. PyTorch state-dict (.pt) — used at runtime via _TorchWrapper
  2. ONNX model (.onnx) — used at runtime via ONNXPredictor (faster inference)

Usage:
    python train_predictor.py \\
        --data-dir /path/to/representative/data \\
        --model lstm \\
        --epochs 5 \\
        --output models/predictor.pt \\
        --export-onnx models/predictor.onnx

    python train_predictor.py \\
        --data-dir /path/to/data \\
        --model transformer \\
        --embed-dim 64 \\
        --hidden 256 \\
        --epochs 10 \\
        --output models/transformer.pt

Training strategy:
  - Sliding window of `context_len` bytes → predict next byte
  - CrossEntropyLoss on 256-class logits
  - AdamW optimizer, cosine LR schedule
  - Validation split: last 10% of data
  - Early stopping if validation loss doesn't improve for `patience` epochs
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise SystemExit("PyTorch is required for training.  "
                     "Install with: pip install torch")

from hybrid_predictor import LSTMPredictor, MiniTransformer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ByteStreamDataset(Dataset):
    """
    Yields (context, target) pairs from a flat byte stream.
    context: int64 tensor of length `context_len`
    target:  int64 scalar (next byte)
    """

    def __init__(self, data: bytes, context_len: int = 64, stride: int = 1):
        self._data = np.frombuffer(data, dtype=np.uint8)
        self._ctx = context_len
        self._stride = stride
        # Pre-compute valid start indices
        n = len(self._data)
        self._starts = list(range(0, n - context_len - 1, stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        i = self._starts[idx]
        x = torch.tensor(self._data[i: i + self._ctx], dtype=torch.long)
        y = torch.tensor(int(self._data[i + self._ctx]), dtype=torch.long)
        return x, y


def load_data(data_dir: str, max_bytes: int = 50 * 1024 * 1024) -> bytes:
    """Load all files in data_dir up to max_bytes total."""
    files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
    )
    files = [f for f in files if os.path.isfile(f)]
    random.shuffle(files)
    buf = bytearray()
    for f in files:
        if len(buf) >= max_bytes:
            break
        try:
            with open(f, "rb") as fh:
                buf.extend(fh.read(max_bytes - len(buf)))
        except OSError:
            pass
    print(f"Loaded {len(buf) / 1e6:.1f} MB from {data_dir}")
    return bytes(buf)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    data: bytes,
    context_len: int = 64,
    batch_size: int = 512,
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 2,
    device: str = "cpu",
    stride: int = 4,
) -> dict:
    """
    Train model on data.  Returns history dict with train/val losses.
    """
    dev = torch.device(device)
    model = model.to(dev)

    dataset = ByteStreamDataset(data, context_len, stride)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_bpc": []}
    best_val = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()

            if isinstance(model, MiniTransformer):
                logits = model(x)[:, -1, :]
            else:  # LSTM
                logits, _ = model(x)
                logits = logits[:, -1, :]

            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                if isinstance(model, MiniTransformer):
                    logits = model(x)[:, -1, :]
                else:
                    logits, _ = model(x)
                    logits = logits[:, -1, :]
                val_loss += criterion(logits, y).item()
                n_val += 1
        val_loss /= max(n_val, 1)
        bpc = val_loss / np.log(2)  # bits per character

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_bpc"].append(round(bpc, 4))

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"bpc={bpc:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.1f}s)")

        scheduler.step()

        # Early stopping
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, output_path: str, context_len: int = 64) -> None:
    """Export trained model to ONNX (opset 13)."""
    model.eval()
    dummy = torch.zeros(1, context_len, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=["context"],
        output_names=["logits"],
        dynamic_axes={"context": {0: "batch", 1: "seq_len"}},
        opset_version=13,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported ONNX model to {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Also train the tiny ONNX type-classifier
# ---------------------------------------------------------------------------

class StructureClassifier(nn.Module):
    """
    64-dim feature → 8 DataClass logits.
    Used by StructureDiscovery's ONNXStructureClassifier.
    """
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


def train_structure_classifier(
    labelled_samples: list[tuple[bytes, int]],  # (chunk, class_id)
    epochs: int = 20,
    lr: float = 1e-3,
) -> StructureClassifier:
    """
    Train the tiny structure classifier on labelled chunks.
    labelled_samples: list of (raw_bytes, DataClass integer index)
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from structure_discovery import _extract_features

    clf = StructureClassifier()
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X = np.vstack([_extract_features(s, 4) for s, _ in labelled_samples])
    y = np.array([label for _, label in labelled_samples], dtype=np.int64)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    for ep in range(1, epochs + 1):
        clf.train()
        optimizer.zero_grad()
        logits = clf(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        if ep % 5 == 0:
            acc = (logits.argmax(1) == y_t).float().mean().item()
            print(f"Classifier epoch {ep:02d}: loss={loss.item():.4f} acc={acc:.3f}")

    return clf


def export_classifier_onnx(clf: StructureClassifier, path: str) -> None:
    clf.eval()
    dummy = torch.zeros(1, 64)
    torch.onnx.export(
        clf, (dummy,), path,
        input_names=["features"], output_names=["logits"],
        opset_version=13,
    )
    print(f"Classifier exported to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train neural predictor for OpenZL hybrid compression"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory of representative training files")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--embed-dim",   type=int, default=32)
    parser.add_argument("--hidden",      type=int, default=128)
    parser.add_argument("--num-layers",  type=int, default=2)
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--patience",    type=int, default=2)
    parser.add_argument("--device",      default="cpu",
                        help="cpu | cuda | mps")
    parser.add_argument("--output",      default="models/predictor.pt")
    parser.add_argument("--export-onnx", default="",
                        help="If set, also export to this ONNX path")
    parser.add_argument("--max-bytes",   type=int, default=50 * 1024 * 1024)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data_dir, args.max_bytes)

    if args.model == "lstm":
        model = LSTMPredictor(
            embed_dim=args.embed_dim,
            hidden_size=args.hidden,
            num_layers=args.num_layers,
        )
    else:
        model = MiniTransformer(
            d_model=args.embed_dim,
            nhead=max(1, args.embed_dim // 16),
            num_layers=args.num_layers,
            max_len=args.context_len,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params={n_params:,}  "
          f"context={args.context_len}  device={args.device}")

    history = train(
        model, data,
        context_len=args.context_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )

    torch.save(model.state_dict(), args.output)
    print(f"Saved state-dict → {args.output}")

    if args.export_onnx:
        Path(args.export_onnx).parent.mkdir(parents=True, exist_ok=True)
        export_onnx(model, args.export_onnx, args.context_len)

    print("\nTraining history:")
    for ep, (tl, vl, bpc) in enumerate(zip(
        history["train_loss"], history["val_loss"], history["val_bpc"]
    ), 1):
        print(f"  epoch {ep:02d}  train={tl:.4f}  val={vl:.4f}  bpc={bpc:.3f}")


if __name__ == "__main__":
    main()
