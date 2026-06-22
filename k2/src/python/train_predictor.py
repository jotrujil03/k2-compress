"""
k2/src/python/train_predictor.py

Offline Training Script — K2
------------------------------
Trains the two ML models Python actually uses (see hybrid_predictor.py's
module docstring for why the old LSTM/MiniTransformer next-byte predictors
were removed — they were never wired into the live C++ bridge path):

  1. StructureClassifier — DataClass classification from engineered
     features. Trained on the OUTPUT of label_corpus.py + review_labels.py:
     a semi-supervised labelled set built from a real directory, not
     synthetic data (synthetic samples were explicitly rejected as a
     training source for this project — see label_corpus.py's docstring).

  2. TransformGainPredictor — predicts expected gain (log2 ratio) from
     applying columnsplit vs leaving data untouched. Trained on REAL
     measured ASDP/CM compression sizes via measure_gain_tool (a small
     standalone C++ binary, see k2/src/cpp/measure_gain_tool.cpp) — not a
     zlib or entropy proxy, since the whole point is to predict what the
     actual entropy backend will do.

Both export to ONNX for runtime inference (see structure_discovery.py's
ONNXStructureClassifier and hybrid_predictor.py's ONNXGainPredictor).

Usage:
    # Step 1: label a REAL directory (game install, asset dump, etc.)
    python label_corpus.py --data-dir /path/to/game --output-dir labels/

    # Step 2: review the low-confidence queue by hand
    python review_labels.py --input labels/needs_review.jsonl

    # Step 3: train the structure classifier
    python train_predictor.py classifier \\
        --labels-dir labels/ \\
        --output models/structure_classifier.onnx

    # Step 4: train the gain predictor (needs measure_gain_tool built —
    # see k2/src/cpp/measure_gain_tool.cpp; requires libasdp.a)
    python train_predictor.py gain-predictor \\
        --data-dir /path/to/game \\
        --measure-tool /path/to/measure_gain_tool \\
        --output models/gain_predictor.onnx

Neither subcommand ships a model by default — there is no labelled corpus
or measurement data available until pointed at real, representative game
data. Running either command without real input data (or skipping the
label_corpus.py / review_labels.py steps) will train on too little or too
biased data to be useful; the commands intentionally do not fall back to
synthetic data.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise SystemExit("PyTorch is required for training.  "
                     "Install with: pip install torch")

from hybrid_predictor import StructureClassifier, TransformGainPredictor
from structure_discovery import DataClass, _extract_features
from label_corpus import (
    LabelledChunk, ChunkRef, read_chunk, read_jsonl, iter_chunks,
)


_CLASS_ORDER = list(DataClass)
_CLASS_TO_IDX = {c.name: i for i, c in enumerate(_CLASS_ORDER)}


# ---------------------------------------------------------------------------
# Shared training loop (small dataset, small model — no need for the
# epoch/scheduler/early-stopping machinery the old byte-predictor training
# needed; these models train in seconds to low minutes on CPU)
# ---------------------------------------------------------------------------

def _train_loop(
    model: "nn.Module",
    X: np.ndarray,
    y: np.ndarray,
    criterion,
    epochs: int,
    lr: float,
    batch_size: int,
    val_fraction: float = 0.15,
) -> dict:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y)

    dataset = TensorDataset(X_t, y_t)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError(
            f"only {len(dataset)} samples available — need more than "
            f"{n_val} to hold out a validation split. Label more data."
        )
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        train_loss = total / max(n_batches, 1)

        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                vtotal += criterion(out, yb).item()
                vn += 1
        val_loss = vtotal / max(vn, 1)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        print(f"  epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}")

    return history


def _export_onnx(model: "nn.Module", in_dim: int, output_path: str,
                  input_name: str, output_name: str) -> None:
    """
    Exports model to ONNX with a dynamic batch dimension.

    torch 2.9+ defaults torch.onnx.export to the newer torch.export-based
    ("dynamo=True") exporter, which wants dynamic_shapes (built from
    torch.export.Dim) instead of the older dynamic_axes dict -- passing
    dynamic_axes there now prints a UserWarning and is silently converted
    on a best-effort basis rather than used directly.

    Also stopped requesting opset_version=13: on the dynamo path, the
    exporter builds the graph at a newer internal opset and then tries to
    DOWNGRADE to whatever opset_version was requested via onnxscript's
    version converter. That downgrade step itself was failing on a real
    run (RuntimeError: No Adapter From Version 16 for Identity) and
    silently falling back to keeping the newer opset anyway -- so the
    requested 13 was never actually being honored, just generating a
    scary-looking but harmless traceback. onnxruntime handles the
    resulting newer opset fine (confirmed against onnxruntime 1.27); no
    reason to keep requesting a downgrade that doesn't work and isn't
    needed.

    Tries the new dynamic_shapes API first; falls back to the old
    dynamic_axes/opset_version=13 path on a torch version old enough that
    torch.onnx.export doesn't accept dynamic_shapes at all (pre-2.9-ish),
    so this keeps working across a range of torch versions rather than
    only the one this was last tested against.

    Also passes dynamic_axes alongside dynamic_shapes on the primary
    attempt: torch's own 2.9 docs recommend this explicitly, since the
    dynamo exporter can internally fall back to the older TorchScript path
    (exactly what happened during the real run that surfaced this whole
    issue, triggered by the opset-downgrade failure) -- without
    dynamic_axes also present, that internal fallback would silently
    export a static-batch-size model instead of erroring, which would be
    a much harder bug to notice than a loud failure.

    IMPORTANT: dynamic_shapes keys by the model's actual forward()
    PYTHON ARGUMENT NAME (traced via torch.export), which is completely
    unrelated to input_name/output_name here (those are only the
    ONNX-graph-facing labels passed to input_names/output_names -- pure
    metadata for the exported graph, never seen by torch.export at all).
    Both StructureClassifier.forward(self, x) and
    TransformGainPredictor.forward(self, x) take a single positional
    argument named x, so the correct dict key is "x", not input_name's
    value ("features"/"gain") -- confirmed by a real run: keying with
    input_name raised "top-level keys must be the arg names ['x'] of
    inputs, but here they are ['features']". To avoid hardcoding "x" and
    silently breaking again if a model's forward() signature ever
    changes, dynamic_shapes is specified POSITIONALLY (a tuple matching
    the args tuple) instead of by name -- this is the documented
    alternative torch's own error message points to ("you could also
    ignore arg names entirely and specify dynamic_shapes as a list/tuple
    matching inputs").
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, in_dim, dtype=torch.float32)

    try:
        batch = torch.export.Dim("batch")
        torch.onnx.export(
            model, (dummy,), output_path,
            input_names=[input_name], output_names=[output_name],
            dynamic_shapes=({0: batch},),  # positional: matches the (dummy,) args tuple
            dynamic_axes={input_name: {0: "batch"}},
        )
    except TypeError:
        # Older torch: torch.onnx.export doesn't know dynamic_shapes at
        # all (raises TypeError on the unexpected kwarg) -- fall back to
        # the pre-2.9 dynamo=False-style API, where dynamic_axes and an
        # explicit opset_version are both the correct, non-deprecated way
        # to do this.
        torch.onnx.export(
            model, (dummy,), output_path,
            input_names=[input_name], output_names=[output_name],
            dynamic_axes={input_name: {0: "batch"}},
            opset_version=13,
        )

    size_kb = os.path.getsize(output_path) / 1e3
    print(f"Exported ONNX model to {output_path} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Subcommand 1: structure classifier
# ---------------------------------------------------------------------------

def _load_labelled_chunks(labels_dir: str) -> list[LabelledChunk]:
    """
    Merges auto_accepted.jsonl with the (possibly partially) reviewed
    needs_review.jsonl, keeping only rows that have a real label:
    auto-accepted rows always do (that's what auto-accept means); reviewed
    rows only if a human has actually set human_class (skipped rows stay
    excluded, not silently treated as their stale heuristic guess).
    """
    rows: list[LabelledChunk] = []

    auto_path = os.path.join(labels_dir, "auto_accepted.jsonl")
    if os.path.exists(auto_path):
        rows.extend(read_jsonl(auto_path))

    review_path = os.path.join(labels_dir, "needs_review.jsonl")
    if os.path.exists(review_path):
        _review_rows = read_jsonl(review_path)
        reviewed = [r for r in _review_rows if r.human_class is not None]
        skipped = sum(1 for r in _review_rows if r.human_class is None)
        if skipped:
            print(f"note: {skipped} reviewed-queue chunks have no human_class "
                  f"yet (skipped or not yet reached) — excluded from training.")
        rows.extend(reviewed)

    return rows


def _effective_label(row: LabelledChunk) -> str:
    return row.human_class if row.human_class is not None else row.heuristic_class


def train_classifier_cmd(args: argparse.Namespace) -> None:
    rows = _load_labelled_chunks(args.labels_dir)
    if not rows:
        raise SystemExit(
            f"no labelled chunks found under {args.labels_dir} — run "
            f"label_corpus.py (and review_labels.py for low-confidence "
            f"chunks) first."
        )

    from collections import Counter
    dist = Counter(_effective_label(r) for r in rows)
    print(f"Training on {len(rows)} labelled chunks:")
    for cls, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:15s} {n:6d}")
    rare = [c for c, n in dist.items() if n < 5]
    if rare:
        print(f"\nwarning: classes with <5 samples ({', '.join(rare)}) — "
              f"the model will likely not learn these reliably. Label "
              f"more data covering these classes if they matter.")

    print("\nExtracting features...")
    X, y = [], []
    for row in rows:
        ref = ChunkRef(row.file_path, row.offset, row.length)
        try:
            data = read_chunk(ref)
        except OSError:
            continue
        if not data:
            continue
        # element_size=4 matches the default used at inference time in
        # structure_discovery.py when calling _extract_features from
        # StructureDiscovery.analyze() before classification — keeping
        # this consistent matters more than picking the "best" value,
        # since the model must see the same feature distribution at
        # train and inference time.
        feats = _extract_features(data, element_size=4)
        X.append(feats.reshape(-1))
        y.append(_CLASS_TO_IDX[_effective_label(row)])

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    model = StructureClassifier(in_dim=X.shape[1], n_classes=len(_CLASS_ORDER))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: StructureClassifier  params={n_params:,}")

    history = _train_loop(
        model, X, y, nn.CrossEntropyLoss(),
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    if args.output_pt:
        Path(args.output_pt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output_pt)
        print(f"Saved state-dict -> {args.output_pt}")

    _export_onnx(model, X.shape[1], args.output, "features", "logits")


# ---------------------------------------------------------------------------
# Subcommand 2: transform-gain predictor
# ---------------------------------------------------------------------------

def _measure_gain(measure_tool: str, file_path: str, stride: int,
                   offset: Optional[int] = None, length: Optional[int] = None) -> Optional[float]:
    """
    Calls measure_gain_tool, returns log2(raw_compressed / split_compressed)
    -- positive means columnsplit helped, negative means it hurt. Returns
    None on any failure (tool missing, file unreadable, etc.) so the caller
    can skip that sample rather than crash a long training-data-generation
    run over one bad file.

    offset/length bound the measurement to a chunk of the file rather than
    the whole thing -- REQUIRED for real use. Without them, a large source
    file (a multi-hundred-MB/GB game archive) makes measure_gain_tool run
    real ASDP/CM compression on the entire file twice, which can take many
    minutes per sample. The caller passes ChunkRef.offset/length to keep
    each measurement bounded to the actual probe window.
    """
    cmd = [measure_tool, file_path, str(stride)]
    if offset is not None and length is not None:
        cmd += [str(offset), str(length)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if len(parts) != 3:
        return None
    try:
        _orig, raw_c, split_c = (int(p) for p in parts)
    except ValueError:
        return None
    if raw_c <= 0 or split_c <= 0:
        return None
    return float(np.log2(raw_c / split_c))


def train_gain_predictor_cmd(args: argparse.Namespace) -> None:
    if not os.path.exists(args.measure_tool):
        raise SystemExit(
            f"measure tool not found at {args.measure_tool} — build it "
            f"first: see k2/src/cpp/measure_gain_tool.cpp "
            f"(g++ ... measure_gain_tool.cpp libasdp.a -o measure_gain_tool)"
        )

    print(f"Scanning {args.data_dir} for candidate files "
          f"(this calls real ASDP/CM compression per sample via "
          f"{args.measure_tool} — can be slow on large corpora; use "
          f"--max-samples to bound it)...")

    refs = list(iter_chunks(
        args.data_dir, args.probe_bytes, max_chunks_per_file=1,
        min_file_bytes=args.stride * 16,  # need enough rows for a meaningful transpose
    ))
    rng = random.Random(args.seed)
    rng.shuffle(refs)
    refs = refs[: args.max_samples]

    if not refs:
        raise SystemExit(f"no candidate files found under {args.data_dir}")

    X, y = [], []
    n_measured, n_skipped = 0, 0
    for i, ref in enumerate(refs):
        gain = _measure_gain(args.measure_tool, ref.file_path, args.stride,
                              offset=ref.offset, length=ref.length)
        if gain is None:
            n_skipped += 1
            continue
        try:
            data = read_chunk(ref)
        except OSError:
            n_skipped += 1
            continue
        if not data:
            n_skipped += 1
            continue
        feats = _extract_features(data, element_size=4)
        X.append(feats.reshape(-1))
        y.append(gain)
        n_measured += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(refs)} processed "
                  f"({n_measured} measured, {n_skipped} skipped)")

    print(f"\nMeasured {n_measured} samples ({n_skipped} skipped: "
          f"too small, unreadable, or measure_gain_tool failed).")
    if n_measured < 20:
        raise SystemExit(
            "too few measured samples to train a meaningful regressor — "
            "point --data-dir at a larger, more varied real directory."
        )

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    print(f"gain distribution: min={y.min():.3f} max={y.max():.3f} "
          f"mean={y.mean():.3f} (positive = columnsplit helped)")

    model = TransformGainPredictor(in_dim=X.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TransformGainPredictor  params={n_params:,}")

    history = _train_loop(
        model, X, y, nn.MSELoss(),
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    if args.output_pt:
        Path(args.output_pt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output_pt)
        print(f"Saved state-dict -> {args.output_pt}")

    _export_onnx(model, X.shape[1], args.output, "features", "gain")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("classifier", help="Train StructureClassifier")
    p1.add_argument("--labels-dir", required=True,
                    help="Directory containing auto_accepted.jsonl and/or "
                         "needs_review.jsonl from label_corpus.py")
    p1.add_argument("--epochs", type=int, default=30)
    p1.add_argument("--lr", type=float, default=1e-3)
    p1.add_argument("--batch-size", type=int, default=32)
    p1.add_argument("--output", required=True, help="Output .onnx path")
    p1.add_argument("--output-pt", default=None, help="Optional .pt state-dict path")
    p1.set_defaults(func=train_classifier_cmd)

    p2 = sub.add_parser("gain-predictor", help="Train TransformGainPredictor")
    p2.add_argument("--data-dir", required=True,
                    help="Real directory to sample files from for measurement")
    p2.add_argument("--measure-tool", required=True,
                    help="Path to compiled measure_gain_tool binary")
    p2.add_argument("--stride", type=int, default=12,
                    help="columnsplit stride to measure gain for")
    p2.add_argument("--probe-bytes", type=int, default=65536,
                    help="Bytes per sample (kept smaller than "
                         "StructureDiscovery's default since this also "
                         "drives a real ASDP compress call per sample)")
    p2.add_argument("--max-samples", type=int, default=500)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--epochs", type=int, default=30)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--batch-size", type=int, default=32)
    p2.add_argument("--output", required=True, help="Output .onnx path")
    p2.add_argument("--output-pt", default=None, help="Optional .pt state-dict path")
    p2.set_defaults(func=train_gain_predictor_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
