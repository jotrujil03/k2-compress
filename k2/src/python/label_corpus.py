"""
k2/src/python/label_corpus.py

Semi-supervised labelling pipeline — K2
-----------------------------------------
Builds a labelled training set for StructureClassifier / TransformGainPredictor
from a REAL directory (a game install, asset dump, etc.) — not synthetic data.
Synthetic data was explicitly rejected as a training source for this project:
a classifier trained on synthetic INTEGER_ARRAY/FLOAT_ARRAY/COLUMNAR samples
generated from clean rules would learn to recognize clean synthetic data, not
the messier byte layouts real game assets actually have.

There is no ground truth available automatically. File extension alone
cannot distinguish INTEGER_ARRAY vs TIMESERIES vs COLUMNAR vs FLOAT_ARRAY —
those are about byte layout, not file format — and using the existing
heuristic classifier to label its own training data would just bake in
whatever biases it already has, not improve on them.

Strategy: semi-supervised.
  1. Walk the directory, chunk every file into probe_bytes-sized samples
     (matching StructureDiscovery's own analysis window, so training-time
     features match inference-time features).
  2. Run the existing heuristic classifier (StructureDiscovery) on every
     chunk. High-confidence results (>= --auto-accept-threshold) are
     auto-labelled with the heuristic's own call.
  3. Everything else goes into a review queue for a human to label via
     review_labels.py (a separate small CLI built alongside this file).
  4. train_predictor.py consumes the merged auto-accepted + human-reviewed
     set.

This deliberately produces NO trained model by itself — only the labelled
dataset (as a directory of .npz shards, so it doesn't need to hold the
whole corpus in memory) and a review queue. Point this at a real game
directory to actually use it; there is no usable corpus in this sandbox.

Usage:
    python label_corpus.py \\
        --data-dir /path/to/game/install \\
        --output-dir labels/ \\
        --auto-accept-threshold 0.75 \\
        --probe-bytes 524288 \\
        --max-chunks-per-file 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from structure_discovery import DataClass, StructureDiscovery, StructureHint


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class ChunkRef:
    """
    Points at one sample without holding its bytes in memory until needed.
    file_path / offset / length let label_corpus.py and review_labels.py
    re-read the exact same bytes on demand, so the dataset on disk is just
    these small references plus the (initially heuristic, possibly
    human-corrected) label -- not a copy of the corpus itself.
    """
    file_path: str
    offset: int
    length: int


def iter_chunks(
    data_dir: str,
    probe_bytes: int,
    max_chunks_per_file: int,
    min_file_bytes: int = 4096,
) -> Iterator[ChunkRef]:
    """
    Walk data_dir, yield up to max_chunks_per_file ChunkRefs per file.

    For files <= probe_bytes, yields one chunk covering the whole file.
    For larger files, samples max_chunks_per_file chunks spread across the
    file (start, end, and evenly-spaced points between) rather than always
    just the first probe_bytes -- a file's header/start is often
    atypically structured (magic bytes, fixed-format header fields)
    compared to its bulk content, so always sampling offset 0 would bias
    the training set toward header-like layouts.
    """
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                continue
            if fsize < min_file_bytes:
                continue

            if fsize <= probe_bytes:
                yield ChunkRef(fpath, 0, fsize)
                continue

            n = max(1, max_chunks_per_file)
            if n == 1:
                offsets = [0]
            else:
                last_valid = fsize - probe_bytes
                offsets = [
                    int(i * last_valid / (n - 1)) for i in range(n)
                ]
            for off in offsets:
                yield ChunkRef(fpath, off, probe_bytes)


def read_chunk(ref: ChunkRef) -> bytes:
    with open(ref.file_path, "rb") as f:
        f.seek(ref.offset)
        return f.read(ref.length)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

@dataclass
class LabelledChunk:
    file_path: str
    offset: int
    length: int
    heuristic_class: str
    heuristic_confidence: float
    # Filled in later by review_labels.py; None until a human looks at it.
    human_class: Optional[str] = None


def label_directory(
    data_dir: str,
    probe_bytes: int,
    max_chunks_per_file: int,
    auto_accept_threshold: float,
    sample_fraction: float = 1.0,
    seed: int = 0,
    force_review_extensions: tuple[str, ...] = (),
) -> tuple[list[LabelledChunk], list[LabelledChunk]]:
    """
    Returns (auto_accepted, needs_review).

    sample_fraction < 1.0 randomly subsamples chunks before classification
    (useful for a first pass over a very large corpus to gauge class
    balance / review-queue size before committing to the full run).

    force_review_extensions: file extensions (e.g. (".bsa",)) that always
    go to needs_review regardless of heuristic confidence. Exists because
    confidence itself can be misleadingly high for archive-container
    formats: a small probe window landing inside packed mesh/texture data
    can show genuine local periodicity (high confidence, "COLUMNAR") while
    being meaningless as a description of the file as a whole, which is
    opaque packed bytes at the file level -- confirmed directly against a
    real Skyrim .bsa corpus, where this caused systematic COLUMNAR
    over-labelling (62/65 auto-accepted chunks in one run) that didn't
    reflect the actual class distribution at all. Forcing these to manual
    review lets a human apply judgment the heuristic's confidence score
    can't be trusted for here, rather than silently absorbing a
    confidently-wrong label into training data.
    """
    rng = random.Random(seed)
    disc = StructureDiscovery(probe_bytes=probe_bytes)
    force_review_extensions = tuple(e.lower() for e in force_review_extensions)

    auto_accepted: list[LabelledChunk] = []
    needs_review: list[LabelledChunk] = []

    n_seen = 0
    for ref in iter_chunks(data_dir, probe_bytes, max_chunks_per_file):
        if sample_fraction < 1.0 and rng.random() > sample_fraction:
            continue
        n_seen += 1
        try:
            data = read_chunk(ref)
        except OSError:
            continue
        if not data:
            continue

        hint = disc.analyze(data)
        row = LabelledChunk(
            file_path=ref.file_path,
            offset=ref.offset,
            length=ref.length,
            heuristic_class=hint.data_class.name,
            heuristic_confidence=hint.confidence,
        )
        forced = force_review_extensions and ref.file_path.lower().endswith(force_review_extensions)
        if hint.confidence >= auto_accept_threshold and not forced:
            auto_accepted.append(row)
        else:
            needs_review.append(row)

    return auto_accepted, needs_review


# ---------------------------------------------------------------------------
# Persistence — plain JSONL, one record per line.
# ---------------------------------------------------------------------------
# Deliberately NOT storing chunk bytes here -- file_path/offset/length is
# enough to re-read them later (corpus is assumed to stay in place between
# labelling and training), and keeps these label files small (KB, not GB)
# regardless of corpus size.

def write_jsonl(rows: list[LabelledChunk], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def read_jsonl(path: str) -> list[LabelledChunk]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(LabelledChunk(**json.loads(line)))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a semi-supervised labelled training set from a "
                     "real directory (game install, asset dump, etc.)."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Real directory to label. Must be representative "
                             "data, not synthetic samples.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-bytes", type=int, default=512 * 1024,
                        help="Must match StructureDiscovery's probe_bytes "
                             "used at inference time, or the trained model "
                             "will see a different feature distribution "
                             "than it sees in production.")
    parser.add_argument("--max-chunks-per-file", type=int, default=4)
    parser.add_argument("--auto-accept-threshold", type=float, default=0.75,
                        help="Heuristic confidence >= this is auto-labelled; "
                             "below this goes to the review queue.")
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="Randomly subsample this fraction of chunks. "
                             "Use < 1.0 for a quick first pass over a very "
                             "large corpus.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-review-extensions", nargs="*", default=[],
                        help="File extensions (e.g. .bsa) that always go to "
                             "the review queue regardless of heuristic "
                             "confidence. Use for archive/container formats "
                             "where a small probe window can show "
                             "misleadingly high-confidence local structure "
                             "(e.g. periodicity from packed mesh data) that "
                             "doesn't reflect the file's actual content as "
                             "a whole.")
    args = parser.parse_args()

    auto_accepted, needs_review = label_directory(
        args.data_dir, args.probe_bytes, args.max_chunks_per_file,
        args.auto_accept_threshold, args.sample_fraction, args.seed,
        force_review_extensions=tuple(args.force_review_extensions),
    )

    auto_path = os.path.join(args.output_dir, "auto_accepted.jsonl")
    review_path = os.path.join(args.output_dir, "needs_review.jsonl")
    write_jsonl(auto_accepted, auto_path)
    write_jsonl(needs_review, review_path)

    total = len(auto_accepted) + len(needs_review)
    print(f"Labelled {total} chunks from {args.data_dir}")
    print(f"  auto-accepted: {len(auto_accepted)} ({auto_path})")
    print(f"  needs review:  {len(needs_review)} ({review_path})")
    if needs_review:
        print(f"\nRun review_labels.py --input {review_path} to label "
              f"the review queue by hand before training.")

    if total > 0:
        from collections import Counter
        dist = Counter(r.heuristic_class for r in auto_accepted)
        print("\nAuto-accepted class distribution:")
        for cls, n in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"  {cls:15s} {n:6d}  ({100*n/max(len(auto_accepted),1):.1f}%)")


if __name__ == "__main__":
    main()
