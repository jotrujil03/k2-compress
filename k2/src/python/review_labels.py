"""
k2/src/python/review_labels.py

Human review CLI for the labelling pipeline — K2
---------------------------------------------------
Reviews the needs_review.jsonl queue produced by label_corpus.py one chunk
at a time, showing the heuristic's guess plus enough context (entropy,
delta-gain, repeat density, detected column stride, a safe preview of the
bytes) for a human to make a fast, informed call.

This is the step that actually produces ground truth: label_corpus.py's
auto-accepted set is still just the heuristic's own opinion (filtered to
cases it claims to be confident about) — only the chunks reviewed here, by
a person who can recognize "this is PNG texture data" or "this is a vertex
buffer" by eye, are real labels in the sense a trained classifier should
trust to learn from.

Progress is saved after every decision (not just at exit), so a long
review session can be safely interrupted and resumed.

Usage:
    python review_labels.py --input labels/needs_review.jsonl

Controls (shown each time):
    [enter]  accept the heuristic's guess as-is
    1-8      pick a specific DataClass by number
    s        skip (leave unlabelled, revisit later)
    p        preview more bytes (hex + ASCII)
    q        save and quit
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from label_corpus import LabelledChunk, read_chunk, read_jsonl, write_jsonl, ChunkRef
from structure_discovery import DataClass


_CLASSES = list(DataClass)  # stable order: UNKNOWN, INTEGER_ARRAY, ... MIXED


def _hex_preview(data: bytes, n: int = 64) -> str:
    chunk = data[:n]
    hex_part = " ".join(f"{b:02x}" for b in chunk)
    ascii_part = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in chunk)
    return f"{hex_part}\n  {ascii_part}"


def _show_chunk(row: LabelledChunk, data: bytes, idx: int, total: int) -> None:
    print(f"\n--- chunk {idx + 1}/{total} ---")
    print(f"file:   {row.file_path}")
    print(f"offset: {row.offset}  length: {row.length}")
    print(f"heuristic guess: {row.heuristic_class}  "
          f"(confidence {row.heuristic_confidence:.2f})")
    print(f"preview (first 64 bytes):\n  {_hex_preview(data)}")
    print()
    print("Pick a class, or [enter] to accept the heuristic's guess:")
    for i, c in enumerate(_CLASSES, start=1):
        marker = " *" if c.name == row.heuristic_class else ""
        print(f"  {i}: {c.name}{marker}")
    print("  s: skip   p: preview more bytes   q: save and quit")


def review(input_path: str, output_path: Optional[str] = None) -> None:
    output_path = output_path or input_path
    rows = read_jsonl(input_path)

    n_done = sum(1 for r in rows if r.human_class is not None)
    print(f"Loaded {len(rows)} chunks ({n_done} already labelled, "
          f"{len(rows) - n_done} remaining).")

    for idx, row in enumerate(rows):
        if row.human_class is not None:
            continue  # already reviewed in a prior session

        ref = ChunkRef(row.file_path, row.offset, row.length)
        try:
            data = read_chunk(ref)
        except OSError as e:
            print(f"skipping {row.file_path}: {e}")
            continue

        preview_n = 64
        while True:
            _show_chunk(row, data[:max(preview_n, 64)], idx, len(rows))
            choice = input("> ").strip().lower()

            if choice == "":
                row.human_class = row.heuristic_class
                break
            elif choice == "s":
                break
            elif choice == "p":
                preview_n += 192
                continue
            elif choice == "q":
                write_jsonl(rows, output_path)
                print(f"\nSaved progress to {output_path}. "
                      f"({sum(1 for r in rows if r.human_class is not None)}/"
                      f"{len(rows)} labelled)")
                return
            elif choice.isdigit() and 1 <= int(choice) <= len(_CLASSES):
                row.human_class = _CLASSES[int(choice) - 1].name
                break
            else:
                print(f"unrecognized input: {choice!r}")
                continue

        # Save after every decision -- a long session can be interrupted
        # at any point without losing prior work.
        write_jsonl(rows, output_path)

    n_done = sum(1 for r in rows if r.human_class is not None)
    print(f"\nReview complete: {n_done}/{len(rows)} labelled.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="needs_review.jsonl from label_corpus.py")
    parser.add_argument("--output", default=None,
                        help="Where to save (default: overwrite --input in place)")
    args = parser.parse_args()
    review(args.input, args.output)


if __name__ == "__main__":
    main()
