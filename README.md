# K2 Compression Pipeline

**K2** is an adaptive compression pipeline built on top of **ASDP-LH**, a bit-domain context-mixing entropy codec (six order-k context models blended by a logistic mixer, plus a match model). ASDP-LH is the *only* entropy backend — there is no zstd or zlib fallback anywhere in this pipeline.

K2 adds three things on top of raw ASDP-LH:

- **Structure discovery** (Python) — classifies incoming data (text, integer/float arrays, columnar records, timeseries, binary blobs) and, for the one case ASDP can't handle on its own (columnar/struct-of-arrays layouts), suggests a `columnsplit` transform to apply before entropy coding.
- **A bandit-driven strategy selector** (Python) — picks between Python-side transform candidates using a UCB1 multi-armed bandit, seeded with a cheap proxy estimate and refined with real measured ratios as compression actually happens.
- **Multi-volume directory archiving** (`.k2a`, C++) — packs whole directories into size-capped volumes (4 GB default) on top of the same ASDP frame format used for single files.

## What Python does and doesn't do

This is worth being explicit about, since it's changed over the life of this project: **ASDP's own C++ trial-encode (`pick_transform_cm` in `asdp.cpp`) already searches `{none, bytedelta_1/2/4/8, bytesplit_4/8}` against the real CM-coded size, every block.** Python does not duplicate that search — it would be slower and less accurate than C++'s real measurement. Python's structure-discovery layer only ever suggests `columnsplit-N`, the one transform ASDP has no equivalent for, and only when the data actually looks columnar. For every other data class, Python hands bytes to ASDP unchanged and lets the C++ trial-encode make the real call.

Earlier versions of this codebase had a separate neural-mixing + arithmetic-coding entropy path in `hybrid_predictor.py` (LSTM/Transformer prediction blended with a Zstd-derived statistical model). It was never wired into the live C++ bridge and depended on zstd as a fallback — both are gone. What's there now: two small feed-forward ML models (a structure classifier and a transform-gain predictor), neither of which touches entropy coding — see [ML models](#ml-models-optional) below.

---

## Directory Structure

```text
k2/
├── include/
│   ├── k2_bridge.h              # C++ RAII wrapper and plain C API
│   └── k2archive.h              # K2A multi-volume archive format
├── src/
│   ├── cpp/
│   │   ├── k2_bridge.cpp        # Bridge implementation + embedded CPython
│   │   ├── k2cli.cpp            # Command-line interface
│   │   ├── k2archive.cpp        # Multi-volume archive handling
│   │   └── measure_gain_tool.cpp  # Standalone: real ASDP gain measurement
│   │                               # for training data (no Python needed)
│   └── python/
│       ├── structure_discovery.py   # Data-class detection + columnsplit hint
│       ├── hybrid_predictor.py      # ML model definitions (classifier, gain predictor)
│       ├── adaptive_optimizer.py    # UCB1 bandit + K2Pipeline (C++ bridge entry point)
│       ├── label_corpus.py          # Semi-supervised labelling for real training data
│       ├── review_labels.py         # CLI to hand-review low-confidence labels
│       └── train_predictor.py       # Trains + exports the two ONNX models
└── tests/
    ├── k2archive_test.cpp
    └── test_pipeline_integration.py
```

---

## Quick Start

### 1. Install Python Dependencies

```bash
pip install numpy

# Optional: for ONNX model inference at runtime (structure classifier / gain predictor)
pip install onnxruntime

# Optional: only needed to TRAIN the ONNX models (label_corpus.py / running the
# pipeline itself need neither onnxruntime nor torch — both degrade gracefully
# to heuristic-only behavior when absent)
pip install torch
```

There is no `zstandard` dependency. ASDP-LH (backend `0x04`) is the sole entropy backend.

### 2. Build the C++ Components

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 3. Run Tests

```bash
# C++ tests (no Python/pybind11 dependency)
./k2archive_test

# Training-data measurement tool (also no pybind11 dependency — see below)
./measure_gain_tool <file> <columnsplit_stride>

# Python integration tests
cd ../tests
python test_pipeline_integration.py        # works standalone, no pytest required
# or:
pytest test_pipeline_integration.py -v     # if pytest is installed
```

`test_pipeline_integration.py` runs correctly with or without `torch`/`pytest` installed — model-construction tests skip cleanly (reported, not silently dropped) when `torch` is unavailable, and the file falls back to a minimal compatible shim if `pytest` itself isn't installed.

---

## Usage

### Command Line Tool (`k2cli`)

```bash
# Compress / decompress a single file
./k2cli compress input.bin output.k2
./k2cli decompress output.k2 restored.bin

# Archive an entire directory (multi-volume .k2a, 4GB volumes by default)
./k2cli compress ./my_assets/ archive.k2a
./k2cli decompress archive.k2a.001 ./restored_assets/

# Verify a roundtrip end-to-end (file or directory; for directories this
# packs to a temp archive, unpacks, and recursively diffs the result)
./k2cli roundtrip ./my_assets/
```

### Python API

```python
from adaptive_optimizer import K2Pipeline, decode_k2_frame

pipeline = K2Pipeline(exploration=1.0, latency_weight=0.15)

with open("data.bin", "rb") as f:
    data = f.read()

# Analyze a sample and pick an initial strategy
hint = pipeline.prepare(data[:65536])
print(f"Detected data class: {hint.data_class.name}")
print(f"Suggested transforms: {hint.suggested_transforms}")  # usually [], unless COLUMNAR

# compress_full() applies the chosen Python-side transform (if any) and
# returns a K2 frame whose payload is NOT yet entropy-coded. This is the
# real C++ bridge's contract: it calls asdp_compress() on that payload,
# then reseal_frame() to splice the real compressed bytes back in.
frame = pipeline.compress_full(data)
```

The C++ bridge (`k2_bridge.cpp`) is what actually drives this sequence in production — `compress_full` → `asdp_compress` → `reseal_frame`. Calling `compress_full`/`reseal_frame` directly from pure Python (without a real ASDP call in between) is useful for testing the transform/frame logic in isolation, which is exactly what `test_pipeline_integration.py` does — see that file for the pattern.

### C++ API

```c
#include "k2_bridge.h"

int main() {
    K2Handle* pipeline = k2_create(nullptr, 1.0, 0.15);

    // Prepare on a sample
    k2_prepare(pipeline, sample_buffer, sample_size);

    // Compress (internally: prepare→ASDP→reseal, all C++-side)
    size_t compressed_size = 0;
    k2_compress(pipeline, src, src_len, dst, dst_capacity, &compressed_size);

    k2_destroy(pipeline);
    return 0;
}
```

---

## ML Models (optional)

K2 has two small ML levers, both optional — the pipeline works correctly with neither trained, falling back to heuristics:

1. **Structure classifier** — replaces `StructureDiscovery`'s hand-tuned heuristic thresholds with a trained model (64-dim engineered features → 8-class softmax).
2. **Transform-gain predictor** — predicts whether `columnsplit` will actually help, trained on *real* ASDP/CM measurements (not a proxy), replacing a fixed `zlib`-ratio guard threshold.

Neither ships a trained model by default. Synthetic training data was deliberately rejected for this project — a classifier trained on clean synthetic arrays would learn to recognize clean synthetic arrays, not the messier byte layouts real game assets actually have. Training requires a real, representative directory (a game install, asset dump, etc.):

```bash
# 1. Label a real directory (semi-supervised: heuristic pre-labels
#    everything, low-confidence chunks go to a review queue)
python label_corpus.py --data-dir /path/to/game --output-dir labels/

# 2. Review the low-confidence queue by hand (keyboard-driven, saves
#    progress after every decision)
python review_labels.py --input labels/needs_review.jsonl

# 3. Train the structure classifier
python train_predictor.py classifier \
    --labels-dir labels/ \
    --output models/structure_classifier.onnx

# 4. Build the gain-measurement tool (pure C++, no pybind11 needed) and
#    train the gain predictor against REAL ASDP/CM compression results
g++ -std=c++23 -O2 -Iinclude src/cpp/measure_gain_tool.cpp libasdp.a \
    -o measure_gain_tool

python train_predictor.py gain-predictor \
    --data-dir /path/to/game \
    --measure-tool ./measure_gain_tool \
    --output models/gain_predictor.onnx
```

Point `StructureDiscovery`/`K2Pipeline` at the resulting `.onnx` files via their `onnx_model_path` constructor argument to use them at runtime.

---

## Integration Notes

- The C++ bridge safely manages the embedded Python runtime and releases the GIL during performance-critical operations.
- Multiple threads may share a single `K2Handle`: Python calls (compress, decompress, prepare) are serialized by the GIL, while ASDP runs GIL-free in parallel within each call.
- Compressed `.k2` frames and `.k2a` archives are self-describing and versioned.
- `K2A` archives are processed one block at a time sequentially (each block internally parallelizes across `n_threads` via ASDP); this keeps memory bounded to one block's buffers regardless of total archive size, and is what makes the K2A layer thread-safe by construction — see `k2a_format_design.md` for the full design rationale, including a documented bug (now fixed) where this sequential design initially defeated internal parallelism.
- ASDP/CM has been audited with ThreadSanitizer and AddressSanitizer against its real parallel compress/decompress path; two data races and one heap-buffer-underflow were found and fixed. See the ASDP-LH README for details.

---

## License

This project is licensed under the 3-Clause BSD License.

---

For detailed API documentation, benchmark results, and training guides, see the `docs/` folder (when available) or the test suite.
