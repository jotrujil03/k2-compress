#!/usr/bin/env bash
# k2/scripts/smoke_test_ml.sh
#
# Unattended smoke test for the ML training pipeline (label_corpus.py ->
# train_predictor.py -> ONNX inference). Generates a small synthetic corpus
# and runs the FULL pipeline end to end, fast (a couple minutes), to shake
# out bugs in the torch-dependent training code before spending real time
# labelling/training against actual game data.
#
# This is NOT how you should train models for real use:
#   - The corpus here is synthetic, which this project deliberately
#     rejected as a real training source (see label_corpus.py's module
#     docstring) -- a model "trained" by this script will not generalize
#     to real game assets. It exists only to prove the CODE runs correctly.
#   - --auto-accept-threshold 0.0 auto-accepts every chunk with no human
#     review, since this is throwaway data. Real runs should use
#     review_labels.py on the low-confidence queue.
#
# Usage (run from anywhere -- the script locates its own files):
#   bash k2/scripts/smoke_test_ml.sh
#
# Requires: pip install numpy torch onnxruntime. Step 4 (gain predictor)
# also needs a buildable ASDP-LH checkout -- it searches a few likely
# locations based on the real documented layout (k2/ under k2-compress/,
# with asdp-lh/ as a SIBLING OF k2-compress/, not of k2/ itself), or set
# K2_ASDP_DIR explicitly:
#   K2_ASDP_DIR=~/Documents/asdp-lh bash scripts/smoke_test_ml.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# label_corpus.py / train_predictor.py / structure_discovery.py /
# hybrid_predictor.py all live in k2/src/python/ and import each other by
# bare module name (e.g. "from structure_discovery import ..."), which
# only resolves correctly if the CURRENT WORKING DIRECTORY is that
# directory when python3 runs them -- they are not installed as a package.
# Previously this script called `python3 label_corpus.py` as a bare
# relative path and relied on the CALLER having already cd'd into
# k2/src/python/ themselves; if invoked from anywhere else (e.g. from
# k2/ directly), python3 looked for label_corpus.py relative to the
# CALLER's cwd, not the script's own location, and failed with "No such
# file or directory". Fixed by deriving the python source dir from
# SCRIPT_DIR (which bash gives us reliably regardless of cwd) and cd'ing
# into it explicitly, so this script works no matter where it's invoked
# from.
PY_SRC_DIR="$SCRIPT_DIR/../src/python"
if [ ! -f "$PY_SRC_DIR/label_corpus.py" ]; then
    echo "FAIL: label_corpus.py not found at $PY_SRC_DIR"
    echo "      (expected k2/scripts/smoke_test_ml.sh next to k2/src/python/)"
    exit 1
fi
cd "$PY_SRC_DIR"
echo "running from: $(pwd)"

WORK="$(mktemp -d)"
trap 'echo; echo "(scratch dir kept for inspection: $WORK)"' EXIT

echo "=== K2 ML pipeline smoke test ==="
echo "scratch dir: $WORK"
echo

# ---------------------------------------------------------------------------
# Step 0: dependency check
# ---------------------------------------------------------------------------
echo "--- Step 0: checking dependencies ---"
python3 -c "import numpy" || { echo "FAIL: numpy not installed"; exit 1; }
python3 -c "import torch; print('torch', torch.__version__)" \
    || { echo "FAIL: torch not installed (pip install torch)"; exit 1; }
python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)" \
    || { echo "FAIL: onnxruntime not installed (pip install onnxruntime)"; exit 1; }
echo "OK"
echo

# ---------------------------------------------------------------------------
# Step 1: build a small, varied SYNTHETIC corpus
# ---------------------------------------------------------------------------
# Deliberately varied: text, columnar (u32,u32,f32 records -- the layout
# StructureDiscovery's column-stride detector actually recognizes, see
# structure_discovery.py's _detect_column_stride docstring), timeseries,
# float arrays, and incompressible random data. Enough files/chunks to
# clear train_predictor.py's own minimum-sample guardrails (gain-predictor
# needs >=20 measured samples; classifier needs a usable train/val split)
# without taking long to run.
echo "--- Step 1: generating synthetic corpus ---"
CORPUS="$WORK/corpus"
mkdir -p "$CORPUS"

python3 - "$CORPUS" << 'PYEOF'
import sys, os
import numpy as np

corpus = sys.argv[1]
rng = np.random.default_rng(42)

def write(name, data):
    with open(os.path.join(corpus, name), "wb") as f:
        f.write(data)

# Several text files (varied content, not identical, so chunking/labelling
# sees real variety)
for i in range(8):
    words = f"sample log entry {i} status ok request handled latency low ".encode()
    write(f"log_{i}.txt", words * 400)

# Several columnar files: (u32, u32, f32) records, 12 bytes/row -- the one
# layout the real column-stride detector reliably recognizes (verified
# directly during development; see structure_discovery.py).
for i in range(8):
    n = 3000
    a = rng.integers(0, 1000, n).astype("<u4")
    b = np.arange(n).astype("<u4")
    c = rng.standard_normal(n).astype("<f4")
    rec = np.zeros(n, dtype=[("a", "<u4"), ("b", "<u4"), ("c", "<f4")])
    rec["a"] = a; rec["b"] = b; rec["c"] = c
    write(f"records_{i}.bin", rec.tobytes())

# Timeseries (monotone cumulative u64)
for i in range(6):
    n = 8192
    ts = np.cumsum(rng.integers(0, 100, n)).astype("<u8")
    write(f"timeseries_{i}.bin", ts.tobytes())

# Float arrays
for i in range(6):
    write(f"floats_{i}.bin", rng.standard_normal(8192).astype("<f4").tobytes())

# Random/incompressible
for i in range(4):
    write(f"random_{i}.bin", os.urandom(40000))

n_files = len(os.listdir(corpus))
total_bytes = sum(os.path.getsize(os.path.join(corpus, f)) for f in os.listdir(corpus))
print(f"wrote {n_files} files, {total_bytes/1e6:.2f} MB total")
PYEOF
echo

# ---------------------------------------------------------------------------
# Step 2: label_corpus.py (full auto-accept -- no human review for this
# throwaway run)
# ---------------------------------------------------------------------------
echo "--- Step 2: labelling (auto-accept-threshold=0.0, unattended) ---"
LABELS="$WORK/labels"
python3 label_corpus.py \
    --data-dir "$CORPUS" \
    --output-dir "$LABELS" \
    --probe-bytes 65536 \
    --max-chunks-per-file 2 \
    --auto-accept-threshold 0.0
echo

N_LABELLED=$(wc -l < "$LABELS/auto_accepted.jsonl" || echo 0)
echo "labelled chunks: $N_LABELLED"
if [ "$N_LABELLED" -lt 15 ]; then
    echo "FAIL: too few labelled chunks ($N_LABELLED) to proceed -- corpus too small?"
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# Step 3: train the structure classifier
# ---------------------------------------------------------------------------
echo "--- Step 3: training structure classifier ---"
MODELS="$WORK/models"
mkdir -p "$MODELS"
python3 train_predictor.py classifier \
    --labels-dir "$LABELS" \
    --epochs 5 \
    --batch-size 8 \
    --output "$MODELS/structure_classifier.onnx"
echo

if [ ! -f "$MODELS/structure_classifier.onnx" ]; then
    echo "FAIL: classifier ONNX file was not produced"
    exit 1
fi
echo "OK: $(du -h "$MODELS/structure_classifier.onnx" | cut -f1) classifier model produced"
echo

# ---------------------------------------------------------------------------
# Step 4: build measure_gain_tool (if not already built) and train the gain
# predictor
# ---------------------------------------------------------------------------
echo "--- Step 4: gain predictor ---"
MEASURE_TOOL="$WORK/measure_gain_tool"

K2_CPP_SRC="$SCRIPT_DIR/../src/cpp/measure_gain_tool.cpp"

# Locate the ASDP-LH checkout. Real documented layout is:
#   ~/Documents/k2-compress/k2/        <- this script lives under k2/scripts/
#   ~/Documents/asdp-lh/                <- ASDP-LH, a SIBLING OF k2-compress/,
#                                          NOT a sibling of k2/ itself
# i.e. from k2/scripts/ that's ../../../asdp-lh (up out of k2/, up out of
# k2-compress/, into asdp-lh/) -- three levels up, easy to get wrong by one.
# Set K2_ASDP_DIR explicitly to override if your layout differs:
#   K2_ASDP_DIR=/path/to/asdp-lh bash smoke_test_ml.sh
CANDIDATE_ASDP_DIRS=(
    "${K2_ASDP_DIR:-}"
    "$SCRIPT_DIR/../../../asdp-lh"   # ~/Documents/asdp-lh (real documented layout)
    "$SCRIPT_DIR/../../asdp-lh"      # in case k2/ has no k2-compress/ wrapper
    "$SCRIPT_DIR/../../asdp-lh-cm"   # this project's own outputs/ staging name
)

ASDP_DIR=""
for d in "${CANDIDATE_ASDP_DIRS[@]}"; do
    [ -z "$d" ] && continue
    if [ -d "$d/include/asdp" ] && [ -d "$d/src" ]; then
        ASDP_DIR="$d"
        break
    fi
done

if [ ! -f "$K2_CPP_SRC" ]; then
    echo "SKIP: measure_gain_tool.cpp not found at $K2_CPP_SRC -- adjust K2_CPP_SRC in this script"
elif [ -z "$ASDP_DIR" ]; then
    echo "SKIP: could not find an ASDP-LH checkout. Tried:"
    for d in "${CANDIDATE_ASDP_DIRS[@]}"; do
        [ -n "$d" ] && echo "  $d"
    done
    echo "Set K2_ASDP_DIR=/path/to/asdp-lh and re-run, e.g.:"
    echo "  K2_ASDP_DIR=~/Documents/asdp-lh bash scripts/smoke_test_ml.sh"
else
    echo "using ASDP-LH checkout: $ASDP_DIR"
    echo "building measure_gain_tool..."
    g++ -std=c++23 -O2 -I "$ASDP_DIR/include" \
        "$K2_CPP_SRC" "$ASDP_DIR"/src/*.cpp \
        -pthread -o "$MEASURE_TOOL" \
        || { echo "FAIL: measure_gain_tool build failed"; exit 1; }
    echo "OK: built $MEASURE_TOOL"
    echo

    python3 train_predictor.py gain-predictor \
        --data-dir "$CORPUS" \
        --measure-tool "$MEASURE_TOOL" \
        --max-samples 60 \
        --epochs 5 \
        --batch-size 8 \
        --output "$MODELS/gain_predictor.onnx"
    echo

    if [ ! -f "$MODELS/gain_predictor.onnx" ]; then
        echo "FAIL: gain predictor ONNX file was not produced"
        exit 1
    fi
    echo "OK: $(du -h "$MODELS/gain_predictor.onnx" | cut -f1) gain predictor model produced"
fi
echo

# ---------------------------------------------------------------------------
# Step 5: load both models at inference time and sanity-check real predictions
# ---------------------------------------------------------------------------
echo "--- Step 5: inference smoke test ---"
python3 - "$MODELS" "$CORPUS" << 'PYEOF'
import sys, os
import numpy as np

models_dir, corpus_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.getcwd())

from structure_discovery import StructureDiscovery, DataClass

clf_path = os.path.join(models_dir, "structure_classifier.onnx")
disc = StructureDiscovery(onnx_model_path=clf_path if os.path.exists(clf_path) else None)

print(f"classifier model present: {os.path.exists(clf_path)}")

failures = 0
for fname in sorted(os.listdir(corpus_dir))[:6]:
    with open(os.path.join(corpus_dir, fname), "rb") as f:
        data = f.read()
    hint = disc.analyze(data)
    print(f"  {fname:20s} -> class={hint.data_class.name:14s} conf={hint.confidence:.2f} "
          f"transforms={hint.suggested_transforms}")
    if hint.data_class is None:
        failures += 1

from hybrid_predictor import ONNXGainPredictor
from structure_discovery import _extract_features

gain_path = os.path.join(models_dir, "gain_predictor.onnx")
if os.path.exists(gain_path):
    gp = ONNXGainPredictor(gain_path)
    print(f"\ngain predictor available: {gp.available}")
    sample_file = sorted(os.listdir(corpus_dir))[0]
    with open(os.path.join(corpus_dir, sample_file), "rb") as f:
        data = f.read()
    feats = _extract_features(data, element_size=4)
    pred = gp.predict(feats)
    print(f"  sample prediction on {sample_file}: {pred}")
    if pred is None:
        failures += 1
else:
    print("\ngain predictor model not present (step 4 was skipped or failed)")

if failures:
    print(f"\nFAIL: {failures} inference check(s) failed")
    sys.exit(1)
print("\nOK: inference checks passed")
PYEOF

echo
echo "=== Smoke test complete ==="
