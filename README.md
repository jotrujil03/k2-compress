# OpenZL Neural Extension

Layering **auto structure discovery**, **hybrid neural prediction**, and
**adaptive optimization** on top of Meta's [OpenZL](https://github.com/facebook/openzl)
compression framework.

```
openzl_neural/
├── include/
│   └── openzl_neural_bridge.h    C/C++ bridge (pybind11 + plain C API)
├── src/
│   └── python/
│       ├── structure_discovery.py   Layer 1 — auto data-type detection
│       ├── hybrid_predictor.py      Layer 2 — neural + statistical mixer
│       ├── adaptive_optimizer.py    Layer 3 — UCB1 bandit strategy tuner
│       └── train_predictor.py       Offline LSTM / MiniTransformer training
├── tests/
│   └── test_pipeline_integration.py  Full-stack tests + benchmark
├── models/                           (output dir for trained .pt / .onnx files)
└── configs/                          (optional SDDL graph configs)
```

---

## Architecture

```
Raw data stream
      │
      ▼
┌─────────────────────────────┐
│   StructureDiscovery        │  Layer 1
│   • Entropy analysis        │  Lightweight ML probe
│   • Delta / repeat heuristics│  + optional ONNX classifier
│   • Element-size detection  │
│   → StructureHint           │
└──────────────┬──────────────┘
               │ hint (DataClass, transforms[], element_size, ...)
               ▼
┌─────────────────────────────┐
│   AdaptiveOptimizer         │  Layer 3
│   • UCB1 bandit over        │  (initialised with hint-specific
│     strategy pool           │   strategy candidates)
│   • Per-chunk timing        │
│   → Strategy (alpha, mode,  │
│     chunk_size, transforms) │
└──────────────┬──────────────┘
               │ HybridConfig
               ▼
┌─────────────────────────────┐
│   HybridPredictor           │  Layer 2
│   • StatisticalModel        │  Order-1 Markov (always on)
│     (order-1 Markov)        │
│   • NeuralBackend           │  One of:
│     ├─ LinearCMA            │    Weighted order-N model blend
│     ├─ LSTMPredictor        │    PyTorch LSTM (~200K params)
│     ├─ MiniTransformer      │    4-layer causal transformer
│     └─ ONNXPredictor        │    Any exported ONNX model
│   • Log-linear mixing       │
│   → blended P(next byte)    │
│   → ArithmeticEncoder       │
└──────────────┬──────────────┘
               │
               ▼
      OpenZL graph backend
      (Zstd / SDDL transforms)
```

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install numpy onnxruntime zstandard
# Optional for LSTM/Transformer training:
pip install torch
```

### 2. Run the self-tests

```bash
cd openzl_neural
python tests/test_pipeline_integration.py
```

### 3. Use the pipeline in Python

```python
from adaptive_optimizer import OpenZLNeuralPipeline

pipeline = OpenZLNeuralPipeline(
    exploration=1.0,       # UCB1 exploration constant
    latency_weight=0.15,   # speed vs. ratio trade-off
)

with open("my_data.bin", "rb") as f:
    data = f.read()

# Analyse structure, initialise strategies
hint = pipeline.prepare(data[:65536])
print(f"Detected: {hint.data_class.name}")
print(f"Suggested transforms: {hint.suggested_transforms}")

# Compress (adaptive; improves with each call)
compressed = pipeline.compress(data)
print(f"Ratio: {len(data)/len(compressed):.2f}x")
print(pipeline.stats())
```

### 4. Train a neural predictor (optional, improves ratio)

```bash
python src/python/train_predictor.py \
    --data-dir /path/to/representative/data \
    --model lstm \
    --embed-dim 32 \
    --hidden 128 \
    --epochs 10 \
    --output models/predictor.pt \
    --export-onnx models/predictor.onnx
```

Then pass `onnx_model_path="models/predictor.onnx"` to `OpenZLNeuralPipeline`.

### 5. C++ integration (with pybind11)

```cpp
// In your OpenZL codec node:
#include "openzl_neural_bridge.h"

// Option A: C++ RAII wrapper
ozl_neural::NeuralBridge bridge(
    "./src/python",          // directory with .py files
    "models/predictor.onnx", // optional ONNX model
    1.0,                     // exploration
    0.15                     // latency_weight
);
bridge.prepare(header_bytes, header_len);
auto compressed = bridge.compress(data, data_len);

// Option B: Plain C API
OZLNeuralHandle* h = ozl_neural_create(
    "models/predictor.onnx", 1.0, 0.15
);
ozl_neural_prepare(h, sample, sample_len);
uint8_t out[MAX_OUT]; size_t out_len;
ozl_neural_compress(h, src, src_len, out, sizeof(out), &out_len);
printf("%s\n", ozl_neural_stats(h));
ozl_neural_destroy(h);
```

Build with CMake:
```cmake
find_package(pybind11 REQUIRED)
target_link_libraries(your_target PRIVATE pybind11::embed)
target_include_directories(your_target PRIVATE openzl_neural/include)
```

---

## Layer Details

### Layer 1: Structure Discovery (`structure_discovery.py`)

| Signal | Method | Used for |
|---|---|---|
| Byte entropy | Shannon H(X) | Detect already-compressed data |
| Element width | Delta entropy minimisation | le-u32 vs le-u64 etc. |
| Delta gain | Before/after delta transform | Time-series detection |
| Repeat density | 8-byte window hashing | Integer array detection |
| Column stride | Autocorrelation | Struct-of-arrays detection |
| Float heuristic | IEEE 754 exponent spread | Float array detection |
| Text heuristic | Printable ASCII ratio | Text/CSV detection |
| ONNX classifier | 64-dim feature → 8 classes | Overrides heuristics when conf > 0.70 |

### Layer 2: Hybrid Predictor (`hybrid_predictor.py`)

**Predictor modes** (selectable per-chunk by the optimizer):

| Mode | Params | Speed | Quality |
|---|---|---|---|
| `LINEAR_CMA` | 0 (online) | ★★★★★ | ★★★ |
| `LSTM` | ~200K | ★★★ | ★★★★ |
| `MINI_TRANSFORMER` | ~1.2M | ★★ | ★★★★★ |
| `ONNX` | model-dependent | ★★★★ | ★★★★ |
| `PASSTHROUGH` | 0 | ★★★★★ | ★★ (Zstd only) |

**Mixing**: log-linear interpolation  
`log P_blend = (1-α)·log P_stat + α·log P_neural`  
Alpha is tuned dynamically by the optimizer.

### Layer 3: Adaptive Optimizer (`adaptive_optimizer.py`)

**UCB1 bandit** selects among strategy variants that differ in:
- Predictor mode + alpha
- Chunk size (latency / context quality trade-off)
- Transform chain (matches OpenZL SDDL graph path)
- Zstd fallback level

Score function per chunk:  
`score = (1-w)·log₂(ratio) + w·log₂(throughput_MBs)`

Alpha is gradient-free tuned every 10 chunks via performance trend.

---

## Integration with OpenZL's Graph System

OpenZL represents compression as a **DAG of codec nodes** (SDDL graph).
This extension adds a `NeuralCodec` node type that:

1. Sits **after** OpenZL's structural transforms (delta, column-split, etc.)
2. Replaces or supplements the final Zstd entropy-coding stage
3. Reports its compressed output back into the graph's typed message stream

The `StructureHint.suggested_transforms` list maps directly to
OpenZL SDDL primitive names, enabling auto-generation of a compression
graph from the discovery output.

---

## Roadmap / Contribution Ideas

- [ ] Grid/tensor transforms for ML model weights (float16/bfloat16)
- [ ] Distil a smaller LSTM (embed=16, hidden=64) for embedded targets
- [ ] ONNX Runtime quantisation (INT8) for faster CPU inference
- [ ] Online classifier training (label new data-types on the fly)
- [ ] Stream the arithmetic coder for truly zero-latency output
- [ ] Benchmark suite vs. OpenZL baseline + Zstd + LZMA on Silesia corpus
- [ ] Plug into OpenZL's `--trainer` to generate SDDL plans automatically

---

## License

BSD 3-Clause, consistent with OpenZL's own license.
