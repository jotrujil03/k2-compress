# K2 Compression Pipeline

**K2** is a sophisticated, adaptive end-to-end compression pipeline that combines automatic structure discovery, hybrid neural/statistical prediction, and bandit-driven optimization on top of the high-performance **ASDP-LH** bit-domain context-mixing codec.

It intelligently analyzes incoming data streams in real time, detects underlying structure, applies optimal transforms, and dynamically selects compression strategies to achieve superior ratios across text, numerical time-series, columnar data, and mixed binary assets.

---

## Key Features

- **Automatic Structure Discovery** — Analyzes entropy, repeat patterns, element alignment, and strides to classify data (Timeseries, Float Arrays, Text, Columnar, etc.).
- **Hybrid Predictor** — Combines fast online Markov models with optional deep learning models (LSTM / causal Transformers) via log-linear mixing.
- **Bandit-Driven Adaptive Optimizer** — Uses UCB1 multi-armed bandit algorithm to continuously tune strategy parameters for best ratio vs. speed trade-off.
- **ASDP-LH Backend** — Leverages the powerful bit-level logistic mixing engine for final entropy coding.
- **Multi-Format Support** — Single-file `.k2` frames and multi-volume `.k2a` directory archives.

---

## Directory Structure
```text
k2/
├── include/
│   ├── k2_bridge.h          # C++ RAII wrapper and plain C API
│   └── k2archive.h          # K2A multi-volume archive format
├── src/
│   ├── cpp/
│   │   ├── k2_bridge.cpp    # Bridge implementation + CPython management
│   │   ├── k2cli.cpp        # Command-line interface
│   │   └── k2archive.cpp    # Multi-volume archive handling
│   └── python/
│       ├── structure_discovery.py   # Layer 1: Data type & pattern detection
│       ├── hybrid_predictor.py      # Layer 2: Statistical + neural mixing
│       ├── adaptive_optimizer.py    # Layer 3: UCB1 bandit strategy selection
│       └── train_predictor.py       # Offline model training utilities
└── tests/
├── k2archive_test.cpp
└── test_pipeline_integration.py
```
---

## Architecture Overview

1. **Layer 1: Structure Discovery**  
   Samples the input stream to compute byte entropy, repeat density, element widths, and alignment. Maps data to the most suitable transformation strategy.

2. **Layer 2: Hybrid Predictor**  
   Blends fast order-1 Markov models with optional neural networks (LSTM or Mini-Transformers) using sophisticated probability mixing.

3. **Layer 3: Adaptive Optimizer**  
   Employs a UCB1 multi-armed bandit to dynamically select and tune the best compression strategy per chunk, balancing compression ratio and throughput.

The final entropy coding is performed by the **ASDP-LH** engine.

---

## Quick Start

### 1. Install Python Dependencies
```bash
pip install numpy onnxruntime zstandard

# Optional: for training neural predictors
pip install torch
2. Build the C++ Components
Bashmkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
3. Run Tests
Bash# C++ tests
./bin/k2_test
./bin/k2archive_test

# Python tests
python -m pytest src/python/

```

# Usage
## Command Line Tool (k2cli)
```bash

# Compress single file
./bin/k2cli compress input.bin output.k2

# Decompress single file
./bin/k2cli decompress output.k2 restored.bin

# Archive entire directory (multi-volume .k2a)
./bin/k2cli compress ./my_assets/ archive.k2a

# Extract .k2a archive
./bin/k2cli decompress archive.k2a.001 ./restored_assets/
```

# Python API
```python
from adaptive_optimizer import K2Pipeline

pipeline = K2Pipeline(exploration=1.0, latency_weight=0.15)

with open("data.bin", "rb") as f:
    data = f.read()

# Analyze and prepare strategy
hint = pipeline.prepare(data[:65536])
print(f"Detected Data Class: {hint.data_class.name}")

# Full compression
compressed_frame = pipeline.compress_full(data)
```

# C++ API
```c
#include "k2_bridge.h"

int main() {
    K2Handle* pipeline = k2_create(nullptr, 1.0, 0.15);

    // Prepare on sample data
    k2_prepare(pipeline, sample_buffer, sample_size);

    // Compress
    size_t compressed_size = 0;
    k2_compress(pipeline, src, src_len, dst, dst_capacity, &compressed_size);

    k2_destroy(pipeline);
    return 0;
}
```

# Integration Notes

The C++ bridge safely manages the embedded Python runtime and releases the GIL during performance-critical operations.
K2Handle objects are not thread-safe; use one per thread or serialize access.
Compressed .k2 frames and .k2a archives are self-describing and versioned.

---

License
This project is licensed under the 3-Clause BSD License.

---

For detailed API documentation, benchmark results, and training guides, see the docs/ folder (when available) or the test suite.