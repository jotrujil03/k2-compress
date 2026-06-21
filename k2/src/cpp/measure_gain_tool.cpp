// k2/src/cpp/measure_gain_tool.cpp
//
// Standalone CLI: measures REAL ASDP/CM compressed size for a chunk of
// data, both as-is and after applying columnsplit-<stride>, and prints
// both sizes. Used by train_predictor.py to generate ground-truth
// training labels for TransformGainPredictor — calling the actual ASDP
// library rather than a proxy (zlib, entropy heuristics, etc.), since
// the whole point of this model is to predict what ASDP/CM will actually
// do, not what a cheaper stand-in predicts.
//
// Built as a standalone binary (not a pybind11 module) specifically so
// it can be compiled, run, and verified in any environment with a C++
// toolchain and libasdp.a, without requiring Python/C++ bindings to be
// built first — see k2_bridge.cpp for the real pybind11 bridge used in
// production; this tool exists purely as a training-data-generation
// utility and is not part of the runtime compression pipeline.
//
// Usage:
//   measure_gain_tool <input_file> <columnsplit_stride> [offset] [length]
//
// offset/length are OPTIONAL and bound the read to a chunk of the file
// instead of reading the whole thing -- added after a real run against a
// ~20GB game-asset directory showed this tool reading and compressing
// entire multi-hundred-MB/GB files (BSA archives) per "sample" when
// train_predictor.py's gain-predictor command actually intends each
// sample to be a bounded probe_bytes-sized chunk (matching
// label_corpus.py's ChunkRef model elsewhere in this pipeline). Without
// offset/length, a single large input file could make one "sample" take
// many minutes of real CM compression time -- confirmed directly via
// py-spy showing the Python caller blocked in subprocess.communicate()
// on exactly this. If omitted, reads the whole file (preserves the
// original behavior for direct/manual invocation).
//
// Output (stdout, one line, space-separated):
//   <orig_bytes> <raw_compressed_bytes> <columnsplit_compressed_bytes>
//
// Exit code 0 on success, nonzero on any error (message on stderr).

#include "asdp/asdp.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

// Reads either the whole file (length < 0) or exactly `length` bytes
// starting at `offset` (clamped to the actual file size, so a chunk that
// runs past EOF -- e.g. the last chunk of a file whose size isn't an
// exact multiple of probe_bytes -- reads whatever's left rather than
// failing).
std::vector<uint8_t> read_file(const char* path, long long offset,
                                long long length, bool* ok) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { *ok = false; return {}; }
    const long long file_size = static_cast<long long>(f.tellg());

    long long start = (offset < 0) ? 0 : offset;
    if (start > file_size) start = file_size;

    long long want = (length < 0) ? (file_size - start) : length;
    long long available = file_size - start;
    if (want > available) want = available;
    if (want < 0) want = 0;

    f.seekg(start);
    std::vector<uint8_t> buf(static_cast<size_t>(want));
    if (want > 0) {
        f.read(reinterpret_cast<char*>(buf.data()), want);
    }
    *ok = bool(f) || f.eof();
    return buf;
}

// Mirrors adaptive_optimizer.py's _apply_columnsplit exactly (true
// transpose of stride-byte rows) — kept as a literal byte-for-byte port
// rather than a "close enough" reimplementation, since training labels
// must reflect the SAME transform the live Python pipeline would apply,
// not a similar one.
std::vector<uint8_t> columnsplit(const std::vector<uint8_t>& data, int stride) {
    const size_t n = data.size();
    if (stride <= 0 || size_t(stride) > n) return data;
    const size_t n_rows = n / size_t(stride);
    const size_t aligned = n_rows * size_t(stride);
    std::vector<uint8_t> out;
    out.reserve(n);
    // out[col * n_rows + row] = data[row * stride + col]  (transpose)
    out.resize(aligned);
    for (size_t row = 0; row < n_rows; ++row) {
        for (int col = 0; col < stride; ++col) {
            out[size_t(col) * n_rows + row] = data[row * size_t(stride) + size_t(col)];
        }
    }
    out.insert(out.end(), data.begin() + ptrdiff_t(aligned), data.end());
    return out;
}

bool measure_compressed_size(const std::vector<uint8_t>& data, size_t* out_size) {
    asdp_config_t cfg = asdp_default_config();
    cfg.n_threads = 1;  // deterministic, single-block path for training-data generation
    asdp_ctx_t* ctx = asdp_create(&cfg);
    if (!ctx) return false;

    std::vector<uint8_t> comp(asdp_compress_bound(data.size()));
    size_t comp_len = 0;
    const int rc = asdp_compress(ctx, data.data(), data.size(),
                                  comp.data(), comp.size(), &comp_len);
    asdp_destroy(ctx);
    if (rc != ASDP_OK) return false;
    *out_size = comp_len;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::fprintf(stderr,
            "usage: %s <input_file> <columnsplit_stride> [offset] [length]\n"
            "  offset/length are optional; omit both to read the whole file.\n",
            argv[0]);
        return 1;
    }
    const char* path = argv[1];
    const int stride = std::atoi(argv[2]);
    if (stride <= 0) {
        std::fprintf(stderr, "stride must be > 0, got %d\n", stride);
        return 1;
    }

    long long offset = -1, length = -1;
    if (argc == 5) {
        offset = std::atoll(argv[3]);
        length = std::atoll(argv[4]);
        if (offset < 0 || length < 0) {
            std::fprintf(stderr, "offset/length must be >= 0\n");
            return 1;
        }
    }

    bool ok = false;
    std::vector<uint8_t> data = read_file(path, offset, length, &ok);
    if (!ok) {
        std::fprintf(stderr, "failed to read %s\n", path);
        return 1;
    }
    if (data.empty()) {
        std::fprintf(stderr, "empty input (file or requested range)\n");
        return 1;
    }

    size_t raw_compressed = 0;
    if (!measure_compressed_size(data, &raw_compressed)) {
        std::fprintf(stderr, "asdp_compress failed on raw data\n");
        return 1;
    }

    std::vector<uint8_t> split = columnsplit(data, stride);
    size_t split_compressed = 0;
    if (!measure_compressed_size(split, &split_compressed)) {
        std::fprintf(stderr, "asdp_compress failed on columnsplit data\n");
        return 1;
    }

    std::printf("%zu %zu %zu\n", data.size(), raw_compressed, split_compressed);
    return 0;
}

