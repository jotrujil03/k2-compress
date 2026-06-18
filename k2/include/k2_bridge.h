/**
 * k2/include/k2_bridge.h
 *
 * C++ Bridge: K2 Neural Pipeline → OpenZL wire format
 * -----------------------------------------------------
 * Architecture (compress path):
 *
 *   Raw bytes
 *       │
 *       ▼
 *   K2Pipeline (Python)       ← structure discovery + transforms
 *       │  pre-processed bytes
 *       ▼
 *   OpenZL CCtx::compressSerial  ← serial profile, produces valid .zl frame
 *       │
 *       ▼
 *   Valid OpenZL frame  ← decompressible by `zli decompress`
 *
 * Architecture (decompress path):
 *
 *   OpenZL frame
 *       │
 *       ▼
 *   OpenZL DCtx::decompress   ← reverses entropy coding
 *       │  pre-processed bytes
 *       ▼
 *   K2Pipeline.decompress()   ← reverses K2 transforms
 *       │
 *       ▼
 *   Original bytes
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Export macro
#if defined(_WIN32)
#  define K2_API __declspec(dllexport)
#else
#  define K2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct K2Handle K2Handle;

K2_API K2Handle* k2_create(
    const char* onnx_model_path,
    double      exploration,
    double      latency_weight
);

K2_API int k2_prepare(
    K2Handle*      handle,
    const uint8_t* sample,
    size_t         sample_len
);

/** Compress src → valid OpenZL frame in dst. */
K2_API int k2_compress(
    K2Handle*      handle,
    const uint8_t* src,
    size_t         src_len,
    uint8_t*       dst,
    size_t         dst_cap,
    size_t*        out_len
);

/** Decompress an OpenZL frame produced by k2_compress. */
K2_API int k2_decompress(
    K2Handle*      handle,
    const uint8_t* src,
    size_t         src_len,
    uint8_t*       dst,
    size_t         dst_cap,
    size_t*        out_len
);

K2_API const char* k2_stats(K2Handle* handle);
K2_API void        k2_free_str(const char* s);
K2_API void        k2_destroy(K2Handle* handle);

#ifdef __cplusplus
}  // extern "C"
#endif


// ---------------------------------------------------------------------------
// C++ RAII wrapper
// ---------------------------------------------------------------------------

#ifdef __cplusplus

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// OpenZL C++ API
#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/DCtx.hpp"
#include "openzl/zl_compress.h"
#include "openzl/zl_decompress.h"
#include "openzl/codecs/zl_segmenters.h"
#include "openzl/zl_graphs.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace k2 {

// ---------------------------------------------------------------------------
// OpenZL serial profile compressor (built once, reused)
// ---------------------------------------------------------------------------

inline openzl::Compressor make_serial_compressor() {
    openzl::Compressor comp;
    ZL_GraphID inner = ZL_Compressor_buildACEGraphWithDefault(
        comp.get(), ZL_GRAPH_LZ);
    ZL_GraphID graph = ZL_Compressor_buildSerialSegmenter(
        comp.get(), ZL_DEFAULT_SEGMENTER_CHUNK_BYTE_SIZE, inner);
    comp.selectStartingGraph(graph);
    return comp;
}

// ---------------------------------------------------------------------------
// Bridge
// ---------------------------------------------------------------------------

class PYBIND11_EXPORT Bridge {
public:
    explicit Bridge(
        const std::string& python_module_dir,
        const std::string& onnx_model_path = "",
        double exploration    = 1.0,
        double latency_weight = 0.15
    ) : _compressor(make_serial_compressor())
    {
        py::gil_scoped_acquire gil;

        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, python_module_dir);

        py::module_ mod = py::module_::import("adaptive_optimizer");
        py::object  cls = mod.attr("K2Pipeline");

        py::object onnx = onnx_model_path.empty()
            ? py::none()
            : py::cast(onnx_model_path);

        _pipeline = cls(
            "onnx_model_path"_a = onnx,
            "exploration"_a     = exploration,
            "latency_weight"_a  = latency_weight
        );
    }

    std::string prepare(const uint8_t* data, size_t len) {
        py::gil_scoped_acquire gil;
        py::bytes sample(reinterpret_cast<const char*>(data), len);
        py::object hint = _pipeline.attr("prepare")(sample);
        return hint.attr("data_class").attr("name").cast<std::string>();
    }

    /**
     * Compress: K2 transforms → OpenZL serial frame.
     * Output is a valid .zl file decompressible by `zli decompress`.
     */
    std::vector<uint8_t> compress(const uint8_t* data, size_t len) {
        // Step 1: K2 Python transforms
        std::vector<uint8_t> transformed = k2_transform(data, len);

        // Step 2: OpenZL serial profile → valid .zl frame
        openzl::CCtx cctx;
        cctx.setParameter(openzl::CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        cctx.refCompressor(_compressor);

        std::string src(
            reinterpret_cast<const char*>(transformed.data()),
            transformed.size());
        std::string dst = cctx.compressSerial(src);

        return std::vector<uint8_t>(dst.begin(), dst.end());
    }

    /**
     * Decompress: OpenZL decompression → K2 inverse transforms.
     */
    std::vector<uint8_t> decompress(const uint8_t* data, size_t len) {
        // Step 1: OpenZL decompression
        openzl::DCtx dctx;
        std::string src(reinterpret_cast<const char*>(data), len);
        std::string transformed = dctx.decompressSerial(src);

        // Step 2: K2 inverse transforms
        return k2_inverse_transform(
            reinterpret_cast<const uint8_t*>(transformed.data()),
            transformed.size());
    }

    std::vector<uint8_t> compress(const std::vector<uint8_t>& data) {
        return compress(data.data(), data.size());
    }

    std::vector<uint8_t> decompress(const std::vector<uint8_t>& data) {
        return decompress(data.data(), data.size());
    }

    std::string stats() {
        py::gil_scoped_acquire gil;
        py::dict d = _pipeline.attr("stats")();
        py::module_ json = py::module_::import("json");
        return json.attr("dumps")(d).cast<std::string>();
    }

private:
    py::object          _pipeline;
    openzl::Compressor  _compressor;

    // Call K2Pipeline.compress_transforms() — returns pre-processed bytes
    // without the entropy coding step (Zstd), just the structural transforms.
    std::vector<uint8_t> k2_transform(const uint8_t* data, size_t len) {
        py::gil_scoped_acquire gil;
        py::bytes input(reinterpret_cast<const char*>(data), len);
        // compress() on the Python side applies transforms + Zstd.
        // We intercept at the transform level by calling compress_transforms().
        // If that method doesn't exist yet, fall back to raw bytes (transforms
        // will be added to the Python layer in the next step).
        py::bytes output;
        try {
            output = _pipeline.attr("compress_transforms")(input);
        } catch (const py::error_already_set&) {
            // Fallback: return raw bytes, let OpenZL handle entropy only
            PyErr_Clear();
            return std::vector<uint8_t>(data, data + len);
        }
        std::string s = output.cast<std::string>();
        return std::vector<uint8_t>(s.begin(), s.end());
    }

    // Reverse K2 transforms after OpenZL decompression
    std::vector<uint8_t> k2_inverse_transform(
            const uint8_t* data, size_t len) {
        py::gil_scoped_acquire gil;
        py::bytes input(reinterpret_cast<const char*>(data), len);
        try {
            py::bytes output = _pipeline.attr("decompress_transforms")(input);
            std::string s = output.cast<std::string>();
            return std::vector<uint8_t>(s.begin(), s.end());
        } catch (const py::error_already_set&) {
            // Fallback: return as-is (no transforms to reverse yet)
            PyErr_Clear();
            return std::vector<uint8_t>(data, data + len);
        }
    }
};

}  // namespace k2

#endif  // __cplusplus


// ---------------------------------------------------------------------------
// K2_IMPLEMENTATION block
// ---------------------------------------------------------------------------

#ifdef K2_IMPLEMENTATION
#ifdef __cplusplus

#include <cstdlib>

struct K2Handle {
    k2::Bridge* bridge     = nullptr;
    char*       last_stats = nullptr;
};

// k2_create and k2_destroy are in k2_bridge.cpp (manage interpreter lifetime)

extern "C" {

K2_API int k2_prepare(K2Handle* h, const uint8_t* sample, size_t len) {
    if (!h || !h->bridge) return -1;
    try {
        h->bridge->prepare(sample, len);
        return 0;
    } catch (...) { return -2; }
}

K2_API int k2_compress(K2Handle* h,
                const uint8_t* src, size_t src_len,
                uint8_t* dst, size_t dst_cap, size_t* out_len) {
    if (!h || !h->bridge || !dst || !out_len) return -1;
    try {
        auto compressed = h->bridge->compress(src, src_len);
        if (compressed.size() > dst_cap) return -3;
        std::memcpy(dst, compressed.data(), compressed.size());
        *out_len = compressed.size();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "k2_compress error: %s\n", e.what());
        return -2;
    } catch (...) { return -2; }
}

K2_API int k2_decompress(K2Handle* h,
                const uint8_t* src, size_t src_len,
                uint8_t* dst, size_t dst_cap, size_t* out_len) {
    if (!h || !h->bridge || !dst || !out_len) return -1;
    try {
        auto decompressed = h->bridge->decompress(src, src_len);
        if (decompressed.size() > dst_cap) return -3;
        std::memcpy(dst, decompressed.data(), decompressed.size());
        *out_len = decompressed.size();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "k2_decompress error: %s\n", e.what());
        return -2;
    } catch (...) { return -2; }
}

K2_API const char* k2_stats(K2Handle* h) {
    if (!h || !h->bridge) return "{}";
    try {
        std::string s = h->bridge->stats();
        free(h->last_stats);
        h->last_stats = static_cast<char*>(malloc(s.size() + 1));
        std::memcpy(h->last_stats, s.c_str(), s.size() + 1);
        return h->last_stats;
    } catch (...) { return "{}"; }
}

K2_API void k2_free_str(const char* /*s*/) {}

}  // extern "C"

#endif  // __cplusplus
#endif  // K2_IMPLEMENTATION
