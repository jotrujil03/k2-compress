/**
 * k2/include/k2_bridge.h
 *
 * C++ Bridge: OpenZL Graph ↔ K2 Neural Pipeline
 * -----------------------------------------------
 * Provides a thin C interface (extern "C") plus a C++ RAII wrapper so that
 * OpenZL's compression graph can call the K2 neural pipeline via
 * pybind11 or the plain CPython C API.
 *
 * Architecture:
 *   OpenZL graph node (C++)
 *       │
 *       ▼  [k2_compress / k2_stats]
 *   k2::Bridge (C++)   ──pybind11──►  K2Pipeline (Python)
 *       │
 *       ▼
 *   compressed bytes returned to graph
 *
 * Integration pattern in a custom OpenZL codec node:
 *
 *   #include "k2_bridge.h"
 *
 *   class K2Codec : public zl::Codec {
 *       k2::Bridge _bridge;
 *   public:
 *       K2Codec() : _bridge("path/to/src/python") {}
 *
 *       zl::Status compress(const zl::Buffer& in, zl::Buffer& out) override {
 *           return _bridge.compress(in.data(), in.size(), out);
 *       }
 *   };
 *
 * Build notes:
 *   - Link against libpython3.x and pybind11.
 *   - Set PYTHONPATH to include the src/python directory.
 *   - Python GIL is acquired/released around each call automatically.
 *   - Thread-safe: each Bridge holds its own Python object reference.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Plain C API (usable from pure C callers, Python ctypes, etc.)
// ---------------------------------------------------------------------------

// Export macro — ensures symbols are visible in libk2.so
#if defined(_WIN32)
#  define K2_API __declspec(dllexport)
#else
#  define K2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to a K2 pipeline instance. */
typedef struct K2Handle K2Handle;

/**
 * Create a new K2 pipeline.
 */
K2_API K2Handle* k2_create(
    const char* onnx_model_path,
    double      exploration,
    double      latency_weight
);

/**
 * Analyse a data sample and initialise strategy selection.
 * Must be called once before k2_compress on a new stream.
 */
K2_API int k2_prepare(
    K2Handle*      handle,
    const uint8_t* sample,
    size_t         sample_len
);

/**
 * Compress a buffer.
 */
K2_API int k2_compress(
    K2Handle*      handle,
    const uint8_t* src,
    size_t         src_len,
    uint8_t*       dst,
    size_t         dst_cap,
    size_t*        out_len
);

/**
 * Retrieve JSON stats string (caller must free with k2_free_str).
 */
K2_API const char* k2_stats(K2Handle* handle);

/** Free a string returned by k2_stats. */
K2_API void k2_free_str(const char* s);

/** Destroy the pipeline handle and free resources. */
K2_API void k2_destroy(K2Handle* handle);

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

namespace py = pybind11;
using namespace pybind11::literals;

namespace k2 {

/**
 * RAII wrapper around the Python K2Pipeline.
 *
 * Acquires GIL on every call; safe to use from multiple C++ threads
 * (each call is serialised through the GIL — for parallel chunk
 * compression, create one Bridge per worker thread).
 */
class PYBIND11_EXPORT Bridge {
public:
    /**
     * @param python_module_dir  Directory containing structure_discovery.py,
     *                           hybrid_predictor.py, adaptive_optimizer.py.
     * @param onnx_model_path    Optional ONNX model path; empty string = none.
     * @param exploration        UCB1 exploration constant.
     * @param latency_weight     Speed vs. ratio weight.
     */
    explicit Bridge(
        const std::string& python_module_dir,
        const std::string& onnx_model_path = "",
        double exploration    = 1.0,
        double latency_weight = 0.15
    ) {
        py::gil_scoped_acquire gil;

        // Extend sys.path so the pipeline modules are importable
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, python_module_dir);

        // Import K2Pipeline
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

    /**
     * Analyse a sample and prepare strategy pool.
     * @returns Detected DataClass name string (e.g. "TIMESERIES").
     */
    std::string prepare(const uint8_t* data, size_t len) {
        py::gil_scoped_acquire gil;
        py::bytes sample(reinterpret_cast<const char*>(data), len);
        py::object hint = _pipeline.attr("prepare")(sample);
        return hint.attr("data_class").attr("name").cast<std::string>();
    }

    /**
     * Compress data.  Returns compressed bytes as std::vector<uint8_t>.
     */
    std::vector<uint8_t> compress(const uint8_t* data, size_t len) {
        py::gil_scoped_acquire gil;
        py::bytes input(reinterpret_cast<const char*>(data), len);
        py::bytes output = _pipeline.attr("compress")(input);
        std::string s = output.cast<std::string>();
        return std::vector<uint8_t>(s.begin(), s.end());
    }

    /** Convenience overload for std::vector input. */
    std::vector<uint8_t> compress(const std::vector<uint8_t>& data) {
        return compress(data.data(), data.size());
    }

    /** Get JSON stats as string. */
    std::string stats() {
        py::gil_scoped_acquire gil;
        py::dict d = _pipeline.attr("stats")();
        py::module_ json = py::module_::import("json");
        return json.attr("dumps")(d).cast<std::string>();
    }

private:
    py::object _pipeline;
};


// ---------------------------------------------------------------------------
// OpenZL codec node adapter stub
// ---------------------------------------------------------------------------

/**
 * Shows how k2::Bridge maps to an OpenZL Codec interface.
 * Derive from the appropriate OpenZL base class (e.g. zl::StreamCodec)
 * and implement the required virtual methods.
 */
struct K2CodecStub {
    Bridge bridge;

    explicit K2CodecStub(const std::string& py_dir,
                         const std::string& onnx_path = "")
        : bridge(py_dir, onnx_path) {}

    // Called once when a new compression stream starts
    void on_stream_start(const uint8_t* header, size_t header_len) {
        std::string detected = bridge.prepare(header, header_len);
        (void)detected;
    }

    // Called per chunk / control-point
    std::vector<uint8_t> compress_chunk(const uint8_t* data, size_t len) {
        return bridge.compress(data, len);
    }

    std::string get_stats() { return bridge.stats(); }
};

}  // namespace k2

#endif  // __cplusplus


// ---------------------------------------------------------------------------
// Implementation of the C API (include in exactly one .cpp translation unit)
// ---------------------------------------------------------------------------

#ifdef K2_IMPLEMENTATION
#ifdef __cplusplus

#include <cstdlib>

struct K2Handle {
    k2::Bridge* bridge    = nullptr;
    char*       last_stats = nullptr;
};

// Note: k2_create and k2_destroy are defined in k2_bridge.cpp
// because they manage the Python interpreter lifetime.

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

K2_API void k2_free_str(const char* /*s*/) { /* owned by handle */ }

}  // extern "C"

#endif  // __cplusplus
#endif  // K2_IMPLEMENTATION
