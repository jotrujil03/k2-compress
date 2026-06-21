/**
 * k2/include/k2_bridge.h
 *
 * C++ Bridge: K2 Neural Pipeline
 * --------------------------------
 * K2Pipeline (Python) owns the full compression decision.  The C++ bridge
 * handles entropy (ASDP-LH, backend 0x04 — the sole live backend; 0x01–0x03
 * are retired):
 *   - calls asdp_compress() on the (pre-transform) payload, then reseals
 *     the K2 frame around the ASDP output.  On decompress it calls
 *     asdp_decompress() first, reseals, then lets Python invert the
 *     structural transforms.
 *
 * K2 Frame Format (owned by Python, parsed by C++ for dispatch only):
 *   [0..3]   magic  b'K2\xf7\x01'
 *   [4]      version  0x01
 *   [5]      backend  0x04=ASDP  (0x01/02/03 retired)
 *   [6]      flags
 *   [7]      reserved
 *   [8..15]  orig_size  uint64 LE
 *   [16..17] txhdr_len  uint16 LE
 *   [18..]   txhdr + payload
 *
 * Compress path:
 *   compress_full(data) -> K2 frame
 *   payload = asdp_compress(frame.payload)     // GIL released here
 *   frame   = reseal_frame(frame, payload)
 *   return frame
 *
 * Decompress path:
 *   inner    = asdp_decompress(frame.payload)  // GIL released here
 *   resealed = reseal_frame(frame, inner)
 *   decompress_full(resealed) -> original bytes // Python inverts transforms
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

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

/**
 * Extended constructor: like k2_create, but also accepts a path to a
 * trained TransformGainPredictor ONNX model. Pass nullptr/"" for
 * gain_predictor_path to get exactly k2_create's behavior (the gain
 * guard then falls back to the built-in zlib-ratio heuristic). Added as
 * a separate function rather than a 4th parameter on k2_create so every
 * existing 3-argument caller (and the compiled ABI) keeps working
 * unchanged.
 */
K2_API K2Handle* k2_create_ex(
    const char* onnx_model_path,
    const char* gain_predictor_path,
    double      exploration,
    double      latency_weight
);

K2_API int k2_prepare(
    K2Handle*      handle,
    const uint8_t* sample,
    size_t         sample_len
);

/** Compress src -> K2 frame in dst. */
K2_API int k2_compress(
    K2Handle*      handle,
    const uint8_t* src,
    size_t         src_len,
    uint8_t*       dst,
    size_t         dst_cap,
    size_t*        out_len
);

/** Decompress a K2 frame produced by k2_compress. */
K2_API int k2_decompress(
    K2Handle*      handle,
    const uint8_t* src,
    size_t         src_len,
    uint8_t*       dst,
    size_t         dst_cap,
    size_t*        out_len
);

/** Feed final compressed size back to the bandit. */
K2_API void k2_record_result(
    K2Handle* handle,
    size_t    input_size,
    size_t    output_size,
    double    elapsed_ms
);

K2_API const char* k2_stats(K2Handle* handle);
K2_API void        k2_free_str(const char* s);
K2_API void        k2_destroy(K2Handle* handle);

#ifdef __cplusplus
}
#endif


// ---------------------------------------------------------------------------
// C++ RAII wrapper
// ---------------------------------------------------------------------------

#ifdef __cplusplus

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "asdp/asdp.h"   // ASDP-LH C API (replaces OpenZL)

namespace py = pybind11;
using namespace pybind11::literals;

namespace k2 {

// ---------------------------------------------------------------------------
// K2 frame constants (must match Python adaptive_optimizer.py)
// ---------------------------------------------------------------------------

static constexpr uint8_t  K2_FRAME_MAGIC[4] = {
    uint8_t('K'), uint8_t('2'), uint8_t(0xf7), uint8_t(0x01)
};
static constexpr uint8_t  K2_FRAME_VERSION  = 0x01;
static constexpr uint8_t  K2_BACKEND_OPENZL = 0x01;  // retired; kept for decode guard
static constexpr uint8_t  K2_BACKEND_ZSTD   = 0x02;  // retired — decode guard only
static constexpr uint8_t  K2_BACKEND_ZLIB   = 0x03;  // retired — decode guard only
static constexpr uint8_t  K2_BACKEND_ASDP   = 0x04;
static constexpr size_t   K2_FRAME_HDR_SIZE = 18;  // through txhdr_len field

struct K2FrameHeader {
    uint8_t  magic[4];
    uint8_t  version;
    uint8_t  backend;
    uint8_t  flags;
    uint8_t  reserved;
    uint64_t orig_size;
    uint16_t txhdr_len;
} __attribute__((packed));

static_assert(sizeof(K2FrameHeader) == K2_FRAME_HDR_SIZE,
              "K2FrameHeader size mismatch");

static uint8_t frame_backend(const uint8_t* data, size_t len) {
    if (len < K2_FRAME_HDR_SIZE) return 0;
    const auto* h = reinterpret_cast<const K2FrameHeader*>(data);
    if (std::memcmp(h->magic, K2_FRAME_MAGIC, 4) != 0) return 0;
    return h->backend;
}

static uint64_t frame_orig_size(const uint8_t* data, size_t len) {
    if (len < K2_FRAME_HDR_SIZE) return 0;
    const auto* h = reinterpret_cast<const K2FrameHeader*>(data);
    return h->orig_size;
}

static uint16_t frame_txhdr_len(const uint8_t* data, size_t len) {
    if (len < K2_FRAME_HDR_SIZE) return 0;
    const auto* h = reinterpret_cast<const K2FrameHeader*>(data);
    return h->txhdr_len;
}

// Payload starts after fixed header + txhdr
static size_t frame_payload_offset(const uint8_t* data, size_t len) {
    return K2_FRAME_HDR_SIZE + frame_txhdr_len(data, len);
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
        double latency_weight = 0.15,
        int    asdp_level     = 3,
        const std::string& gain_predictor_path = ""
    ) : _asdp_level(asdp_level)
    {
        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, python_module_dir);
        py::module_ mod = py::module_::import("adaptive_optimizer");
        py::object  cls = mod.attr("K2Pipeline");
        py::object onnx = onnx_model_path.empty()
            ? py::none() : py::cast(onnx_model_path);
        py::object gain = gain_predictor_path.empty()
            ? py::none() : py::cast(gain_predictor_path);
        _pipeline = cls(
            "onnx_model_path"_a     = onnx,
            "gain_predictor_path"_a = gain,
            "exploration"_a         = exploration,
            "latency_weight"_a      = latency_weight
        );
    }

    // _pipeline is a py::object; its decref needs the GIL, which is released
    // at rest (see k2_bridge.cpp).  Reacquire before destroying it.
    ~Bridge() {
        py::gil_scoped_acquire gil;
        _pipeline = py::object();
    }

    std::string prepare(const uint8_t* data, size_t len) {
        py::gil_scoped_acquire gil;
        py::bytes sample(reinterpret_cast<const char*>(data), len);
        py::object hint = _pipeline.attr("prepare")(sample);
        return hint.attr("data_class").attr("name").cast<std::string>();
    }

    /**
     * Compress data -> K2 frame.
     *
     * 1. Call Python compress_full(data) -> K2 frame bytes.
     * 2. Inspect backend byte:
     *    - ZSTD/ZLIB: frame is complete, return as-is.
     *    - ASDP: extract payload (pre-transform bytes), run asdp_compress()
     *            with the GIL released, then call Python reseal_frame().
     */
    std::vector<uint8_t> compress(const uint8_t* data, size_t len) {
        auto t0 = std::chrono::steady_clock::now();

        // Step 1: Python decides backend + transforms, returns K2 frame.
        std::vector<uint8_t> frame = call_compress_full(data, len);

        uint8_t backend = frame_backend(frame.data(), frame.size());

        if (backend == K2_BACKEND_ASDP) {
            // Step 2a: entropy-compress the payload with ASDP-LH.
            // call_compress_full() has already released the GIL, so this
            // pure-C++ work runs concurrently across worker threads.
            const size_t poff   = frame_payload_offset(frame.data(), frame.size());
            const uint8_t* pl   = frame.data() + poff;
            const size_t   plen = frame.size() - poff;

            std::string asdp_out = asdp_compress_payload(pl, plen);

            // Step 2b: reseal frame with the compressed payload.
            frame = call_reseal_frame(frame, asdp_out);
        }
        // For ZSTD/ZLIB backends frame is already final — nothing to do.

        auto t1 = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        record_result(len, frame.size(), elapsed_ms);

        return frame;
    }

    /**
     * Decompress a K2 frame -> original bytes.
     */
    std::vector<uint8_t> decompress(const uint8_t* data, size_t len) {
        uint8_t backend = frame_backend(data, len);

        if (backend == K2_BACKEND_ASDP) {
            // Step 1: ASDP entropy decompress (GIL-free).
            const size_t poff   = frame_payload_offset(data, len);
            const uint8_t* pl   = data + poff;
            const size_t   plen = len - poff;

            std::string inner = asdp_decompress_payload(pl, plen);

            // Step 2: reseal frame with decompressed payload.
            std::vector<uint8_t> resealed = call_reseal_frame(
                std::vector<uint8_t>(data, data + len), inner);

            // Step 3: Python inverts the structural transforms.
            return call_decompress_full(resealed.data(), resealed.size(),
                                        frame_orig_size(data, len));
        }

        // ZSTD/ZLIB: Python does everything.
        return call_decompress_full(data, len, frame_orig_size(data, len));
    }

    std::vector<uint8_t> compress(const std::vector<uint8_t>& v) {
        return compress(v.data(), v.size());
    }
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& v) {
        return decompress(v.data(), v.size());
    }

    void record_result(size_t input_size, size_t output_size, double elapsed_ms) {
        py::gil_scoped_acquire gil;
        try {
            _pipeline.attr("update_final_score")(
                py::str(""),
                static_cast<uint64_t>(input_size),
                static_cast<uint64_t>(output_size),
                elapsed_ms
            );
        } catch (const py::error_already_set& e) {
            fprintf(stderr, "k2_bridge: update_final_score failed: %s\n", e.what());
            PyErr_Clear();
        }
    }

    std::string stats() {
        py::gil_scoped_acquire gil;
        py::dict d = _pipeline.attr("stats")();
        return py::module_::import("json").attr("dumps")(d).cast<std::string>();
    }

private:
    py::object _pipeline;
    int        _asdp_level;

    // ------------------------------------------------------------------
    // ASDP entropy backend (pure C++, no GIL).  A context is created per
    // call: asdp_ctx_t is not thread-safe, and a shared Bridge may be driven
    // from multiple threads, so per-call contexts avoid any shared state.
    // ------------------------------------------------------------------

    std::string asdp_compress_payload(const uint8_t* pl, size_t plen) {
        asdp_config_t cfg = asdp_default_config();
        cfg.level = _asdp_level;
        asdp_ctx_t* ctx = asdp_create(&cfg);
        if (!ctx) throw std::runtime_error("asdp_create failed");

        const size_t bound = asdp_compress_bound(plen);
        std::vector<uint8_t> out(bound);
        size_t out_len = 0;
        const int rc = asdp_compress(ctx, pl, plen, out.data(), bound, &out_len);
        asdp_destroy(ctx);
        if (rc != ASDP_OK)
            throw std::runtime_error(std::string("asdp_compress: ") + asdp_error_str(rc));
        return std::string(reinterpret_cast<char*>(out.data()), out_len);
    }

    std::string asdp_decompress_payload(const uint8_t* pl, size_t plen) {
        const uint64_t orig = asdp_frame_orig_size(pl, plen);
        asdp_config_t cfg = asdp_default_config();
        cfg.level = _asdp_level;
        asdp_ctx_t* ctx = asdp_create(&cfg);
        if (!ctx) throw std::runtime_error("asdp_create failed");

        std::vector<uint8_t> out(orig ? orig : plen * 4 + 4096);
        size_t out_len = 0;
        const int rc = asdp_decompress(ctx, pl, plen,
                                       out.data(), out.size(), &out_len);
        asdp_destroy(ctx);
        if (rc != ASDP_OK)
            throw std::runtime_error(std::string("asdp_decompress: ") + asdp_error_str(rc));
        return std::string(reinterpret_cast<char*>(out.data()), out_len);
    }

    // ------------------------------------------------------------------
    // Python call helpers (each acquires the GIL for its scope only)
    // ------------------------------------------------------------------

    std::vector<uint8_t> call_compress_full(const uint8_t* data, size_t len) {
        py::gil_scoped_acquire gil;
        py::bytes input(reinterpret_cast<const char*>(data), len);
        try {
            py::bytes out = _pipeline.attr("compress_full")(input);
            std::string s = out.cast<std::string>();
            return std::vector<uint8_t>(s.begin(), s.end());
        } catch (const py::error_already_set& e) {
            fprintf(stderr, "k2_bridge: compress_full failed: %s\n", e.what());
            PyErr_Clear();
            // Fallback: return raw bytes (decompress side handles legacy path).
            return std::vector<uint8_t>(data, data + len);
        }
    }

    std::vector<uint8_t> call_reseal_frame(
        const std::vector<uint8_t>& frame,
        const std::string& new_payload)
    {
        py::gil_scoped_acquire gil;
        py::bytes py_frame(reinterpret_cast<const char*>(frame.data()), frame.size());
        py::bytes py_payload(new_payload.data(), new_payload.size());
        try {
            py::bytes out = _pipeline.attr("reseal_frame")(py_frame, py_payload);
            std::string s = out.cast<std::string>();
            return std::vector<uint8_t>(s.begin(), s.end());
        } catch (const py::error_already_set& e) {
            fprintf(stderr, "k2_bridge: reseal_frame failed: %s\n", e.what());
            PyErr_Clear();
            return frame;
        }
    }

    std::vector<uint8_t> call_decompress_full(
        const uint8_t* data, size_t len, uint64_t orig_size)
    {
        py::gil_scoped_acquire gil;
        py::bytes input(reinterpret_cast<const char*>(data), len);
        try {
            py::bytes out = _pipeline.attr("decompress_full")(
                input, static_cast<uint64_t>(orig_size));
            std::string s = out.cast<std::string>();
            return std::vector<uint8_t>(s.begin(), s.end());
        } catch (const py::error_already_set& e) {
            fprintf(stderr, "k2_bridge: decompress_full failed: %s\n", e.what());
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

extern "C" {

K2_API int k2_prepare(K2Handle* h, const uint8_t* sample, size_t len) {
    if (!h || !h->bridge) return -1;
    try { h->bridge->prepare(sample, len); return 0; }
    catch (...) { return -2; }
}

K2_API int k2_compress(K2Handle* h,
                        const uint8_t* src, size_t src_len,
                        uint8_t* dst, size_t dst_cap, size_t* out_len) {
    if (!h || !h->bridge || !dst || !out_len) return -1;
    try {
        auto out = h->bridge->compress(src, src_len);
        if (out.size() > dst_cap) return -3;
        std::memcpy(dst, out.data(), out.size());
        *out_len = out.size();
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
        auto out = h->bridge->decompress(src, src_len);
        if (out.size() > dst_cap) return -3;
        std::memcpy(dst, out.data(), out.size());
        *out_len = out.size();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "k2_decompress error: %s\n", e.what());
        return -2;
    } catch (...) { return -2; }
}

K2_API void k2_record_result(K2Handle* h,
                              size_t input_size,
                              size_t output_size,
                              double elapsed_ms) {
    if (!h || !h->bridge) return;
    h->bridge->record_result(input_size, output_size, elapsed_ms);
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
