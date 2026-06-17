/**
 * k2/src/cpp/k2_bridge.cpp
 *
 * Uses raw CPython API for interpreter init (safe from shared libraries).
 * pybind11::scoped_interpreter must NOT be used here — it conflicts when
 * loaded as a shared library into an existing process.
 */

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#ifndef K2_IMPLEMENTATION
#define K2_IMPLEMENTATION
#endif
#include "k2_bridge.h"

// ---------------------------------------------------------------------------
// Interpreter lifetime — reference counted, CPython API
// ---------------------------------------------------------------------------

namespace {

static int g_refcount = 0;

void interp_acquire() {
    if (g_refcount++ == 0) {
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
    }
}

void interp_release() {
    if (--g_refcount == 0) {
        // Don't call Py_Finalize() — unsafe with shared libraries
        // The OS will clean up on process exit
    }
}

}  // namespace

extern "C" {

K2_API K2Handle* k2_create(const char* onnx_model_path,
                            double exploration,
                            double latency_weight) {
    try {
        interp_acquire();
        auto* h = new K2Handle();
        std::string onnx = onnx_model_path ? onnx_model_path : "";
        h->bridge = new k2::Bridge(K2_PYTHON_MODULE_DIR, onnx, exploration, latency_weight);
        return h;
    } catch (const std::exception& e) {
        fprintf(stderr, "k2_create failed: %s\n", e.what());
        interp_release();
        return nullptr;
    } catch (...) {
        fprintf(stderr, "k2_create failed: unknown exception\n");
        interp_release();
        return nullptr;
    }
}

K2_API void k2_destroy(K2Handle* h) {
    if (!h) return;
    delete h->bridge;
    free(h->last_stats);
    delete h;
    interp_release();
}

}  // extern "C"
