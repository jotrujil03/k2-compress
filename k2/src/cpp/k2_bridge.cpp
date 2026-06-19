/**
 * k2/src/cpp/k2_bridge.cpp
 *
 * Uses raw CPython API for interpreter init (safe from shared libraries).
 * pybind11::scoped_interpreter must NOT be used here — it conflicts when
 * loaded as a shared library into an existing process.
 *
 * Threading (fix C)
 * -----------------
 * Py_Initialize() leaves the calling thread holding the GIL.  If we never
 * release it, any other thread that calls into Python via
 * py::gil_scoped_acquire deadlocks, and concurrent k2_create() calls race on
 * interpreter initialisation.  We therefore:
 *   1. Serialise interpreter init/teardown with a mutex.
 *   2. Initialise the interpreter exactly once, then immediately release the
 *      GIL with PyEval_SaveThread().  After that every thread — including the
 *      one that did the init — acquires the GIL per call via
 *      py::gil_scoped_acquire (already done inside k2::Bridge methods).
 *
 * With the GIL released at rest, a single shared Bridge can be driven from
 * many worker threads: the Python calls serialise briefly on the GIL while
 * the heavy C++ entropy work (ASDP) runs GIL-free.  See k2cli.cpp's
 * parallel_compress for the recommended shared-handle pattern.
 */

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <mutex>

#ifndef K2_IMPLEMENTATION
#define K2_IMPLEMENTATION
#endif
#include "k2_bridge.h"

// ---------------------------------------------------------------------------
// Interpreter lifetime — reference counted, mutex-guarded, GIL released at rest
// ---------------------------------------------------------------------------

namespace {

std::mutex     g_interp_mtx;
int            g_refcount    = 0;
PyThreadState* g_main_tstate = nullptr;   // saved main thread state (GIL holder)

void interp_acquire() {
    std::lock_guard<std::mutex> lk(g_interp_mtx);
    if (g_refcount++ == 0) {
        if (!Py_IsInitialized()) {
            Py_Initialize();                       // this thread now holds GIL
            // Release the GIL and stash the thread state so *any* thread can
            // acquire it later via py::gil_scoped_acquire.
            g_main_tstate = PyEval_SaveThread();
        }
    }
}

void interp_release() {
    std::lock_guard<std::mutex> lk(g_interp_mtx);
    if (g_refcount > 0 && --g_refcount == 0) {
        // Do NOT Py_Finalize() from a shared library — it is unsafe to tear
        // down and re-init CPython in-process.  Leave the interpreter resident
        // with the GIL released; the OS reclaims everything on process exit.
        // g_main_tstate is intentionally left saved.
        (void)g_main_tstate;
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
        // Bridge ctor acquires the GIL per py::gil_scoped_acquire internally.
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
    delete h->bridge;     // Bridge dtor releases its py::object under the GIL
    free(h->last_stats);
    delete h;
    interp_release();
}

}  // extern "C"
