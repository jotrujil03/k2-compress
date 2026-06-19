/**
 * k2/src/cpp/k2cli.cpp
 *
 * K2 command-line tool.
 *
 * Usage:
 *   k2cli compress   <input> <output.k2>
 *   k2cli decompress <input.k2> <output>
 *   k2cli roundtrip  <input>
 *   k2cli stats      <input>
 *
 * Output format: K2 frame (not OpenZL .zl).
 * Use k2cli decompress to restore; zli decompress is not compatible.
 */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "k2_bridge.h"   // pulls in asdp/asdp.h for asdp_compress_bound()

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "k2cli: cannot open " << path << "\n"; std::exit(1); }
    auto sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static void write_file(const std::string& path, const uint8_t* data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "k2cli: cannot write " << path << "\n"; std::exit(1); }
    f.write(reinterpret_cast<const char*>(data), len);
}

static K2Handle* make_handle() {
    const char* onnx_env = std::getenv("K2_ONNX_MODEL");
    std::string onnx     = onnx_env ? onnx_env : "";
    K2Handle* h = k2_create(onnx.empty() ? nullptr : onnx.c_str(), 1.0, 0.15);
    if (!h) { std::cerr << "k2cli: failed to create pipeline\n"; std::exit(1); }
    return h;
}

static void usage() {
    std::cerr <<
        "Usage:\n"
        "  k2cli compress   <input> <output.k2>\n"
        "  k2cli decompress <input.k2> <output>\n"
        "  k2cli roundtrip  <input>\n"
        "  k2cli parallel   <input> <n_threads>\n"
        "  k2cli stats      <input>\n"
        "\n"
        "Environment:\n"
        "  K2_ONNX_MODEL   path to ONNX classifier model (optional)\n"
        "\n"
        "Output format: K2 native frame.\n"
        "Not compatible with zli decompress.\n";
    std::exit(1);
}

// Output buffer bound: ASDP guarantees the entropy backend never exceeds
// asdp_compress_bound(n); add K2 frame + txhdr overhead headroom.
static size_t output_bound(size_t input_size) {
    return asdp_compress_bound(input_size) + 256;
}

static int cmd_compress(const std::string& in_path, const std::string& out_path) {
    auto data = read_file(in_path);
    std::cout << "K2 compress: " << in_path << " (" << data.size() << " bytes)\n";

    K2Handle* h = make_handle();
    size_t sample_len = std::min(data.size(), size_t(65536));
    if (k2_prepare(h, data.data(), sample_len) != 0) {
        std::cerr << "k2cli: prepare failed\n"; k2_destroy(h); return 1;
    }

    std::vector<uint8_t> out(output_bound(data.size()));
    size_t out_len = 0;
    int rc = k2_compress(h, data.data(), data.size(), out.data(), out.size(), &out_len);
    if (rc != 0) {
        std::cerr << "k2cli: compress failed (rc=" << rc << ")\n";
        k2_destroy(h); return 1;
    }

    write_file(out_path, out.data(), out_len);
    double ratio = static_cast<double>(data.size()) / out_len;
    std::cout << "  → " << out_path << " (" << out_len << " bytes, "
              << std::fixed << std::setprecision(2) << ratio << "x)\n";
    std::cout << k2_stats(h) << "\n";
    k2_destroy(h);
    return 0;
}

static int cmd_decompress(const std::string& in_path, const std::string& out_path) {
    auto data = read_file(in_path);
    std::cout << "K2 decompress: " << in_path << " (" << data.size() << " bytes)\n";

    K2Handle* h = make_handle();
    // K2 frame carries orig_size — allocate generously.
    std::vector<uint8_t> out(data.size() * 20 + 4096);
    size_t out_len = 0;
    int rc = k2_decompress(h, data.data(), data.size(), out.data(), out.size(), &out_len);
    if (rc != 0) {
        std::cerr << "k2cli: decompress failed (rc=" << rc << ")\n";
        k2_destroy(h); return 1;
    }

    write_file(out_path, out.data(), out_len);
    std::cout << "  → " << out_path << " (" << out_len << " bytes)\n";
    k2_destroy(h);
    return 0;
}

static int cmd_roundtrip(const std::string& in_path) {
    auto original = read_file(in_path);
    std::cout << "K2 roundtrip: " << in_path << " (" << original.size() << " bytes)\n";

    K2Handle* h = make_handle();
    size_t sample_len = std::min(original.size(), size_t(65536));
    k2_prepare(h, original.data(), sample_len);

    std::vector<uint8_t> compressed(output_bound(original.size()));
    size_t comp_len = 0;
    int rc = k2_compress(h, original.data(), original.size(),
                          compressed.data(), compressed.size(), &comp_len);
    if (rc != 0) {
        std::cerr << "  compress failed (rc=" << rc << ")\n"; k2_destroy(h); return 1;
    }
    double ratio = static_cast<double>(original.size()) / comp_len;
    std::cout << "  compressed: " << comp_len << " bytes ("
              << std::fixed << std::setprecision(2) << ratio << "x)\n";

    std::vector<uint8_t> restored(original.size() * 2 + 4096);
    size_t rest_len = 0;
    rc = k2_decompress(h, compressed.data(), comp_len,
                        restored.data(), restored.size(), &rest_len);
    if (rc != 0) {
        std::cerr << "  decompress failed (rc=" << rc << ")\n"; k2_destroy(h); return 1;
    }

    if (rest_len == original.size() &&
        std::memcmp(restored.data(), original.data(), rest_len) == 0) {
        std::cout << "  \xe2\x9c\x93 MATCH \xe2\x80\x94 roundtrip verified (" << rest_len << " bytes)\n";
    } else {
        std::cerr << "  \xe2\x9c\x97 MISMATCH \xe2\x80\x94 restored " << rest_len
                  << " bytes, expected " << original.size() << "\n";
        k2_destroy(h); return 1;
    }

    std::cout << k2_stats(h) << "\n";
    k2_destroy(h);
    return 0;
}

static int cmd_stats(const std::string& in_path) {
    auto data = read_file(in_path);
    K2Handle* h = make_handle();
    size_t sample_len = std::min(data.size(), size_t(65536));
    k2_prepare(h, data.data(), sample_len);
    std::cout << k2_stats(h) << "\n";
    k2_destroy(h);
    return 0;
}

// ---------------------------------------------------------------------------
// Parallel roundtrip — exercises the shared-handle threading model (fix C).
//
// Correct pattern: create ONE K2Handle on the main thread, prepare it, then
// share it across worker threads.  Each worker calls k2_compress/k2_decompress
// on the same handle.  This is safe because:
//   - the interpreter is initialised once with the GIL released at rest
//     (k2_bridge.cpp), so no worker deadlocks acquiring the GIL;
//   - the Python pipeline calls serialise briefly on the GIL, while the heavy
//     ASDP entropy work runs GIL-free, giving real parallel throughput.
//
// Anti-pattern (the original deadlock): each worker calling k2_create() →
// concurrent Py_Initialize() with the GIL held by the first thread.
// ---------------------------------------------------------------------------
static int cmd_parallel(const std::string& in_path, int n_threads) {
    auto original = read_file(in_path);
    std::cout << "K2 parallel roundtrip: " << in_path << " ("
              << original.size() << " bytes) x" << n_threads << " threads\n";

    K2Handle* h = make_handle();                       // ONE shared handle
    size_t sample_len = std::min(original.size(), size_t(65536));
    k2_prepare(h, original.data(), sample_len);

    std::atomic<int> failures{0};
    auto worker = [&](int tid) {
        std::vector<uint8_t> comp(output_bound(original.size()));
        size_t comp_len = 0;
        if (k2_compress(h, original.data(), original.size(),
                        comp.data(), comp.size(), &comp_len) != 0) {
            ++failures; return;
        }
        std::vector<uint8_t> rest(original.size() * 2 + 4096);
        size_t rest_len = 0;
        if (k2_decompress(h, comp.data(), comp_len,
                          rest.data(), rest.size(), &rest_len) != 0) {
            ++failures; return;
        }
        if (rest_len != original.size() ||
            std::memcmp(rest.data(), original.data(), rest_len) != 0) {
            ++failures;
            std::cerr << "  thread " << tid << ": MISMATCH\n";
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    const int f = failures.load();
    if (f == 0) std::cout << "  \xe2\x9c\x93 all " << n_threads << " threads verified\n";
    else        std::cerr << "  \xe2\x9c\x97 " << f << " thread(s) failed\n";

    std::cout << k2_stats(h) << "\n";
    k2_destroy(h);
    return f == 0 ? 0 : 1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) usage();
    std::string cmd = argv[1];
    if (cmd == "compress"   && argc == 4) return cmd_compress(argv[2], argv[3]);
    if (cmd == "decompress" && argc == 4) return cmd_decompress(argv[2], argv[3]);
    if (cmd == "roundtrip"  && argc == 3) return cmd_roundtrip(argv[2]);
    if (cmd == "parallel"   && argc == 4) return cmd_parallel(argv[2], std::atoi(argv[3]));
    if (cmd == "stats"      && argc == 3) return cmd_stats(argv[2]);
    usage(); return 1;
}
