/**
 * k2/src/cpp/k2cli.cpp
 *
 * K2 command-line tool.
 *
 * Usage:
 *   k2cli compress   <input> <output.zl>
 *   k2cli decompress <input.zl> <output>
 *   k2cli roundtrip  <input>          (compress + decompress + diff)
 *   k2cli stats      <input>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "k2_bridge.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "k2cli: cannot open " << path << "\n";
        std::exit(1);
    }
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static void write_file(const std::string& path,
                       const uint8_t* data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "k2cli: cannot write " << path << "\n";
        std::exit(1);
    }
    f.write(reinterpret_cast<const char*>(data), len);
}

static K2Handle* make_handle() {
    const char* py_dir_env  = std::getenv("K2_PYTHON_DIR");
    const char* onnx_env    = std::getenv("K2_ONNX_MODEL");
    std::string py_dir      = py_dir_env ? py_dir_env : K2_PYTHON_MODULE_DIR;
    std::string onnx        = onnx_env   ? onnx_env   : "";

    K2Handle* h = k2_create(onnx.empty() ? nullptr : onnx.c_str(), 1.0, 0.15);
    if (!h) {
        std::cerr << "k2cli: failed to create pipeline\n";
        std::exit(1);
    }
    return h;
}

static void usage() {
    std::cerr <<
        "Usage:\n"
        "  k2cli compress   <input> <output.zl>\n"
        "  k2cli decompress <input.zl> <output>\n"
        "  k2cli roundtrip  <input>\n"
        "  k2cli stats      <input>\n"
        "\n"
        "Environment:\n"
        "  K2_PYTHON_DIR   override Python module directory\n"
        "  K2_ONNX_MODEL   path to ONNX classifier model (optional)\n";
    std::exit(1);
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

static int cmd_compress(const std::string& in_path,
                        const std::string& out_path) {
    auto data = read_file(in_path);
    std::cout << "K2 compress: " << in_path
              << " (" << data.size() << " bytes)\n";

    K2Handle* h = make_handle();

    size_t sample_len = std::min(data.size(), size_t(65536));
    if (k2_prepare(h, data.data(), sample_len) != 0) {
        std::cerr << "k2cli: prepare failed\n";
        k2_destroy(h); return 1;
    }

    std::vector<uint8_t> out(ZL_compressBound(data.size()) + 4096);
    size_t out_len = 0;
    int rc = k2_compress(h, data.data(), data.size(),
                          out.data(), out.size(), &out_len);
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

static int cmd_decompress(const std::string& in_path,
                          const std::string& out_path) {
    auto data = read_file(in_path);
    std::cout << "K2 decompress: " << in_path
              << " (" << data.size() << " bytes)\n";

    K2Handle* h = make_handle();

    // For decompress we don't need prepare() — OpenZL frame is self-describing
    std::vector<uint8_t> out(data.size() * 20 + 4096);  // generous bound
    size_t out_len = 0;
    int rc = k2_decompress(h, data.data(), data.size(),
                            out.data(), out.size(), &out_len);
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
    std::cout << "K2 roundtrip: " << in_path
              << " (" << original.size() << " bytes)\n";

    K2Handle* h = make_handle();

    // Compress
    size_t sample_len = std::min(original.size(), size_t(65536));
    k2_prepare(h, original.data(), sample_len);

    std::vector<uint8_t> compressed(
        ZL_compressBound(original.size()) + 4096);
    size_t comp_len = 0;
    int rc = k2_compress(h, original.data(), original.size(),
                          compressed.data(), compressed.size(), &comp_len);
    if (rc != 0) {
        std::cerr << "  compress failed (rc=" << rc << ")\n";
        k2_destroy(h); return 1;
    }

    double ratio = static_cast<double>(original.size()) / comp_len;
    std::cout << "  compressed: " << comp_len << " bytes ("
              << std::fixed << std::setprecision(2) << ratio << "x)\n";

    // Decompress
    std::vector<uint8_t> restored(original.size() * 2 + 4096);
    size_t rest_len = 0;
    rc = k2_decompress(h, compressed.data(), comp_len,
                        restored.data(), restored.size(), &rest_len);
    if (rc != 0) {
        std::cerr << "  decompress failed (rc=" << rc << ")\n";
        k2_destroy(h); return 1;
    }

    // Verify
    if (rest_len == original.size() &&
        std::memcmp(restored.data(), original.data(), rest_len) == 0) {
        std::cout << "  ✓ MATCH — roundtrip verified (" << rest_len << " bytes)\n";
    } else {
        std::cerr << "  ✗ MISMATCH — restored " << rest_len
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
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) usage();
    std::string cmd = argv[1];

    if (cmd == "compress" && argc == 4)
        return cmd_compress(argv[2], argv[3]);
    if (cmd == "decompress" && argc == 4)
        return cmd_decompress(argv[2], argv[3]);
    if (cmd == "roundtrip" && argc == 3)
        return cmd_roundtrip(argv[2]);
    if (cmd == "stats" && argc == 3)
        return cmd_stats(argv[2]);

    usage();
    return 1;
}
