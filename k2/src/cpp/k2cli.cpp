/**
 * k2/src/cpp/k2cli.cpp
 *
 * K2 command-line tool.
 *
 * Usage:
 *   k2cli compress   <input> <output>
 *   k2cli decompress <input> <output>
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

static void usage() {
    std::cerr <<
        "Usage:\n"
        "  k2cli compress   <input> <output>\n"
        "  k2cli stats      <input>\n"
        "\n"
        "Environment:\n"
        "  K2_PYTHON_DIR   override Python module directory\n"
        "  K2_ONNX_MODEL   path to ONNX classifier model (optional)\n";
    std::exit(1);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) usage();

    std::string cmd = argv[1];

    // Python module directory: compile-time default or env override
    const char* py_dir_env = std::getenv("K2_PYTHON_DIR");
    std::string py_dir = py_dir_env ? py_dir_env : K2_PYTHON_MODULE_DIR;

    const char* onnx_env = std::getenv("K2_ONNX_MODEL");
    std::string onnx = onnx_env ? onnx_env : "";

    if (cmd == "compress") {
        if (argc < 4) usage();
        std::string in_path  = argv[2];
        std::string out_path = argv[3];

        auto data = read_file(in_path);
        std::cout << "K2 compress: " << in_path
                  << " (" << data.size() << " bytes)\n";

        K2Handle* h = k2_create(onnx.empty() ? nullptr : onnx.c_str(),
                                 1.0, 0.15);
        if (!h) {
            std::cerr << "k2cli: failed to create pipeline\n";
            return 1;
        }

        // Use first 64KB as sample for structure discovery
        size_t sample_len = std::min(data.size(), size_t(65536));
        if (k2_prepare(h, data.data(), sample_len) != 0) {
            std::cerr << "k2cli: prepare failed\n";
            k2_destroy(h);
            return 1;
        }

        std::vector<uint8_t> out(data.size() * 2 + 4096);
        size_t out_len = 0;
        int rc = k2_compress(h, data.data(), data.size(),
                              out.data(), out.size(), &out_len);
        if (rc != 0) {
            std::cerr << "k2cli: compress failed (rc=" << rc << ")\n";
            k2_destroy(h);
            return 1;
        }

        write_file(out_path, out.data(), out_len);

        double ratio = static_cast<double>(data.size()) / out_len;
        std::cout << "  → " << out_path << " (" << out_len << " bytes, "
                  << std::fixed << std::setprecision(2) << ratio << "x)\n";
        std::cout << k2_stats(h) << "\n";

        k2_destroy(h);

    } else if (cmd == "stats") {
        if (argc < 3) usage();
        auto data = read_file(argv[2]);

        K2Handle* h = k2_create(onnx.empty() ? nullptr : onnx.c_str(),
                                 1.0, 0.15);
        if (!h) { std::cerr << "k2cli: failed to create pipeline\n"; return 1; }

        size_t sample_len = std::min(data.size(), size_t(65536));
        k2_prepare(h, data.data(), sample_len);
        std::cout << k2_stats(h) << "\n";
        k2_destroy(h);

    } else {
        usage();
    }

    return 0;
}
