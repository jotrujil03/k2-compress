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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "k2_bridge.h"

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
        "  k2cli stats      <input>\n"
        "\n"
        "Environment:\n"
        "  K2_ONNX_MODEL   path to ONNX classifier model (optional)\n"
        "\n"
        "Output format: K2 native frame.\n"
        "Not compatible with zli decompress.\n";
    std::exit(1);
}

// Output buffer bound: K2 frame overhead (≤128 bytes) + worst-case entropy
// expansion (incompressible data ≈ 1.001× input).  ZL_compressBound is still
// available and is always >= input size, so it remains a safe upper bound.
static size_t output_bound(size_t input_size) {
    return ZL_compressBound(input_size) + 256;
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

int main(int argc, char* argv[]) {
    if (argc < 2) usage();
    std::string cmd = argv[1];
    if (cmd == "compress"   && argc == 4) return cmd_compress(argv[2], argv[3]);
    if (cmd == "decompress" && argc == 4) return cmd_decompress(argv[2], argv[3]);
    if (cmd == "roundtrip"  && argc == 3) return cmd_roundtrip(argv[2]);
    if (cmd == "stats"      && argc == 3) return cmd_stats(argv[2]);
    usage(); return 1;
}
