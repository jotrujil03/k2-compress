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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "k2_bridge.h"   // pulls in asdp/asdp.h for asdp_compress_bound()
#include "k2archive.h"   // directory archive (manifest + multi-volume)

namespace fs = std::filesystem;

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
    const char* gain_env = std::getenv("K2_GAIN_MODEL");
    std::string gain     = gain_env ? gain_env : "";
    K2Handle* h = k2_create_ex(
        onnx.empty() ? nullptr : onnx.c_str(),
        gain.empty() ? nullptr : gain.c_str(),
        1.0, 0.15);
    if (!h) { std::cerr << "k2cli: failed to create pipeline\n"; std::exit(1); }
    return h;
}

static void usage() {
    std::cerr <<
        "Usage:\n"
        "  k2cli compress   <input> <output.k2>      (file -> .k2)\n"
        "  k2cli compress   <input_dir> <output>     (directory -> .k2a archive,\n"
        "                                              auto multi-volume if large)\n"
        "  k2cli decompress <input.k2> <output>      (.k2 -> file)\n"
        "  k2cli decompress <archive> <output_dir>   (.k2a[.NNN] -> directory)\n"
        "  k2cli roundtrip  <input>                  (file or directory)\n"
        "  k2cli parallel   <input> <n_threads>\n"
        "  k2cli stats      <input>\n"
        "\n"
        "Environment:\n"
        "  K2_ONNX_MODEL   path to ONNX classifier model (optional)\n"
        "  K2_GAIN_MODEL   path to ONNX transform-gain predictor model (optional)\n"
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

static int cmd_compress_archive(const std::string& in_path, const std::string& out_path) {
    std::cout << "K2 archive compress: " << in_path << " (directory)\n";
    k2a::ArchiveConfig cfg;   // defaults: 4GB volumes, 8MB blocks, auto threads
    std::string err;
    auto rc = k2a::pack_directory(in_path, out_path, cfg, &err);
    if (rc != k2a::ArchiveError::ok) {
        std::cerr << "k2cli: archive pack failed: " << k2a::archive_error_str(rc);
        if (!err.empty()) std::cerr << " (" << err << ")";
        std::cerr << "\n";
        return 1;
    }
    std::cout << "  → " << out_path << ".001"
                  "  (run 'k2cli decompress " << out_path << " <out_dir>' to unpack)\n";
    return 0;
}

static int cmd_decompress_archive(const std::string& in_path, const std::string& out_dir) {
    std::cout << "K2 archive decompress: " << in_path << " → " << out_dir << "/\n";
    k2a::ArchiveConfig cfg;
    std::string err;
    auto rc = k2a::unpack_archive(in_path, out_dir, cfg, &err);
    if (rc != k2a::ArchiveError::ok) {
        std::cerr << "k2cli: archive unpack failed: " << k2a::archive_error_str(rc);
        if (!err.empty()) std::cerr << " (" << err << ")";
        std::cerr << "\n";
        return 1;
    }
    std::cout << "  ✓ unpacked to " << out_dir << "/\n";
    return 0;
}

static int cmd_compress(const std::string& in_path, const std::string& out_path) {
    std::error_code ec;
    if (fs::is_directory(in_path, ec)) {
        return cmd_compress_archive(in_path, out_path);
    }

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
    if (k2a::looks_like_k2a(in_path)) {
        return cmd_decompress_archive(in_path, out_path);
    }

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

// Recursively compares two directory trees: same relative paths (files and
// empty dirs) and byte-identical file contents. Returns true on match;
// on mismatch, writes a short description to *detail.
static bool trees_match(const fs::path& a, const fs::path& b, std::string* detail) {
    std::vector<fs::path> rel_a, rel_b;
    std::error_code ec;
    for (auto& e : fs::recursive_directory_iterator(a, ec))
        rel_a.push_back(fs::relative(e.path(), a));
    for (auto& e : fs::recursive_directory_iterator(b, ec))
        rel_b.push_back(fs::relative(e.path(), b));
    std::sort(rel_a.begin(), rel_a.end());
    std::sort(rel_b.begin(), rel_b.end());
    if (rel_a != rel_b) {
        if (detail) *detail = "directory structure differs (file/dir set mismatch)";
        return false;
    }
    for (auto& rp : rel_a) {
        fs::path pa = a / rp, pb = b / rp;
        if (fs::is_directory(pa, ec)) continue;
        auto fa = read_file(pa.string());
        auto fb = read_file(pb.string());
        if (fa != fb) {
            if (detail) *detail = "content mismatch: " + rp.string();
            return false;
        }
    }
    return true;
}

static int cmd_roundtrip_archive(const std::string& in_dir) {
    std::error_code ec;
    uint64_t src_bytes = 0;
    for (auto& e : fs::recursive_directory_iterator(in_dir, ec))
        if (fs::is_regular_file(e, ec)) src_bytes += fs::file_size(e, ec);
    std::cout << "K2 archive roundtrip: " << in_dir << " (" << src_bytes << " bytes)\n";

    // Use a sibling of the input directory, not fs::temp_directory_path().
    // /tmp is frequently tmpfs (RAM-backed) or a small dedicated partition
    // with far less room than the filesystem actually holding the source
    // data — for large inputs (multi-GB+) that risks a mid-run write
    // failure. The archive + restored copy together need roughly
    // src_bytes (compressed output) + src_bytes (restored copy) of
    // headroom in the worst case (no compression at all); check for that
    // up front rather than discovering it after a long compression pass.
    fs::path in_path(in_dir);
    fs::path parent = fs::absolute(in_path, ec).parent_path();
    fs::path tmp_root = parent /
        ("." + in_path.filename().string() + "_k2cli_roundtrip_" +
         std::to_string(uint64_t(std::chrono::steady_clock::now()
             .time_since_epoch().count())));

    const auto space = fs::space(parent, ec);
    if (!ec) {
        const uint64_t needed = src_bytes * 2;   // archive + restored copy, worst case
        if (space.available < needed) {
            std::cerr << "  refusing to start: " << parent.string() << " has "
                      << (space.available / (1024 * 1024)) << " MB free, need roughly "
                      << (needed / (1024 * 1024)) << " MB (archive + restored copy, "
                      << "worst case no compression). Free up space or pass a smaller "
                      << "input.\n";
            return 1;
        }
    }
    fs::create_directories(tmp_root, ec);

    fs::path archive_base = tmp_root / "archive";
    fs::path out_dir = tmp_root / "restored";

    k2a::ArchiveConfig cfg;
    std::string err;
    auto rc = k2a::pack_directory(in_dir, archive_base.string(), cfg, &err);
    if (rc != k2a::ArchiveError::ok) {
        std::cerr << "  pack failed: " << k2a::archive_error_str(rc);
        if (!err.empty()) std::cerr << " (" << err << ")";
        std::cerr << "\n";
        fs::remove_all(tmp_root, ec);
        return 1;
    }

    uint64_t archive_bytes = 0;
    int n_volumes = 0;
    for (int i = 1; ; ++i) {
        char suffix[16]; std::snprintf(suffix, sizeof(suffix), ".%03d", i);
        fs::path vp = archive_base.string() + suffix;
        if (!fs::exists(vp, ec)) break;
        archive_bytes += fs::file_size(vp, ec);
        ++n_volumes;
    }
    const double ratio = archive_bytes ? double(src_bytes) / double(archive_bytes) : 0.0;
    std::cout << "  compressed: " << archive_bytes << " bytes across " << n_volumes
              << " volume(s) (" << std::fixed << std::setprecision(2) << ratio << "x)\n";

    rc = k2a::unpack_archive(archive_base.string(), out_dir.string(), cfg, &err);
    if (rc != k2a::ArchiveError::ok) {
        std::cerr << "  unpack failed: " << k2a::archive_error_str(rc);
        if (!err.empty()) std::cerr << " (" << err << ")";
        std::cerr << "\n";
        fs::remove_all(tmp_root, ec);
        return 1;
    }

    std::string mismatch;
    const bool ok = trees_match(in_dir, out_dir, &mismatch);
    fs::remove_all(tmp_root, ec);   // clean up the temp archive + restored copy either way

    if (ok) {
        std::cout << "  \xe2\x9c\x93 MATCH \xe2\x80\x94 roundtrip verified (" << src_bytes << " bytes)\n";
        return 0;
    }
    std::cerr << "  \xe2\x9c\x97 MISMATCH \xe2\x80\x94 " << mismatch << "\n";
    return 1;
}

static int cmd_roundtrip(const std::string& in_path) {
    std::error_code ec;
    if (fs::is_directory(in_path, ec)) {
        return cmd_roundtrip_archive(in_path);
    }

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
