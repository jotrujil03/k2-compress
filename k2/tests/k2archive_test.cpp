// k2archive_test.cpp — round-trip + edge-case verification for k2archive.
#include "k2archive.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { std::printf("  PASS  %s\n", msg); } \
    else { std::printf("  FAIL  %s\n", msg); ++g_fail; } \
    std::fflush(stdout); \
} while (0)

static void write_file(const fs::path& p, const std::vector<uint8_t>& data) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

static std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

static std::vector<uint8_t> make_text(size_t n, uint32_t seed = 1) {
    const char* s = "the quick brown fox jumps over the lazy dog ";
    size_t sl = strlen(s);
    std::vector<uint8_t> o(n);
    for (size_t i = 0; i < n; ++i) o[i] = uint8_t(s[(i + seed) % sl]);
    return o;
}

static std::vector<uint8_t> make_random(size_t n, uint32_t seed) {
    std::vector<uint8_t> o(n);
    uint32_t s = seed;
    for (auto& b : o) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; b = uint8_t(s); }
    return o;
}

// Compares two directory trees byte-for-byte, including structure.
static bool trees_match(const fs::path& a, const fs::path& b, std::string* mismatch) {
    std::vector<fs::path> rel_a, rel_b;
    for (auto& e : fs::recursive_directory_iterator(a))
        rel_a.push_back(fs::relative(e.path(), a));
    for (auto& e : fs::recursive_directory_iterator(b))
        rel_b.push_back(fs::relative(e.path(), b));
    std::sort(rel_a.begin(), rel_a.end());
    std::sort(rel_b.begin(), rel_b.end());
    if (rel_a != rel_b) {
        if (mismatch) *mismatch = "directory structure differs (file/dir set mismatch)";
        return false;
    }
    for (auto& rp : rel_a) {
        fs::path pa = a / rp, pb = b / rp;
        if (fs::is_directory(pa)) continue;
        auto da = read_file(pa), db = read_file(pb);
        if (da != db) {
            if (mismatch) *mismatch = "content mismatch: " + rp.string();
            return false;
        }
    }
    return true;
}

static bool pack_unpack_verify(const fs::path& src_dir, const fs::path& work_dir,
                                const std::string& label, k2a::ArchiveConfig cfg) {
    fs::path archive_base = work_dir / (label + "_archive");
    fs::path out_dir = work_dir / (label + "_unpacked");
    fs::remove_all(out_dir);

    std::string err;
    auto rc = k2a::pack_directory(src_dir.string(), archive_base.string(), cfg, &err);
    if (rc != k2a::ArchiveError::ok) {
        std::printf("    pack failed: %s (%s)\n", k2a::archive_error_str(rc), err.c_str());
        return false;
    }
    rc = k2a::unpack_archive(archive_base.string(), out_dir.string(), cfg, &err);
    if (rc != k2a::ArchiveError::ok) {
        std::printf("    unpack failed: %s (%s)\n", k2a::archive_error_str(rc), err.c_str());
        return false;
    }
    std::string mismatch;
    bool ok = trees_match(src_dir, out_dir, &mismatch);
    if (!ok) std::printf("    %s\n", mismatch.c_str());

    // report volume count + sizes
    uint64_t total_archive_bytes = 0;
    int vol_count = 0;
    for (int i = 1; ; ++i) {
        char buf[16]; std::snprintf(buf, sizeof(buf), ".%03d", i);
        fs::path vp = archive_base.string() + buf;
        if (!fs::exists(vp)) break;
        total_archive_bytes += fs::file_size(vp);
        ++vol_count;
    }
    uint64_t src_bytes = 0;
    for (auto& e : fs::recursive_directory_iterator(src_dir))
        if (fs::is_regular_file(e)) src_bytes += fs::file_size(e);
    std::printf("    %d volume(s), %llu -> %llu bytes (%.2fx)\n",
                vol_count, (unsigned long long)src_bytes,
                (unsigned long long)total_archive_bytes,
                src_bytes ? double(src_bytes) / double(total_archive_bytes) : 0.0);
    return ok;
}

int main() {
    fs::path tmp = fs::temp_directory_path() / "k2archive_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    // --- Test 1: small mixed directory, single volume expected ---
    {
        std::printf("\n[test_basic_directory]\n");
        fs::path src = tmp / "src1";
        write_file(src / "readme.txt", make_text(2000, 1));
        write_file(src / "assets/icon1.png", make_random(50000, 2));
        write_file(src / "assets/icon2.png", make_random(48000, 3));
        write_file(src / "assets/sub/deep.bin", make_text(10000, 4));
        write_file(src / "config.json", make_text(500, 5));

        k2a::ArchiveConfig cfg;
        cfg.volume_size_bytes = uint64_t(1) << 30;  // 1GB, won't split
        cfg.block_target_bytes = uint64_t(1) << 20; // 1MB blocks
        bool ok = pack_unpack_verify(src, tmp, "basic", cfg);
        CHECK(ok, "basic mixed directory roundtrip");
    }

    // --- Test 2: empty directories preserved ---
    {
        std::printf("\n[test_empty_dirs]\n");
        fs::path src = tmp / "src2";
        write_file(src / "file.txt", make_text(100, 6));
        fs::create_directories(src / "empty_dir");
        fs::create_directories(src / "nested/empty_inner");

        k2a::ArchiveConfig cfg;
        cfg.volume_size_bytes = uint64_t(1) << 30;
        cfg.block_target_bytes = uint64_t(1) << 20;
        bool ok = pack_unpack_verify(src, tmp, "emptydirs", cfg);
        CHECK(ok, "empty directories preserved through roundtrip");
    }

    // --- Test 3: zero-byte files ---
    {
        std::printf("\n[test_zero_byte_files]\n");
        fs::path src = tmp / "src3";
        write_file(src / "empty.txt", {});
        write_file(src / "normal.txt", make_text(1000, 7));

        k2a::ArchiveConfig cfg;
        cfg.volume_size_bytes = uint64_t(1) << 30;
        cfg.block_target_bytes = uint64_t(1) << 20;
        bool ok = pack_unpack_verify(src, tmp, "zerobyte", cfg);
        CHECK(ok, "zero-byte files preserved through roundtrip");
    }

    // --- Test 4: forced multi-volume split ---
    {
        std::printf("\n[test_multi_volume]\n");
        fs::path src = tmp / "src4";
        for (int i = 0; i < 10; ++i)
            write_file(src / ("file_" + std::to_string(i) + ".bin"),
                       make_random(600 * 1024, uint32_t(100 + i)));  // ~6MB total

        k2a::ArchiveConfig cfg;
        cfg.volume_size_bytes = uint64_t(2) << 20;   // 2MB volumes -> forces split
        cfg.block_target_bytes = uint64_t(512) << 10; // 512KB blocks
        bool ok = pack_unpack_verify(src, tmp, "multivol", cfg);
        CHECK(ok, "multi-volume roundtrip correctness");

        int vol_count = 0;
        for (int i = 1; ; ++i) {
            char buf[16]; std::snprintf(buf, sizeof(buf), ".%03d", i);
            if (!fs::exists((tmp / "multivol_archive").string() + buf)) break;
            ++vol_count;
        }
        char msg[64]; std::snprintf(msg, sizeof(msg), "split into multiple volumes (got %d)", vol_count);
        CHECK(vol_count > 1, msg);
    }

    // --- Test 5: single large file (bigger than block_target) ---
    {
        std::printf("\n[test_large_file]\n");
        fs::path src = tmp / "src5";
        write_file(src / "huge_texture.dds", make_text(8 * 1024 * 1024, 8));  // 8MB
        write_file(src / "small.txt", make_text(200, 9));

        k2a::ArchiveConfig cfg;
        cfg.volume_size_bytes = uint64_t(1) << 30;
        cfg.block_target_bytes = uint64_t(1) << 20;  // 1MB -- huge file isolated
        bool ok = pack_unpack_verify(src, tmp, "largefile", cfg);
        CHECK(ok, "large file (> block target) roundtrip");
    }

    // --- Test 6: archive-uid mismatch detection ---
    {
        std::printf("\n[test_uid_mismatch_detection]\n");
        fs::path srcA = tmp / "srcA";
        write_file(srcA / "a1.bin", make_random(8000, 10));
        write_file(srcA / "a2.bin", make_random(8000, 11));
        write_file(srcA / "a3.bin", make_random(8000, 12));
        fs::path srcB = tmp / "srcB";
        write_file(srcB / "b1.bin", make_random(8000, 20));
        write_file(srcB / "b2.bin", make_random(8000, 21));
        write_file(srcB / "b3.bin", make_random(8000, 22));

        k2a::ArchiveConfig cfg;
        cfg.volume_size_bytes = uint64_t(10) << 10;  // 10KB -- forces split (random data ~1.0x)
        cfg.block_target_bytes = uint64_t(8) << 10;  // 8KB blocks -> ~3 blocks per dir
        std::string err;
        k2a::pack_directory(srcA.string(), (tmp / "uidA_archive").string(), cfg, &err);
        k2a::pack_directory(srcB.string(), (tmp / "uidB_archive").string(), cfg, &err);

        // Frankenstein: copy archive A's part1, but B's part2 (different uid)
        fs::copy_file(tmp / "uidA_archive.001", tmp / "frank_archive.001",
                      fs::copy_options::overwrite_existing);
        fs::copy_file(tmp / "uidB_archive.002", tmp / "frank_archive.002",
                      fs::copy_options::overwrite_existing);

        auto rc = k2a::unpack_archive((tmp / "frank_archive").string(),
                                       (tmp / "frank_out").string(), cfg, &err);
        CHECK(rc != k2a::ArchiveError::ok, "mismatched-archive volumes rejected (not silently corrupted)");
        std::printf("    (error: %s / %s)\n", k2a::archive_error_str(rc), err.c_str());
    }

    // --- Test 7: looks_like_k2a sanity ---
    {
        std::printf("\n[test_looks_like_k2a]\n");
        CHECK(k2a::looks_like_k2a((tmp / "basic_archive").string()), "real archive detected");
        write_file(tmp / "not_an_archive.001", make_text(50, 99));
        CHECK(!k2a::looks_like_k2a((tmp / "not_an_archive").string()), "non-archive correctly rejected");
    }

    // --- Test 8: internal parallelism actually engages (regression test) ---
    //
    // Bug history: ArchiveConfig::block_target_bytes (K2A's file-grouping
    // size) was previously also used as asdp_config_t::min_block_bytes (the
    // floor for ASDP's OWN internal sub-block splitting). Since each K2A
    // block buffer is exactly block_target_bytes by construction, this made
    // ASDP's planner take its "src_len <= min_block" single-block early-out
    // every time — compression silently ran on exactly 1 thread regardless
    // of n_threads, while decompression (a separate code path with its own
    // untouched default config) parallelized correctly. Found by directly
    // observing CPU utilization during a real 12GB+ roundtrip: one core
    // active during compress, all cores during decompress.
    //
    // This test forces n_threads=4 (independent of how many cores the test
    // runner actually has — what matters is whether ASDP's internal planner
    // is GIVEN the chance to use more than one thread, not whether the
    // hardware can satisfy it) and a directory just over the default
    // block_target_bytes, then asserts more than one thread was actually
    // used for at least one K2A block.
    {
        std::printf("\n[test_internal_parallelism_engages]\n");
        fs::path src = tmp / "src8";
        // ~12MB of incompressible-ish data across a few files: large enough
        // relative to ASDP's internal min_block_bytes default (8MB) that a
        // single K2A block has real room to split, small enough to stay
        // fast in CI.
        for (int i = 0; i < 4; ++i)
            write_file(src / ("part_" + std::to_string(i) + ".bin"),
                       make_random(3 * 1024 * 1024, uint32_t(200 + i)));

        k2a::ArchiveConfig cfg;
        cfg.n_threads = 4;
        cfg.volume_size_bytes = uint64_t(1) << 30;
        // Leave block_target_bytes at its default (128MB) -- the whole 12MB
        // test directory lands in a single K2A block, exactly the
        // regression scenario.
        k2a::PackStats stats;
        std::string err;
        auto rc = k2a::pack_directory(src.string(), (tmp / "parallel_archive").string(),
                                       cfg, &err, &stats);
        CHECK(rc == k2a::ArchiveError::ok, "pack succeeds for internal-parallelism test");
        CHECK(stats.n_k2a_blocks == 1u, "test directory lands in a single K2A block (as intended)");

        int max_threads_used = 0;
        for (int t : stats.asdp_threads_used_per_block) max_threads_used = std::max(max_threads_used, t);
        std::printf("    n_k2a_blocks=%u  threads_used_per_block=[", stats.n_k2a_blocks);
        for (size_t i = 0; i < stats.asdp_threads_used_per_block.size(); ++i)
            std::printf("%s%d", i ? "," : "", stats.asdp_threads_used_per_block[i]);
        std::printf("]\n");
        CHECK(max_threads_used > 1,
              "internal ASDP parallelism actually engages (n_threads_used > 1) -- "
              "regression guard for the single-thread-compress bug");
    }

    std::printf("\n%s\n", g_fail == 0 ? "ALL K2ARCHIVE TESTS PASSED" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
