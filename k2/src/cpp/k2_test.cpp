/**
 * k2/src/cpp/k2_test.cpp
 *
 * C++ bridge smoke tests.
 * Runs without any external test framework — exits 0 on pass, 1 on fail.
 */

#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "k2_bridge.h"

// ---------------------------------------------------------------------------
// Minimal test runner
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (cond) {                                               \
            std::printf("  PASS  %s\n", msg);                    \
            ++g_passed;                                           \
        } else {                                                  \
            std::printf("  FAIL  %s  (line %d)\n", msg, __LINE__); \
            ++g_failed;                                           \
        }                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

static std::vector<uint8_t> make_timeseries(size_t n = 16384) {
    std::vector<uint8_t> out(n * 8);
    uint64_t val = 0;
    for (size_t i = 0; i < n; ++i) {
        val += (i % 100) + 1;
        std::memcpy(out.data() + i * 8, &val, 8);
    }
    return out;
}

static std::vector<uint8_t> make_text(size_t n = 8192) {
    const char* words = "the quick brown fox jumps over the lazy dog ";
    size_t wlen = std::strlen(words);
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<uint8_t>(words[i % wlen]);
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_create_destroy() {
    std::printf("\n[test_create_destroy]\n");
    K2Handle* h = k2_create(nullptr, 1.0, 0.15);
    CHECK(h != nullptr, "k2_create returns non-null");
    k2_destroy(h);
    CHECK(true, "k2_destroy does not crash");
}

static void test_prepare() {
    std::printf("\n[test_prepare]\n");
    auto data = make_timeseries();
    K2Handle* h = k2_create(nullptr, 1.0, 0.15);
    CHECK(h != nullptr, "handle created");

    int rc = k2_prepare(h, data.data(), std::min(data.size(), size_t(65536)));
    CHECK(rc == 0, "k2_prepare returns 0");

    k2_destroy(h);
}

static void test_compress_produces_output() {
    std::printf("\n[test_compress_produces_output]\n");
    auto data = make_timeseries(4096);
    K2Handle* h = k2_create(nullptr, 1.0, 0.15);

    k2_prepare(h, data.data(), data.size());

    std::vector<uint8_t> out(data.size() * 2 + 4096);
    size_t out_len = 0;
    int rc = k2_compress(h, data.data(), data.size(),
                          out.data(), out.size(), &out_len);

    CHECK(rc == 0,       "k2_compress returns 0");
    CHECK(out_len > 0,   "compressed output is non-empty");
    CHECK(out_len < data.size() * 2, "output within expected bounds");

    k2_destroy(h);
}

static void test_compress_ratio_timeseries() {
    std::printf("\n[test_compress_ratio_timeseries]\n");
    auto data = make_timeseries(8192);
    K2Handle* h = k2_create(nullptr, 0.0, 0.15);  // exploitation only

    k2_prepare(h, data.data(), data.size());

    std::vector<uint8_t> out(data.size() * 2);
    size_t out_len = 0;
    k2_compress(h, data.data(), data.size(),
                out.data(), out.size(), &out_len);

    double ratio = static_cast<double>(data.size()) / out_len;
    std::printf("    timeseries ratio: %.2fx\n", ratio);
    CHECK(ratio > 1.5, "timeseries compresses better than 1.5x");

    k2_destroy(h);
}

static void test_compress_text() {
    std::printf("\n[test_compress_text]\n");
    auto data = make_text(8192);
    K2Handle* h = k2_create(nullptr, 0.0, 0.15);

    k2_prepare(h, data.data(), data.size());

    std::vector<uint8_t> out(data.size() * 2);
    size_t out_len = 0;
    k2_compress(h, data.data(), data.size(),
                out.data(), out.size(), &out_len);

    double ratio = static_cast<double>(data.size()) / out_len;
    std::printf("    text ratio: %.2fx\n", ratio);
    CHECK(ratio > 1.0, "text compresses to something smaller");

    k2_destroy(h);
}

static void test_stats_returns_json() {
    std::printf("\n[test_stats_returns_json]\n");
    auto data = make_timeseries(4096);
    K2Handle* h = k2_create(nullptr, 1.0, 0.15);

    k2_prepare(h, data.data(), data.size());

    std::vector<uint8_t> out(data.size() * 2);
    size_t out_len = 0;
    k2_compress(h, data.data(), data.size(),
                out.data(), out.size(), &out_len);

    const char* stats = k2_stats(h);
    CHECK(stats != nullptr,              "k2_stats returns non-null");
    CHECK(std::strlen(stats) > 2,        "stats string is non-empty");
    CHECK(stats[0] == '{',               "stats looks like JSON");
    std::printf("    stats: %s\n", stats);

    k2_destroy(h);
}

static void test_null_handle_safety() {
    std::printf("\n[test_null_handle_safety]\n");
    // These should not crash
    int rc = k2_prepare(nullptr, nullptr, 0);
    CHECK(rc < 0, "k2_prepare(null) returns error");

    uint8_t buf[16]; size_t len = 0;
    rc = k2_compress(nullptr, buf, 8, buf, 16, &len);
    CHECK(rc < 0, "k2_compress(null) returns error");

    const char* s = k2_stats(nullptr);
    CHECK(s != nullptr, "k2_stats(null) returns non-null");

    k2_destroy(nullptr);  // should not crash
    CHECK(true, "k2_destroy(null) does not crash");
}

static void test_cpp_bridge() {
    std::printf("\n[test_cpp_bridge]\n");
    auto data = make_timeseries(4096);

    k2::Bridge bridge(K2_PYTHON_MODULE_DIR);
    std::string detected = bridge.prepare(data.data(), data.size());
    CHECK(!detected.empty(), "Bridge::prepare returns class name");
    std::printf("    detected: %s\n", detected.c_str());

    auto compressed = bridge.compress(data.data(), data.size());
    CHECK(!compressed.empty(), "Bridge::compress returns bytes");

    double ratio = static_cast<double>(data.size()) / compressed.size();
    std::printf("    ratio: %.2fx\n", ratio);
    CHECK(ratio > 1.0, "C++ bridge achieves compression");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== K2 C++ Bridge Tests ===\n");

    test_create_destroy();
    test_prepare();
    test_compress_produces_output();
    test_compress_ratio_timeseries();
    test_compress_text();
    test_stats_returns_json();
    test_null_handle_safety();
    test_cpp_bridge();

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
