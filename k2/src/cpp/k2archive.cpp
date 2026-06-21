/**
 * k2/src/cpp/k2archive.cpp
 *
 * See k2archive.h and k2a_format_design.md for the format specification.
 */

#include "k2archive.h"
#include "asdp/asdp.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>
#include <type_traits>

namespace fs = std::filesystem;

namespace k2a {

const char* archive_error_str(ArchiveError e) noexcept {
    switch (e) {
        case ArchiveError::ok:                    return "ok";
        case ArchiveError::cannot_open_input:      return "cannot open input";
        case ArchiveError::cannot_create_output:   return "cannot create output";
        case ArchiveError::bad_magic:              return "bad K2A magic";
        case ArchiveError::bad_version:            return "unsupported K2A version";
        case ArchiveError::archive_uid_mismatch:   return "volume belongs to a different archive";
        case ArchiveError::volume_index_mismatch:  return "volume index out of order";
        case ArchiveError::missing_volume:         return "a required volume part is missing";
        case ArchiveError::manifest_corrupt:       return "manifest corrupt or truncated";
        case ArchiveError::asdp_error:             return "ASDP compression/decompression error";
        case ArchiveError::io_error:               return "I/O error";
    }
    return "unknown error";
}

// ---------------------------------------------------------------------------
// Small binary I/O helpers (host is little-endian, same assumption ASDP makes)
// ---------------------------------------------------------------------------

namespace {

template <typename T>
void put_le(std::vector<uint8_t>& out, T v) {
    static_assert(std::is_integral_v<T>);
    for (size_t i = 0; i < sizeof(T); ++i)
        out.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}

template <typename T>
bool get_le(const uint8_t* p, size_t avail, T& out) {
    if (avail < sizeof(T)) return false;
    T v = 0;
    for (size_t i = 0; i < sizeof(T); ++i)
        v |= T(p[i]) << (8 * i);
    out = v;
    return true;
}

uint32_t crc32(const uint8_t* data, size_t len) {
    // C++11 guarantees thread-safe initialization of function-local statics.
    // Previously this used a separate `static uint32_t table[256]` +
    // `static bool init` with a non-atomic check-then-fill — the same race
    // pattern found and fixed in ASDP's cm.cpp stretch/squash tables (see
    // that file for the ThreadSanitizer-confirmed repro). Not currently
    // reachable concurrently in this codebase (both call sites run on the
    // single thread driving pack_directory/unpack_archive, never from the
    // per-block parallel workers), but fixed regardless since that's a
    // fragile invariant to rely on going forward.
    struct Table {
        uint32_t v[256];
        Table() noexcept {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int k = 0; k < 8; ++k)
                    c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                v[i] = c;
            }
        }
    };
    static const Table table;

    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        c = table.v[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

std::string normalize_rel_path(const fs::path& base, const fs::path& full) {
    fs::path rel = fs::relative(full, base);
    std::string s = rel.generic_string();   // always '/' separators
    return s;
}

uint64_t random_uid() {
    // Not cryptographic — only needs to distinguish "did the user mix parts
    // from two different archives", a 64-bit space is ample for that.
    std::random_device rd;
    uint64_t hi = uint64_t(rd()) << 32, lo = uint64_t(rd());
    return hi ^ lo ^ uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
}

}  // namespace

// ---------------------------------------------------------------------------
// Manifest serialization
// ---------------------------------------------------------------------------

namespace {

std::vector<uint8_t> serialize_manifest(const std::vector<ManifestEntry>& entries) {
    std::vector<uint8_t> out;
    put_le<uint32_t>(out, uint32_t(entries.size()));
    for (const auto& e : entries) {
        put_le<uint16_t>(out, uint16_t(e.path.size()));
        out.insert(out.end(), e.path.begin(), e.path.end());
        put_le<uint64_t>(out, e.file_size);
        out.push_back(e.flags);
        put_le<uint32_t>(out, e.block_id);
        put_le<uint64_t>(out, e.block_offset);
    }
    return out;
}

bool deserialize_manifest(const uint8_t* p, size_t len, std::vector<ManifestEntry>& out) {
    size_t pos = 0;
    uint32_t n = 0;
    if (!get_le(p + pos, len - pos, n)) return false;
    pos += 4;
    out.clear();
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        ManifestEntry e;
        uint16_t path_len = 0;
        if (!get_le(p + pos, len - pos, path_len)) return false;
        pos += 2;
        if (pos + path_len > len) return false;
        e.path.assign(reinterpret_cast<const char*>(p + pos), path_len);
        pos += path_len;
        if (!get_le(p + pos, len - pos, e.file_size)) return false;
        pos += 8;
        if (pos >= len) return false;
        e.flags = p[pos]; pos += 1;
        if (!get_le(p + pos, len - pos, e.block_id)) return false;
        pos += 4;
        if (!get_le(p + pos, len - pos, e.block_offset)) return false;
        pos += 8;
        out.push_back(std::move(e));
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Directory walk + block planning
// ---------------------------------------------------------------------------

namespace {

struct WalkedFile {
    std::string rel_path;
    uint64_t    size;
    bool        is_dir;   // true only for EMPTY directories (others are
                           // implied by their files' paths)
};

std::vector<WalkedFile> walk_directory(const fs::path& root, std::string* err) {
    std::vector<WalkedFile> files;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) { if (err) *err = "directory walk error: " + ec.message(); break; }
        const auto& entry = *it;
        if (entry.is_directory(ec)) {
            if (fs::is_empty(entry.path(), ec) && !ec) {
                files.push_back({normalize_rel_path(root, entry.path()), 0, true});
            }
            continue;
        }
        if (entry.is_regular_file(ec)) {
            uint64_t sz = uint64_t(entry.file_size(ec));
            files.push_back({normalize_rel_path(root, entry.path()), sz, false});
        }
        // symlinks, devices, etc. are silently skipped (repack use case:
        // game asset directories don't contain those in practice).
    }
    // Deterministic order: sort by path. This also clusters files that
    // share a directory prefix adjacent to each other, which helps the CM
    // match model find structure within a block (similarly-named/typed
    // files often have similar headers/layouts).
    std::sort(files.begin(), files.end(),
              [](const WalkedFile& a, const WalkedFile& b) { return a.rel_path < b.rel_path; });
    return files;
}

// A planned block: which files (by index into `files`) it contains.
// Large files (> block_target_bytes) get a block of their own.
struct PlannedBlock {
    std::vector<size_t> file_indices;   // indices into the walked-files list
    uint64_t total_size = 0;
};

std::vector<PlannedBlock> plan_blocks(const std::vector<WalkedFile>& files,
                                       uint64_t block_target_bytes) {
    std::vector<PlannedBlock> blocks;
    PlannedBlock cur;
    for (size_t i = 0; i < files.size(); ++i) {
        const auto& f = files[i];
        if (f.is_dir || f.size == 0) {
            // Empty directories and zero-byte files ride along with the
            // current block (contribute no bytes); flush logic below is
            // unaffected since they add 0 to total_size.
            cur.file_indices.push_back(i);
            continue;
        }
        if (f.size > block_target_bytes) {
            // Large file: isolate it. Flush whatever's pending first.
            if (!cur.file_indices.empty()) { blocks.push_back(std::move(cur)); cur = PlannedBlock{}; }
            PlannedBlock solo;
            solo.file_indices.push_back(i);
            solo.total_size = f.size;
            blocks.push_back(std::move(solo));
            continue;
        }
        if (!cur.file_indices.empty() && cur.total_size + f.size > block_target_bytes) {
            blocks.push_back(std::move(cur));
            cur = PlannedBlock{};
        }
        cur.file_indices.push_back(i);
        cur.total_size += f.size;
    }
    if (!cur.file_indices.empty()) blocks.push_back(std::move(cur));
    return blocks;
}

}  // namespace

// ---------------------------------------------------------------------------
// Volume writer — handles rollover transparently
// ---------------------------------------------------------------------------

namespace {

class VolumeWriter {
public:
    VolumeWriter(std::string base_path, uint64_t cap, uint64_t archive_uid)
        : _base(std::move(base_path)), _cap(cap), _uid(archive_uid) {}

    // Call once, before writing any blocks: reserves space for part 1's
    // header by writing it (total_volumes patched at finalize()).
    bool begin(const std::vector<uint8_t>& manifest_bytes, uint64_t total_orig_size,
               std::string* err) {
        _manifest_len = manifest_bytes.size();
        if (!open_volume(1, err)) return false;

        std::vector<uint8_t> hdr;
        hdr.insert(hdr.end(), MAGIC_PART1, MAGIC_PART1 + 4);
        hdr.push_back(FORMAT_VERSION);
        hdr.push_back(0);              // flags placeholder, patched at finalize()
        put_le<uint16_t>(hdr, 0);      // reserved
        put_le<uint32_t>(hdr, 0);      // total_volumes placeholder
        put_le<uint64_t>(hdr, _uid);
        put_le<uint64_t>(hdr, total_orig_size);
        put_le<uint64_t>(hdr, _manifest_len);
        put_le<uint32_t>(hdr, crc32(manifest_bytes.data(), manifest_bytes.size()));
        _header_len = hdr.size();
        if (!write_raw(hdr.data(), hdr.size(), err)) return false;
        if (!write_raw(manifest_bytes.data(), manifest_bytes.size(), err)) return false;
        _bytes_in_volume = hdr.size() + manifest_bytes.size();
        return true;
    }

    // Writes one block (header + ASDP frame), rolling to a new volume first
    // if it wouldn't fit. A block is NEVER split across volumes.
    bool write_block(uint32_t block_orig_len, const std::vector<uint8_t>& frame,
                      std::string* err) {
        const uint64_t block_total = 4 + 4 + frame.size();
        const uint64_t min_volume_floor = (_volume_index == 1) ? _header_len + _manifest_len
                                                                  : cont_header_len();
        if (_bytes_in_volume > min_volume_floor && _bytes_in_volume + block_total > _cap) {
            if (!roll_to_next_volume(err)) return false;
        }
        std::vector<uint8_t> bh;
        put_le<uint32_t>(bh, block_orig_len);
        put_le<uint32_t>(bh, uint32_t(frame.size()));
        if (!write_raw(bh.data(), bh.size(), err)) return false;
        if (!write_raw(frame.data(), frame.size(), err)) return false;
        _bytes_in_volume += block_total;
        return true;
    }

    // Patches part 1's total_volumes field now that it's known, and closes
    // the current (last) volume file.
    bool finalize(std::string* err) {
        close_current();
        // Reopen part 1 in-place to patch flags + total_volumes.
        const std::string part1_name = volume_name(1);
        std::fstream f(part1_name, std::ios::in | std::ios::out | std::ios::binary);
        if (!f) { if (err) *err = "cannot reopen " + part1_name + " to finalize header"; return false; }
        uint8_t flags = (_volume_index > 1) ? FLAG_MULTI_VOLUME : 0;
        f.seekp(5); f.put(char(flags));
        f.seekp(8);
        uint8_t tv[4];
        uint32_t total = uint32_t(_volume_index);
        for (int i = 0; i < 4; ++i) tv[i] = uint8_t((total >> (8 * i)) & 0xFF);
        f.write(reinterpret_cast<char*>(tv), 4);
        return bool(f);
    }

    uint32_t volumes_written() const { return _volume_index; }
    std::string last_part1_name() const { return volume_name(1); }

private:
    std::string _base;
    uint64_t    _cap;
    uint64_t    _uid;
    uint32_t    _volume_index = 0;
    uint64_t    _bytes_in_volume = 0;
    uint64_t    _header_len = 0;
    uint64_t    _manifest_len = 0;
    std::ofstream _f;

    static uint64_t cont_header_len() { return 24; }

    std::string volume_name(uint32_t idx) const {
        std::ostringstream oss;
        oss << _base << "." << std::setfill('0') << std::setw(3) << idx;
        return oss.str();
    }

    void close_current() { if (_f.is_open()) _f.close(); }

    bool open_volume(uint32_t idx, std::string* err) {
        close_current();
        _volume_index = idx;
        _bytes_in_volume = 0;
        const std::string name = volume_name(idx);
        _f.open(name, std::ios::binary | std::ios::trunc);
        if (!_f) { if (err) *err = "cannot create " + name; return false; }
        return true;
    }

    bool roll_to_next_volume(std::string* err) {
        const uint32_t next = _volume_index + 1;
        if (!open_volume(next, err)) return false;
        std::vector<uint8_t> ch;
        ch.insert(ch.end(), MAGIC_CONT, MAGIC_CONT + 4);
        ch.push_back(FORMAT_VERSION);
        ch.push_back(0); ch.push_back(0); ch.push_back(0);   // reserved
        put_le<uint64_t>(ch, _uid);
        put_le<uint32_t>(ch, next);
        put_le<uint32_t>(ch, 0);   // total_volumes unknown yet; patched only
                                    // in part 1 — readers rely on part 1's
                                    // value and use this only as a sanity hint
        if (!write_raw(ch.data(), ch.size(), err)) return false;
        _bytes_in_volume = ch.size();
        return true;
    }

    bool write_raw(const uint8_t* p, size_t n, std::string* err) {
        _f.write(reinterpret_cast<const char*>(p), std::streamsize(n));
        if (!_f) {
            if (err) {
                *err = "write failed to " + volume_name(_volume_index) +
                       " (" + std::strerror(errno) + ") -- check available disk "
                       "space at that location";
            }
            return false;
        }
        return true;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// pack_directory
// ---------------------------------------------------------------------------

ArchiveError pack_directory(const std::string& input_dir,
                             const std::string& output_path,
                             const ArchiveConfig& cfg,
                             std::string* err_detail,
                             PackStats* stats_out) {
    std::error_code ec;
    fs::path root(input_dir);
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) {
        if (err_detail) *err_detail = input_dir + " is not a directory";
        return ArchiveError::cannot_open_input;
    }

    std::string walk_err;
    std::vector<WalkedFile> files = walk_directory(root, &walk_err);
    if (!walk_err.empty() && err_detail) *err_detail = walk_err;

    uint64_t total_orig_size = 0;
    for (const auto& f : files) total_orig_size += f.size;

    auto blocks = plan_blocks(files, cfg.block_target_bytes);

    // Build manifest in-step with planning (we know each file's block_id /
    // block_offset once block assignment is fixed, before any compression).
    std::vector<ManifestEntry> manifest(files.size());
    for (uint32_t bi = 0; bi < blocks.size(); ++bi) {
        uint64_t offset_in_block = 0;
        for (size_t fi : blocks[bi].file_indices) {
            ManifestEntry& e = manifest[fi];
            e.path = files[fi].rel_path;
            e.file_size = files[fi].size;
            e.flags = files[fi].is_dir ? ENTRY_FLAG_DIR : 0;
            e.block_id = bi;
            e.block_offset = offset_in_block;
            offset_in_block += files[fi].size;
        }
    }
    std::vector<uint8_t> manifest_bytes = serialize_manifest(manifest);

    const uint64_t uid = random_uid();
    VolumeWriter writer(output_path, cfg.volume_size_bytes, uid);
    if (!writer.begin(manifest_bytes, total_orig_size, err_detail))
        return ArchiveError::cannot_create_output;

    // Process K2A blocks ONE AT A TIME (see k2a_format_design.md open
    // question 1): each individual asdp_compress() call internally
    // parallelizes across cfg.n_threads for that block's data, so no
    // archive-level thread pool is needed, memory stays bounded to one
    // block's buffers, and volume writes are strictly sequential.
    //
    // IMPORTANT: acfg.min_block_bytes is the floor for ASDP's OWN internal
    // sub-block splitting within a single asdp_compress() call — a DIFFERENT
    // knob from cfg.block_target_bytes (K2A-level file grouping, ratio-tuned).
    // These two must stay independent: if acfg.min_block_bytes is set to
    // cfg.block_target_bytes, every K2A block buffer matches ASDP's splitting
    // floor, asdp_compress()'s plan_blocks() always takes the "src_len <=
    // min_block" early-out, and compression silently runs single-threaded
    // regardless of cfg.n_threads. Leave ASDP's internal floor at its own
    // default so each K2A block can split across threads internally.
    asdp_config_t acfg = asdp_default_config();
    acfg.level = cfg.asdp_level;
    acfg.n_threads = cfg.n_threads;
    // acfg.min_block_bytes intentionally left at asdp_default_config()'s
    // own default (8 MB) — do not tie it to cfg.block_target_bytes.
    acfg.no_split_below_bytes = std::min<uint64_t>(cfg.block_target_bytes, uint64_t(4) << 20);

    for (const auto& blk : blocks) {
        // Concatenate this block's files into one buffer.
        std::vector<uint8_t> block_buf;
        block_buf.reserve(blk.total_size);
        for (size_t fi : blk.file_indices) {
            if (files[fi].is_dir || files[fi].size == 0) continue;
            std::ifstream f(root / fs::path(files[fi].rel_path), std::ios::binary);
            if (!f) {
                if (err_detail) *err_detail = "cannot open " + files[fi].rel_path;
                return ArchiveError::cannot_open_input;
            }
            const size_t old_size = block_buf.size();
            block_buf.resize(old_size + files[fi].size);
            f.read(reinterpret_cast<char*>(block_buf.data() + old_size),
                   std::streamsize(files[fi].size));
            if (size_t(f.gcount()) != files[fi].size) {
                if (err_detail) *err_detail = "short read on " + files[fi].rel_path;
                return ArchiveError::io_error;
            }
        }

        asdp_ctx_t* actx = asdp_create(&acfg);
        if (!actx) return ArchiveError::asdp_error;
        std::vector<uint8_t> comp(asdp_compress_bound(block_buf.size()));
        size_t comp_len = 0;
        const int rc = asdp_compress(actx, block_buf.data(), block_buf.size(),
                                      comp.data(), comp.size(), &comp_len);
        if (stats_out) {
            asdp_stats_t st{};
            asdp_stats(actx, &st);
            stats_out->asdp_threads_used_per_block.push_back(st.n_threads_used);
        }
        asdp_destroy(actx);
        if (rc != ASDP_OK) {
            if (err_detail) *err_detail = std::string("asdp_compress: ") + asdp_error_str(rc);
            return ArchiveError::asdp_error;
        }
        comp.resize(comp_len);

        if (!writer.write_block(uint32_t(block_buf.size()), comp, err_detail))
            return ArchiveError::io_error;
    }

    if (!writer.finalize(err_detail)) return ArchiveError::io_error;
    if (stats_out) {
        stats_out->n_k2a_blocks = uint32_t(blocks.size());
        stats_out->n_volumes = writer.volumes_written();
    }
    return ArchiveError::ok;
}

// ---------------------------------------------------------------------------
// Volume reader
// ---------------------------------------------------------------------------

namespace {

class VolumeReader {
public:
    explicit VolumeReader(std::string base_path) : _base(std::move(base_path)) {}

    bool open_part1(std::string* err) {
        const std::string name = part_name(1);
        _f.open(name, std::ios::binary);
        if (!_f) { if (err) *err = "cannot open " + name; return false; }

        uint8_t hdr[40];
        _f.read(reinterpret_cast<char*>(hdr), 40);
        if (_f.gcount() != 40) { if (err) *err = "truncated header in " + name; return false; }
        if (std::memcmp(hdr, MAGIC_PART1, 4) != 0) { if (err) *err = "bad magic in " + name; return false; }
        if (hdr[4] != FORMAT_VERSION) { if (err) *err = "unsupported version in " + name; return false; }
        _flags = hdr[5];
        get_le(hdr + 8, 4, _total_volumes);
        get_le(hdr + 12, 8, _uid);
        get_le(hdr + 20, 8, _total_orig_size);
        uint64_t manifest_len = 0;
        get_le(hdr + 28, 8, manifest_len);
        uint32_t manifest_crc = 0;
        get_le(hdr + 36, 4, manifest_crc);

        std::vector<uint8_t> manifest_bytes(manifest_len);
        _f.read(reinterpret_cast<char*>(manifest_bytes.data()), std::streamsize(manifest_len));
        if (uint64_t(_f.gcount()) != manifest_len) {
            if (err) *err = "truncated manifest in " + name;
            return false;
        }
        if (crc32(manifest_bytes.data(), manifest_bytes.size()) != manifest_crc) {
            if (err) *err = "manifest CRC mismatch in " + name;
            return false;
        }
        if (!deserialize_manifest(manifest_bytes.data(), manifest_bytes.size(), manifest)) {
            if (err) *err = "manifest parse error in " + name;
            return false;
        }
        _volume_index = 1;
        return true;
    }

    // Reads the next block in the (possibly multi-volume) stream. Returns
    // false + sets *done=true at clean end of archive; false + *done=false
    // on a real error.
    bool next_block(std::vector<uint8_t>& frame_out, uint32_t& orig_len_out,
                     bool& done, std::string* err) {
        done = false;
        for (;;) {
            uint8_t bh[8];
            _f.read(reinterpret_cast<char*>(bh), 8);
            const std::streamsize got = _f.gcount();
            if (got == 8) {
                get_le(bh, 4, orig_len_out);
                uint32_t frame_len = 0;
                get_le(bh + 4, 4, frame_len);
                frame_out.resize(frame_len);
                _f.read(reinterpret_cast<char*>(frame_out.data()), frame_len);
                if (uint64_t(_f.gcount()) != frame_len) {
                    if (err) *err = "truncated block frame";
                    return false;
                }
                return true;
            }
            if (got != 0) { if (err) *err = "truncated block header"; return false; }
            // Clean EOF on this volume: advance to next, if any.
            if (_volume_index >= _total_volumes) { done = true; return false; }
            if (!open_continuation(_volume_index + 1, err)) return false;
        }
    }

    uint64_t total_orig_size() const { return _total_orig_size; }
    std::vector<ManifestEntry> manifest;

private:
    std::string _base;
    std::ifstream _f;
    uint8_t  _flags = 0;
    uint32_t _total_volumes = 1;
    uint64_t _uid = 0;
    uint64_t _total_orig_size = 0;
    uint32_t _volume_index = 0;

    std::string part_name(uint32_t idx) const {
        std::ostringstream oss;
        oss << _base << "." << std::setfill('0') << std::setw(3) << idx;
        return oss.str();
    }

    bool open_continuation(uint32_t idx, std::string* err) {
        _f.close();
        const std::string name = part_name(idx);
        _f.open(name, std::ios::binary);
        if (!_f) { if (err) *err = "missing volume " + name; return false; }
        uint8_t hdr[24];
        _f.read(reinterpret_cast<char*>(hdr), 24);
        if (_f.gcount() != 24) { if (err) *err = "truncated continuation header in " + name; return false; }
        if (std::memcmp(hdr, MAGIC_CONT, 4) != 0) { if (err) *err = "bad continuation magic in " + name; return false; }
        uint64_t uid = 0;
        get_le(hdr + 8, 8, uid);
        if (uid != _uid) { if (err) *err = name + " belongs to a different archive"; return false; }
        uint32_t vidx = 0;
        get_le(hdr + 16, 4, vidx);
        if (vidx != idx) { if (err) *err = name + " has unexpected volume index"; return false; }
        _volume_index = idx;
        return true;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// unpack_archive
// ---------------------------------------------------------------------------

namespace {

// Derive the "base path" (without .NNN suffix) from whatever path the user
// gave us — they may have pointed at part 1, part 3, or a bare base name.
std::string derive_base_path(const std::string& given) {
    fs::path p(given);
    const std::string ext = p.extension().string();
    // Matches ".001".. ".999"
    if (ext.size() == 4 && ext[0] == '.' &&
        std::isdigit((unsigned char)ext[1]) && std::isdigit((unsigned char)ext[2]) &&
        std::isdigit((unsigned char)ext[3])) {
        return p.parent_path().empty()
            ? p.stem().string()
            : (p.parent_path() / p.stem()).string();
    }
    return given;
}

}  // namespace

bool looks_like_k2a(const std::string& path) noexcept {
    const std::string base = derive_base_path(path);
    std::ostringstream oss;
    oss << base << ".001";
    std::ifstream f(oss.str(), std::ios::binary);
    if (!f) {
        // also accept a bare single-volume file with no .001 suffix
        f.open(base, std::ios::binary);
        if (!f) return false;
    }
    uint8_t magic[4];
    f.read(reinterpret_cast<char*>(magic), 4);
    return f.gcount() == 4 && std::memcmp(magic, MAGIC_PART1, 4) == 0;
}

ArchiveError unpack_archive(const std::string& archive_path,
                             const std::string& output_dir,
                             const ArchiveConfig& /*cfg*/,
                             std::string* err_detail) {
    const std::string base = derive_base_path(archive_path);
    VolumeReader reader(base);

    // Part 1 may be named "<base>.001" (multi-volume convention, used even
    // for single-volume archives per the format spec) or just "<base>" if
    // the caller created it without the numeric suffix.
    std::string err;
    if (!reader.open_part1(&err)) {
        // Retry treating archive_path itself as a bare single-volume file
        // with no ".001" suffix — handled inside VolumeReader::part_name
        // via the base/".001" convention, so if that failed, surface the
        // original error.
        if (err_detail) *err_detail = err;
        return ArchiveError::missing_volume;
    }

    fs::path out_root(output_dir);
    std::error_code ec;
    fs::create_directories(out_root, ec);

    // Open output files lazily, write each file's bytes as its block is
    // decoded (a file's bytes are entirely within one block, by
    // construction at pack time).
    std::vector<ManifestEntry>& manifest = reader.manifest;

    // Pre-create empty directories (files create their own parent dirs
    // on demand below).
    for (const auto& e : manifest) {
        if (e.flags & ENTRY_FLAG_DIR) {
            fs::create_directories(out_root / fs::path(e.path), ec);
        }
    }

    uint32_t block_id = 0;
    bool done = false;
    while (true) {
        std::vector<uint8_t> frame;
        uint32_t orig_len = 0;
        std::string berr;
        if (!reader.next_block(frame, orig_len, done, &berr)) {
            if (done) break;
            if (err_detail) *err_detail = berr;
            return ArchiveError::io_error;
        }

        asdp_config_t acfg = asdp_default_config();
        asdp_ctx_t* actx = asdp_create(&acfg);
        if (!actx) return ArchiveError::asdp_error;
        std::vector<uint8_t> payload(orig_len);
        size_t out_len = 0;
        const int rc = asdp_decompress(actx, frame.data(), frame.size(),
                                        payload.data(), payload.size(), &out_len);
        asdp_destroy(actx);
        if (rc != ASDP_OK || out_len != orig_len) {
            if (err_detail) *err_detail = std::string("asdp_decompress: ") + asdp_error_str(rc);
            return ArchiveError::asdp_error;
        }

        for (const auto& e : manifest) {
            if (e.flags & ENTRY_FLAG_DIR) continue;
            if (e.block_id != block_id) continue;
            if (e.block_offset + e.file_size > payload.size()) {
                if (err_detail) *err_detail = "manifest offset out of range for " + e.path;
                return ArchiveError::manifest_corrupt;
            }
            fs::path out_file = out_root / fs::path(e.path);
            fs::create_directories(out_file.parent_path(), ec);
            std::ofstream of(out_file, std::ios::binary | std::ios::trunc);
            if (!of) {
                if (err_detail) *err_detail = "cannot create " + out_file.string();
                return ArchiveError::cannot_create_output;
            }
            if (e.file_size > 0)
                of.write(reinterpret_cast<const char*>(payload.data() + e.block_offset),
                         std::streamsize(e.file_size));
        }
        ++block_id;
    }

    return ArchiveError::ok;
}

}  // namespace k2a
