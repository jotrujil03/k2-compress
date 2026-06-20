/**
 * k2/include/k2archive.h
 *
 * K2A — directory archive format with manifest and multi-volume splitting.
 *
 * Layered strictly on top of ASDP: each "block" in a .k2a archive is a
 * complete, independent output of asdp_compress() (a full, self-describing
 * ASDP frame). K2A adds: a filename manifest, file-boundary-respecting
 * block grouping, and automatic multi-volume splitting when the archive
 * would exceed a size threshold.
 *
 * See k2a_format_design.md for the full wire-format specification this
 * header implements.
 *
 * Repack use case only: build once, unpack once, no partial/random access.
 * No installer/GUI layer — this is the archive mechanism only.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace k2a {

// ---------------------------------------------------------------------------
// Wire format constants
// ---------------------------------------------------------------------------

inline constexpr uint8_t  MAGIC_PART1[4] = {'K', '2', 'A', 0x01};
inline constexpr uint8_t  MAGIC_CONT[4]  = {'K', '2', 'A', 'C'};
inline constexpr uint8_t  FORMAT_VERSION = 0x01;

inline constexpr uint8_t  FLAG_MULTI_VOLUME = 0x01;

// Default volume size cap: 4 GiB (FAT32-safe, matches the long-standing
// 7z/rar split convention). Override via ArchiveConfig::volume_size_bytes.
inline constexpr uint64_t DEFAULT_VOLUME_SIZE = uint64_t(4) * 1024 * 1024 * 1024;

// Entry flags (manifest)
inline constexpr uint8_t  ENTRY_FLAG_DIR = 0x01;   // empty directory marker

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

enum class ArchiveError {
    ok = 0,
    cannot_open_input,
    cannot_create_output,
    bad_magic,
    bad_version,
    archive_uid_mismatch,
    volume_index_mismatch,
    missing_volume,
    manifest_corrupt,
    asdp_error,
    io_error,
};

const char* archive_error_str(ArchiveError e) noexcept;

// ---------------------------------------------------------------------------
// Manifest entry (in-memory representation; see design doc for wire format)
// ---------------------------------------------------------------------------

struct ManifestEntry {
    std::string path;          // relative, '/' separated, UTF-8
    uint64_t    file_size = 0; // original size in bytes (0 for directories)
    uint8_t     flags     = 0; // ENTRY_FLAG_*
    uint32_t    block_id  = 0; // which block this file's data starts in
    uint64_t    block_offset = 0; // byte offset within that block's payload
};

// ---------------------------------------------------------------------------
// Archive configuration
// ---------------------------------------------------------------------------

struct ArchiveConfig {
    uint64_t volume_size_bytes = DEFAULT_VOLUME_SIZE;
    // Target size for each K2A block (NOT the same as asdp_config_t's
    // min_block_bytes, though it's used as that value when compressing each
    // block — see k2a_format_design.md open-question 1 for why K2A blocks
    // are sized independently rather than as total_size/n_threads).
    uint64_t block_target_bytes = uint64_t(8) * 1024 * 1024;
    int      n_threads = 0;          // 0 = hardware_concurrency()
    int      asdp_level = 3;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Recursively pack `input_dir` into a K2A archive at `output_path`.
 *
 * If the resulting archive fits within one volume, writes exactly
 * `output_path` (or `output_path.k2a` if it lacks that extension).
 * If it would exceed cfg.volume_size_bytes, writes `output_path.001`,
 * `output_path.002`, ... automatically — the caller does not need to
 * decide up front; K2A decides after compressing each block.
 *
 * Returns ArchiveError::ok on success.
 */
ArchiveError pack_directory(const std::string& input_dir,
                             const std::string& output_path,
                             const ArchiveConfig& cfg,
                             std::string* err_detail = nullptr);

/**
 * Unpack a K2A archive (single or multi-volume) into `output_dir`.
 *
 * `archive_path` may name the single-volume file, the bare base name, or
 * any one part of a multi-volume set (e.g. "game.k2a.003") — part 1 is
 * always required and located automatically by deriving its name from
 * whichever path is given. All volume files must be present in the same
 * directory as part 1.
 *
 * Returns ArchiveError::ok on success.
 */
ArchiveError unpack_archive(const std::string& archive_path,
                             const std::string& output_dir,
                             const ArchiveConfig& cfg,
                             std::string* err_detail = nullptr);

/**
 * Quick check: does `path` look like a K2A archive (part 1, by magic),
 * without unpacking anything? Used by k2cli to decide compress-vs-decompress
 * dispatch when given an ambiguous path.
 */
bool looks_like_k2a(const std::string& path) noexcept;

}  // namespace k2a
