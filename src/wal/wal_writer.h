#ifndef ARROW_WAL_WRITER_H
#define ARROW_WAL_WRITER_H

#include <filesystem>
#include <optional>
#include <span>
#include "binary.h"
#include "arrow/utils/result.h"

namespace arrow::wal {

// Forward declarations from wal.h
struct Entry;

/// Holds an open BinaryWriter for the WAL file. Appends entries without reopening.
/// Replaces the pattern of opening, writing, fsyncing, closing on every log() call.
///
/// Maintains a cached file descriptor for fsync, avoiding the overhead of
/// open()/close() on every sync operation.
class WALWriter {
 public:
  /// Factory method to open a WAL file for writing.
  /// Creates parent directory if needed, writes header if new file, opens in append mode.
  /// Also opens a cached fd for efficient fsync.
  static utils::Result<WALWriter> open(const std::filesystem::path& walFilePath);

  ~WALWriter();

  WALWriter(WALWriter&& other) noexcept;
  WALWriter& operator=(WALWriter&& other) noexcept;

  WALWriter(const WALWriter&) = delete;
  WALWriter& operator=(const WALWriter&) = delete;

  /// Append a single entry and fsync (full durability).
  utils::Status append(const Entry& entry);

  /// Append a single entry without fsync (caller is responsible for calling sync() later).
  /// The entry is flushed to the OS page cache but not fsynced to disk.
  utils::Status appendDeferred(const Entry& entry);

  /// Append multiple entries in a batch with single fsync.
  utils::Status appendBatch(std::span<const Entry> entries);

  /// Sync the WAL to disk (fsync via cached fd).
  utils::Status sync();

  /// Close the writer and release the cached sync fd.
  void close();

  /// Check if the writer is open.
  bool isOpen() const noexcept { return writer_.has_value(); }

 private:
  explicit WALWriter(BinaryWriter&& writer, std::filesystem::path path, int syncFd);

  /// Fsync via cached fd, falling back to syncFile() if no cached fd.
  utils::Status syncInternal();

  std::optional<BinaryWriter> writer_;
  std::filesystem::path path_;
  int syncFd_ = -1;  // Cached fd for fsync (avoids open/close per sync)
};

}  // namespace arrow::wal

#endif  // ARROW_WAL_WRITER_H
