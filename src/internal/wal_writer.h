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
class WALWriter {
 public:
  /// Factory method to open a WAL file for writing.
  /// Creates parent directory if needed, writes header if new file, opens in append mode.
  static utils::Result<WALWriter> open(const std::filesystem::path& walFilePath);

  /// Append a single entry to the WAL.
  utils::Status append(const Entry& entry);

  /// Append multiple entries in a batch (more efficient than multiple append() calls).
  utils::Status appendBatch(std::span<const Entry> entries);

  /// Sync the WAL to disk (fsync).
  utils::Status sync();

  /// Close the writer.
  void close();

  /// Check if the writer is open.
  bool isOpen() const noexcept { return writer_.has_value(); }

 private:
  explicit WALWriter(BinaryWriter&& writer, std::filesystem::path path);

  std::optional<BinaryWriter> writer_;
  std::filesystem::path path_;
};

}  // namespace arrow::wal

#endif  // ARROW_WAL_WRITER_H
