// Copyright 2025 ArrowDB
#include "internal/wal_writer.h"
#include "internal/wal.h"
#include "internal/filesync.h"
#include <ctime>

namespace arrow::wal {

using utils::Status;
using utils::OkStatus;

template <typename T> using Result = utils::Result<T>;

WALWriter::WALWriter(BinaryWriter&& writer, std::filesystem::path path)
    : writer_(std::move(writer)), path_(std::move(path)) {}

Result<WALWriter> WALWriter::open(const std::filesystem::path& walFilePath) {
  namespace fs = std::filesystem;

  // Create parent directory if needed
  fs::path parentDir = walFilePath.parent_path();
  if (!parentDir.empty() && !fs::exists(parentDir)) {
    try {
      fs::create_directories(parentDir);
    } catch (const std::exception& e) {
      return Status(StatusCode::kIoError,
                    "Failed to create WAL directory: " + std::string(e.what()));
    }
  }

  // Check if file exists to determine if we need to write a header
  bool needsHeader = !fs::exists(walFilePath);

  // Open in append mode (or truncate if new file)
  std::ios::openmode mode = std::ios::out | std::ios::binary |
                            (needsHeader ? std::ios::trunc : std::ios::app);

  auto ofs = std::make_unique<std::ofstream>(walFilePath, mode);
  if (!ofs->is_open()) {
    return Status(StatusCode::kIoError, "Failed to open WAL file for writing");
  }

  BinaryWriter writer(std::move(ofs));

  // Write header if this is a new file
  if (needsHeader) {
    Header header;
    header.magic = kWalMagic;
    header.creationTime = std::time(nullptr);
    header.headerCrc32 = header.computeCrc32();
    header.padding = 0;

    Status headerStatus = WriteHeader(header, writer);
    if (!headerStatus.ok()) {
      return headerStatus;
    }

    writer.flush();
    if (!utils::syncFile(walFilePath.string())) {
      return Status(StatusCode::kIoError, "fsync failed during header write");
    }
  }

  return WALWriter(std::move(writer), walFilePath);
}

Status WALWriter::append(const Entry& entry) {
  if (!writer_.has_value()) {
    return Status(StatusCode::kIoError, "WALWriter is not open");
  }

  Status writeStatus = WriteEntry(entry, writer_.value());
  if (!writeStatus.ok()) {
    return writeStatus;
  }

  // Flush and fsync after each entry for durability
  writer_.value().flush();
  if (!utils::syncFile(path_.string())) {
    return Status(StatusCode::kIoError, "fsync failed during append");
  }

  return OkStatus();
}

Status WALWriter::appendBatch(std::span<const Entry> entries) {
  if (!writer_.has_value()) {
    return Status(StatusCode::kIoError, "WALWriter is not open");
  }

  // Write all entries
  for (const Entry& entry : entries) {
    Status writeStatus = WriteEntry(entry, writer_.value());
    if (!writeStatus.ok()) {
      return writeStatus;
    }
  }

  // Single flush and fsync for entire batch (more efficient)
  writer_.value().flush();
  if (!utils::syncFile(path_.string())) {
    return Status(StatusCode::kIoError, "fsync failed during batch append");
  }

  return OkStatus();
}

Status WALWriter::sync() {
  if (!writer_.has_value()) {
    return Status(StatusCode::kIoError, "WALWriter is not open");
  }

  writer_.value().flush();
  if (!utils::syncFile(path_.string())) {
    return Status(StatusCode::kIoError, "fsync failed");
  }

  return OkStatus();
}

void WALWriter::close() {
  writer_.reset();
}

}  // namespace arrow::wal
