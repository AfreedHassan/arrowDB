// Copyright 2025 ArrowDB
#include "wal/wal.h"
#include "wal/wal_writer.h"
#include "utils/crc32.h"
#include "utils/filesync.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace arrow::wal {
using namespace utils;

static constexpr size_t FILECRC32SIZE = sizeof(uint32_t);

// Header helpers
uint32_t Header::computeCrc32() const noexcept {
  return utils::crc32((const void *)this, offsetof(Header, headerCrc32));
}

utils::json Header::toJson() const {
  json j = json::object();
  j["magic"] = magic;
  j["version"] = version;
  j["flags"] = flags;
  j["creationTime"] = creationTime;
  j["headerCrc32"] = headerCrc32;
  j["padding"] = padding;
  return j;
}

void Header::print() const noexcept {
  std::cout << this->toJson().dump(2) << "\n";
}

//////////////////////////////////////////////////////////////////////////
// Entry helpers
//////////////////////////////////////////////////////////////////////////

uint32_t Entry::computePayloadCrc() const noexcept {
  uint32_t crc = 0;
  // Include entry metadata fields for complete integrity coverage
  crc = utils::crc32(&type, sizeof(type), crc);
  crc = utils::crc32(&version, sizeof(version), crc);
  crc = utils::crc32(&lsn, sizeof(lsn), crc);
  crc = utils::crc32(&txid, sizeof(txid), crc);
  // Include dimension
  crc = utils::crc32(&dimension, sizeof(dimension), crc);
  // Include vectorID (all 128 bytes)
  crc = utils::crc32(vectorID, wal::kVectorIDSize, crc);
  // Include embedding
  if (!embedding.empty()) {
    crc = utils::crc32(embedding.data(), embedding.size() * sizeof(float), crc);
  }
  return crc;
}

uint32_t Entry::computeHeaderCrc() const noexcept {
  uint32_t crc = 0;
  crc = utils::crc32(&type, sizeof(type), crc);
  crc = utils::crc32(&version, sizeof(version), crc);
  crc = utils::crc32(&lsn, sizeof(lsn), crc);
  crc = utils::crc32(&txid, sizeof(txid), crc);
  return crc;
}

utils::json Entry::toJson() const {
  json j = json::object();
  std::string typeStr;
  switch (type) {
  case OperationType::COMMIT_TXN:
    typeStr = "COMMIT_TXN";
    break;
  case OperationType::ABORT_TXN:
    typeStr = "ABORT_TXN";
    break;
  case OperationType::INSERT:
    typeStr = "INSERT";
    break;
  case OperationType::DELETE:
    typeStr = "DELETE";
    break;
  case OperationType::UPDATE:
    typeStr = "UPDATE";
    break;
  case OperationType::BATCH_INSERT:
    typeStr = "BATCH_INSERT";
    break;
  default:
    typeStr = "INVALID";
    break;
  }
  j["type"] = typeStr;
  j["lsn"] = lsn;
  j["txid"] = txid;
  j["vectorId"] = getVectorID();
  j["dimension"] = dimension;
  j["embedding"] = embedding;
  return j;
}

void Entry::print() const noexcept { std::cout << toJson().dump(2) << "\n"; }

//////////////////////////////////////////////////////////////////////////
// Entry Builder
//////////////////////////////////////////////////////////////////////////

Result<Entry> EntryBuilder::buildInsert(const std::string& vectorID,
                                         uint32_t dimension,
                                         const std::vector<float>& embedding) {
  if (vectorID.size() > kMaxVectorIDSize) {
    return Status(StatusCode::kInvalidArgument,
                  "Vector ID exceeds maximum length of " +
                      std::to_string(kMaxVectorIDSize) + " bytes");
  }

  Entry entry{
      .type = OperationType::INSERT,
      .version = 1,
      .lsn = lsn_++,
      .txid = txid_++,
      .headerCRC = 0,
      .payloadLength = 0,
      .dimension = dimension,
      .padding = 0,
      .embedding = embedding,
      .payloadCRC = 0};

  auto status = entry.setVectorID(vectorID);
  if (!status.ok()) {
    return status;
  }

  entry.headerCRC = entry.computeHeaderCrc();
  entry.payloadCRC = entry.computePayloadCrc();
  entry.payloadLength = entry.computePayloadLength();

  return entry;
}

Result<Entry> EntryBuilder::buildDelete(const std::string& vectorID) {
  if (vectorID.size() > kMaxVectorIDSize) {
    return Status(StatusCode::kInvalidArgument,
                  "Vector ID exceeds maximum length of " +
                      std::to_string(kMaxVectorIDSize) + " bytes");
  }

  Entry entry{
      .type = OperationType::DELETE,
      .version = 1,
      .lsn = lsn_++,
      .txid = txid_++,
      .headerCRC = 0,
      .payloadLength = 0,
      .dimension = 0,
      .padding = 0,
      .embedding = {},
      .payloadCRC = 0};

  auto status = entry.setVectorID(vectorID);
  if (!status.ok()) {
    return status;
  }

  entry.headerCRC = entry.computeHeaderCrc();
  entry.payloadCRC = entry.computePayloadCrc();
  entry.payloadLength = entry.computePayloadLength();

  return entry;
}

//////////////////////////////////////////////////////////////////////////
// Protocol: Header
//////////////////////////////////////////////////////////////////////////

Result<Header> ParseHeader(BinaryReader &r) {
  Header h;
  if (!r.read(h.magic))
    return Status(StatusCode::kBadHeader, "Failed to read WAL header magic");

  if (h.magic != kWalMagic)
    return Status(StatusCode::kBadHeader, "Invalid WAL magic number");

  if (!r.read(h.version))
    return Status(StatusCode::kBadHeader, "Failed to read WAL header version");

  if (!r.read(h.flags))
    return Status(StatusCode::kBadHeader, "Failed to read WAL header flags");

  if (!r.read(h.creationTime))
    return Status(StatusCode::kBadHeader,
                  "Failed to read WAL header creationTime");

  if (!r.read(h.headerCrc32))
    return Status(StatusCode::kBadHeader,
                  "Failed to read WAL header headerCrc32");

  if (!r.read(h.padding))
    return Status(StatusCode::kBadHeader, "Failed to read WAL header padding");

  // Validate header CRC after reading all fields
  Status validationStatus = IsHeaderValid(h);
  if (!validationStatus.ok()) {
    return validationStatus;
  }

  return h;
}

Status WriteHeader(const Header &h, BinaryWriter &w) {
  w.write(h.magic);
  w.write(h.version);
  w.write(h.flags);
  w.write(h.creationTime);
  w.write(h.headerCrc32);
  w.write(h.padding);
  if (!w.good()) {
    return Status(StatusCode::kIoError, "Failed to write WAL header");
  }
  return OkStatus();
}

Status IsHeaderValid(const Header &h) noexcept {
  if (h.magic != kWalMagic) {
    return Status(StatusCode::kBadHeader, "Invalid WAL magic number");
  }
  uint32_t computedCrc = utils::crc32((const void *)&h, offsetof(Header, headerCrc32));
  if (computedCrc != h.headerCrc32) {
    return Status(StatusCode::kChecksumMismatch, "Header CRC32 mismatch");
  }
  return OkStatus();
}

//////////////////////////////////////////////////////////////////////////
// Protocol: Entry
//////////////////////////////////////////////////////////////////////////

Result<Entry> ParseEntry(BinaryReader &r) {
  const std::streampos startPos = r.tell();

  Entry e;

  if (!r.read(e.type) || !r.read(e.version) || !r.read(e.lsn) ||
      !r.read(e.txid)) {
    return Status(StatusCode::kIoError, "Failed to read entry header fields");
  }

  // Validate OperationType enum to prevent undefined behavior from corrupt data
  uint16_t typeValue = static_cast<uint16_t>(e.type);
  if (typeValue < kMinOperationType || typeValue > kMaxOperationType) {
    return Status(StatusCode::kBadRecord, "Invalid operation type");
  }

  if (!r.read(e.headerCRC) || !r.read(e.payloadLength) ||
      !r.read(e.vectorID) ||
      !r.read(e.dimension) || !r.read(e.padding)) {
    e.print();
    return Status(StatusCode::kIoError, "Failed to read entry metadata fields");
  }

  // Validate dimension before allocating memory to prevent memory exhaustion
  if (e.dimension > kMaxDimension) {
    return Status(StatusCode::kBadRecord,
                  "Dimension exceeds maximum allowed: " + std::to_string(e.dimension));
  }

  // Validate remaining file bytes can actually hold the embedding data.
  // This prevents OOM from a corrupt WAL declaring large dimensions with
  // only a few bytes of actual data.
  if (e.dimension > 0) {
    const size_t embeddingBytes = static_cast<size_t>(e.dimension) * sizeof(float);
    std::streampos curPos = r.tell();
    r.seek(0, std::ios::end);
    std::streampos endPos = r.tell();
    r.seek(curPos, std::ios::beg);
    if (endPos - curPos < static_cast<std::streamoff>(embeddingBytes + sizeof(uint32_t))) {
      return Status(StatusCode::kBadRecord,
                    "WAL entry claims " + std::to_string(embeddingBytes) +
                    " bytes for embedding but only " +
                    std::to_string(static_cast<size_t>(endPos - curPos)) + " bytes remain");
    }
  }

  e.embedding.resize(e.dimension);
  if (!r.read(e.embedding)) {
    return Status(StatusCode::kIoError, "Failed to read entry embedding data");
  }

  if (!r.read(e.payloadCRC)) {
    return Status(StatusCode::kIoError, "Failed to read entry payload CRC");
  }

  const std::streampos endPos = r.tell();
  if (endPos <= startPos) {
    return Status(StatusCode::kCorruption,
                  "no forward progress while reading WAL entry");
  }

  uint32_t computedHeaderCrc = e.computeHeaderCrc();
  if (e.headerCRC != computedHeaderCrc) {
    return Status(StatusCode::kChecksumMismatch,
                  "Header CRC mismatch: stored=" + std::to_string(e.headerCRC) +
                      ", computed=" + std::to_string(computedHeaderCrc));
  }

  uint32_t computedPayloadCrc = e.computePayloadCrc();
  if (e.payloadCRC != computedPayloadCrc) {
    return Status(StatusCode::kChecksumMismatch, "Payload CRC mismatch");
  }

  if (e.dimension != e.embedding.size()) {
    return Status(StatusCode::kBadRecord, "embedding dimension mismatch");
  }

  return e;
}

/// Writes an Entry to the binary stream.
///
/// Note: This function computes headerCRC, payloadLength, and payloadCRC
/// on-the-fly from the Entry's data. The corresponding fields in the Entry
/// struct (e.headerCRC, e.payloadLength, e.payloadCRC) are ignored.
/// Callers do not need to pre-compute these values.
Status WriteEntry(const Entry &e, BinaryWriter &w) {
  assert(e.dimension == e.embedding.size());
  w.write(e.type);
  w.write(e.version);
  w.write(e.lsn);
  w.write(e.txid);
  w.write(e.computeHeaderCrc());
  w.write(e.computePayloadLength());
  w.write(e.vectorID);
  w.write(e.dimension);
  w.write(e.padding);
  w.write(e.embedding);
  w.write(e.computePayloadCrc());
  if (!w.good()) {
    return Status(StatusCode::kIoError, "Failed to write WAL entry");
  }
  return OkStatus();
}

Status IsEntryValid(const Entry &e) noexcept {
  if (e.dimension != e.embedding.size()) {
    return Status(StatusCode::kBadRecord, "embedding dimension mismatch");
  }
  uint32_t computedHeaderCrc = e.computeHeaderCrc();
  if (e.headerCRC != computedHeaderCrc) {
    return Status(StatusCode::kChecksumMismatch, "Header CRC mismatch");
  }
  uint32_t computedPayloadCrc = e.computePayloadCrc();
  if (e.payloadCRC != computedPayloadCrc) {
    return Status(StatusCode::kChecksumMismatch, "Payload CRC mismatch");
  }
  return OkStatus();
}


Result<BinaryReader> OpenBinaryReader(const std::filesystem::path &dir,
                                      const std::string &filename) {
  namespace fs = std::filesystem;
  if (!fs::exists(dir)) {
    return Status(StatusCode::kNotFound, "WAL directory does not exist");
  }

  if (!fs::is_directory(dir)) {
    return Status(StatusCode::kNotFound,
                  "WAL path exists but is not a directory");
  }

  fs::path filePath = dir / filename;

  auto pWalFile = std::make_unique<std::ifstream>(filePath, std::ios::in | std::ios::binary);

  if (!pWalFile->is_open()) {
    return Status(StatusCode::kIoError, "Failed to open WAL file");
  }

  return BinaryReader(std::move(pWalFile));
}

Result<BinaryWriter> OpenBinaryWriter(const std::filesystem::path &dir,
                                      const std::string &filename,
                                      bool append) {
  namespace fs = std::filesystem;

  if (!fs::exists(dir)) {
    try {
      fs::create_directories(dir);
    } catch (const std::exception &e) {
      return Status(StatusCode::kIoError,
                    "Failed to create WAL directory: " + std::string(e.what()));
    }
  } else if (!fs::is_directory(dir)) {
    return Status(StatusCode::kIoError,
                  "WAL path exists but is not a directory");
  }

  fs::path p = dir / filename;

  std::ios::openmode mode = std::ios::out | std::ios::binary |
                            (append ? std::ios::app : std::ios::trunc);

  auto ofs = std::make_unique<std::ofstream>(p, mode);
  if (!ofs->is_open()) {
    return Status(StatusCode::kIoError, "failed to open WAL file");
  }

  return BinaryWriter(std::move(ofs));
}

Result<Header> LoadHeader(const std::filesystem::path &dir,
                          const std::string &filename) {

  Result<BinaryReader> res = OpenBinaryReader(dir, filename);

  if (!res.ok()) {
    return res.status();
  }

  BinaryReader &r = res.value();
  r.seek(0, std::ios::end);
  size_t fileSize = r.tell();
  if (fileSize < kHeaderWireSize) {
    return Status(StatusCode::kBadHeader,
                  "WAL file is too small to contain a valid header");
  }
  r.seek(0, std::ios::beg);
  return ParseHeader(r);
}

Result<WALContents> ReadAll(const std::filesystem::path& walFilePath) {
  namespace fs = std::filesystem;

  if (!fs::exists(walFilePath)) {
    return Status(StatusCode::kNotFound, "WAL file does not exist");
  }

  auto fileSize = fs::file_size(walFilePath);
  if (fileSize == 0) {
    return Status(StatusCode::kEof, "File is empty");
  }

  auto ifs = std::make_unique<std::ifstream>(walFilePath, std::ios::in | std::ios::binary);
  if (!ifs->is_open()) {
    return Status(StatusCode::kIoError, "Failed to open WAL file");
  }

  BinaryReader reader(std::move(ifs));

  // Parse header
  Result<Header> headerResult = ParseHeader(reader);
  if (!headerResult.ok()) {
    return headerResult.status();
  }

  // Parse entries
  reader.seek(0, std::ios::end);
  const std::streampos fileEnd = reader.tell();
  reader.seek(kHeaderWireSize, std::ios::beg);

  if (!reader.good()) {
    return Status(StatusCode::kEof, "Failed to seek past header");
  }

  std::vector<Entry> entries;
  while (reader.good() && reader.tell() < fileEnd) {
    auto curPos = reader.tell();
    Result<Entry> entryResult = ParseEntry(reader);
    if (!entryResult.ok()) {
      // If no progress was made, we're stuck - return what we have
      if (reader.tell() == curPos) {
        break;
      }
      // Otherwise, corruption detected - fail fast
      return entryResult.status();
    }
    entries.push_back(std::move(entryResult.value()));
  }

  return WALContents{.header = std::move(headerResult.value()), .entries = std::move(entries)};
}

//////////////////////////////////////////////////////////////////////////
// WAL orchestration
//////////////////////////////////////////////////////////////////////////

WAL::WAL(std::filesystem::path walDir, std::unique_ptr<WALWriter>&& writer)
    : walDir_(std::move(walDir)), writer_(std::move(writer)) {}

WAL::~WAL() = default;

WAL::WAL(WAL&&) noexcept = default;
WAL& WAL::operator=(WAL&&) noexcept = default;

Result<WAL> WAL::open(std::filesystem::path walDir) {
  namespace fs = std::filesystem;

  // Create directory if needed
  if (!fs::exists(walDir)) {
    try {
      fs::create_directories(walDir);
    } catch (const std::exception& e) {
      return Status(StatusCode::kIoError,
                    "Failed to create WAL directory: " + std::string(e.what()));
    }
  }

  // Open the WAL file
  fs::path walFilePath = walDir / "db.wal";
  Result<WALWriter> writerResult = WALWriter::open(walFilePath);
  if (!writerResult.ok()) {
    return writerResult.status();
  }

  auto writer = std::make_unique<WALWriter>(std::move(writerResult.value()));
  return WAL(std::move(walDir), std::move(writer));
}

Status WAL::log(const Entry& entry) {
  if (!writer_) {
    return Status(StatusCode::kIoError, "WAL writer is not open");
  }
  return writer_->append(entry);
}

Status WAL::logDeferred(const Entry& entry) {
  if (!writer_) {
    return Status(StatusCode::kIoError, "WAL writer is not open");
  }
  return writer_->appendDeferred(entry);
}

Status WAL::sync() {
  if (!writer_) {
    return Status(StatusCode::kIoError, "WAL writer is not open");
  }
  return writer_->sync();
}

Status WAL::logBatch(std::span<const Entry> entries) {
  if (!writer_) {
    return Status(StatusCode::kIoError, "WAL writer is not open");
  }
  return writer_->appendBatch(entries);
}

Result<WALContents> WAL::readAll() const {
  namespace fs = std::filesystem;
  fs::path walFilePath = walDir_ / "db.wal";
  return ReadAll(walFilePath);
}

void WAL::print() const {
  Result<WALContents> contents = readAll();
  if (!contents.ok()) {
    std::cerr << contents.status().message() << "\n";
    return;
  }

  contents.value().header.print();

  const auto& entries = contents.value().entries;
  std::cout << "WAL Entries (" << entries.size() << "):\n";
  for (const auto& entry : entries) {
    entry.print();
  }
}

Status WAL::truncate() {
  namespace fs = std::filesystem;

  // Close writer
  writer_.reset();

  // Truncate file and reopen
  fs::path walFilePath = walDir_ / "db.wal";

  // Remove the file and sync directory to make removal durable
  if (fs::exists(walFilePath)) {
    fs::remove(walFilePath);
    utils::syncDir(walDir_.string());
  }

  // Reopen writer (will create fresh header)
  Result<WALWriter> writerResult = WALWriter::open(walFilePath);
  if (!writerResult.ok()) {
    return writerResult.status();
  }

  writer_ = std::make_unique<WALWriter>(std::move(writerResult.value()));
  return OkStatus();
}

Result<RecoveryReport> WAL::recover() {
  namespace fs = std::filesystem;

  fs::path walFile = walDir_ / "db.wal";

  RecoveryReport report;

  if (!fs::exists(walFile)) {
    return report;
  }

  auto fileSize = fs::file_size(walFile);
  if (fileSize == 0) {
    return report;
  }

  // Close writer before reading
  writer_.reset();

  auto ifs = std::make_unique<std::ifstream>(walFile, std::ios::in | std::ios::binary);
  if (!ifs->is_open()) {
    return Status(StatusCode::kIoError, "Failed to open WAL for recovery");
  }
  BinaryReader reader(std::move(ifs));

  // Validate header
  Result<Header> headerResult = ParseHeader(reader);
  if (!headerResult.ok()) {
    // Corrupt header - truncate entire file
    {
      std::ofstream ofs(walFile, std::ios::binary | std::ios::trunc);
      ofs.close();
    }
    utils::syncFile(walFile.string());
    utils::syncDir(walFile.parent_path().string());
    report.discardedBytes = fileSize;
    report.truncationPerformed = true;
    // Reopen writer
    Result<WALWriter> writerResult = WALWriter::open(walFile);
    if (!writerResult.ok()) {
      return writerResult.status();
    }
    writer_ = std::make_unique<WALWriter>(std::move(writerResult.value()));
    return report;
  }

  std::streampos lastValidPos = reader.tell();
  uint64_t validEntries = 0;

  while (reader.good()) {
    std::streampos entryStart = reader.tell();

    Result<Entry> entryResult = ParseEntry(reader);
    if (!entryResult.ok()) {
      if (!reader.good() && entryStart == lastValidPos) {
        break;
      }
      break;
    }
    validEntries++;
    lastValidPos = reader.tell();
  }

  report.validEntries = validEntries;

  if (lastValidPos < static_cast<std::streampos>(fileSize)) {
    report.discardedBytes = fileSize - lastValidPos;
    report.truncationPerformed = true;

    if (::truncate(walFile.string().c_str(), lastValidPos) != 0) {
      return Status(StatusCode::kIoError, "Failed to truncate WAL file");
    }
    if (!utils::syncFile(walFile.string())) {
      return Status(StatusCode::kIoError, "Failed to fsync WAL after truncation");
    }
    utils::syncDir(walFile.parent_path().string());
  }

  // Reopen writer in append mode
  Result<WALWriter> writerResult = WALWriter::open(walFile);
  if (!writerResult.ok()) {
    return writerResult.status();
  }
  writer_ = std::make_unique<WALWriter>(std::move(writerResult.value()));

  return report;
}

} // namespace arrow::wal
