// Copyright 2025 ArrowDB
#include "arrow/wal.h"
#include "arrow/utils/crc32.h"
#include "arrow/utils/filesync.h"
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
  return utils::crc32((const void *)this, 16);
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
  j["vectorId"] = vectorID;
  j["dimension"] = dimension;
  j["embedding"] = embedding;
  return j;
}

void Entry::print() const noexcept { std::cout << toJson().dump(2) << "\n"; }

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
  return OkStatus();
}

Status IsHeaderValid(const Header &h) noexcept {
  if (h.magic != kWalMagic) {
    return Status(StatusCode::kBadHeader, "Invalid WAL magic number");
  }
  uint32_t computedCrc = utils::crc32((const void *)&h, 16);
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

  if (!r.read(e.headerCRC) || !r.read(e.payloadLength) || !r.read(e.vectorID) ||
      !r.read(e.dimension) || !r.read(e.padding)) {
    e.print();
    return Status(StatusCode::kIoError, "Failed to read entry metadata fields");
  }

  // Validate dimension before allocating memory to prevent memory exhaustion
  if (e.dimension > kMaxDimension) {
    return Status(StatusCode::kBadRecord,
                  "Dimension exceeds maximum allowed: " + std::to_string(e.dimension));
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

//////////////////////////////////////////////////////////////////////////
// Filesystem helpers
//////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////
// WAL orchestration
//////////////////////////////////////////////////////////////////////////

WAL::WAL(std::filesystem::path dbPath) : walPath_(std::move(dbPath)) {
  namespace fs = std::filesystem;
  if (!fs::exists(walPath_)) {
    fs::create_directories(walPath_);
    std::cout << "Created WAL directory at " << fs::absolute(walPath_) << "\n";
  }
}

WAL::~WAL() = default;

Result<Header> WAL::loadHeader(const std::string& pathParam) const {
  namespace fs = std::filesystem;
  fs::path path = walPath_;

  if (!pathParam.empty()) {
    path = fs::path(pathParam);
  }

  return LoadHeader(path);
}

Status WAL::writeHeader(const Header &header, const std::string& pathParam) const {
  namespace fs = std::filesystem;
  fs::path path = walPath_;

  if (!pathParam.empty()) {
    path = fs::path(pathParam);
  }

  const std::string filename = "db.wal";
  Result<BinaryWriter> res = OpenBinaryWriter(path, filename, false);
  if (!res.ok()) {
    return res.status();
  }
  BinaryWriter &w = res.value();

  Status writeStatus = WriteHeader(header, w);
  if (!writeStatus.ok()) {
    return writeStatus;
  }

  w.flush();
  utils::syncFile((path / filename).string());

  return OkStatus();
}

Status WAL::log(const Entry &entry, const std::string& pathParam, bool reset) {
  namespace fs = std::filesystem;
  fs::path path = walPath_;
  const std::string filename = "db.wal";

  if (!pathParam.empty()) {
    path = pathParam;
  }

  Result<BinaryWriter> res = OpenBinaryWriter(path, filename, !reset);
  if (!res.ok()) {
    return res.status();
  }
  BinaryWriter &w = res.value();

  if (reset) {
    Header header;
    header.magic = kWalMagic;
    header.creationTime = time(nullptr);
    header.headerCrc32 = header.computeCrc32();
    header.padding = 2;
    Status writeStatus = WriteHeader(header, w);
    if (!writeStatus.ok()) {
      return Status(StatusCode::kIoError, "Failed to write WAL header to file");
    }
  }

  Status writeStatus = WriteEntry(entry, w);
  if (!writeStatus.ok()) {
    return Status(StatusCode::kIoError, "Failed to write WAL entry to file");
  }

  w.flush();
  utils::syncFile((path / filename).string());
  return OkStatus();
}

Status WAL::logBatch(const std::vector<Entry>& entries, const std::string& pathParam) {
  namespace fs = std::filesystem;
  fs::path path = walPath_;
  const std::string filename = "db.wal";

  if (!pathParam.empty()) {
    path = pathParam;
  }

  // Check if we need to write a header (file doesn't exist yet)
  fs::path filePath = path / filename;
  bool needsHeader = !fs::exists(filePath);

  Result<BinaryWriter> res = OpenBinaryWriter(path, filename, !needsHeader);
  if (!res.ok()) {
    return res.status();
  }
  BinaryWriter& w = res.value();

  // Write header if needed (first time creating the file)
  if (needsHeader) {
    Header header;
    header.magic = kWalMagic;
    header.creationTime = time(nullptr);
    header.headerCrc32 = header.computeCrc32();
    header.padding = 0;
    Status headerStatus = WriteHeader(header, w);
    if (!headerStatus.ok()) {
      return headerStatus;
    }
  }

  // Write all entries
  for (const Entry& entry : entries) {
    Status writeStatus = WriteEntry(entry, w);
    if (!writeStatus.ok()) {
      return writeStatus;
    }
  }

  // Single flush and fsync for entire batch
  w.flush();
  utils::syncFile((path / filename).string());
  return OkStatus();
}

Result<Entry> WAL::readNext(BinaryReader &r) const { return ParseEntry(r); }

Result<std::vector<Entry>> WAL::readAll(const std::string& pathParam) const {
  namespace fs = std::filesystem;
  fs::path path = walPath_;

  if (!pathParam.empty()) {
    path = fs::path(pathParam);
  }
  const std::string filename = "db.wal";

  Result<BinaryReader> res = OpenBinaryReader(path, filename);
  if (!res.ok()) {
    return res.status();
  }
  BinaryReader &r = res.value();

  fs::path filePath = path / filename;
  // Use filesystem for reliable file size detection
  auto fileSize = fs::file_size(filePath);

  if (fileSize == 0) {
    return Status(StatusCode::kEof, "File is empty");
  }

  r.seek(0, std::ios::end);
  const std::streampos fileEnd = r.tell();
  r.seek(0, std::ios::beg);

  Result<Header> resHeader = ParseHeader(r);
  if (!resHeader.ok()) {
    return resHeader.status();
  }

  if (!r.good()) {
    return Status(StatusCode::kEof, "Failed to seek past header");
  }
  std::vector<Entry> entries;
  // Parse entries - fail fast on corruption rather than trying to recover
  while (r.good() && r.tell() < fileEnd) {
    auto curPos = r.tell();
    Result<Entry> resEntry = ParseEntry(r);
    if (!resEntry.ok()) {
      // If no progress was made, we're stuck - return what we have
      if (r.tell() == curPos) {
        break;
      }
      // Otherwise, corruption detected - fail fast
      return resEntry.status();
    }
    entries.push_back(std::move(resEntry.value()));
  }
  return entries;
}

void WAL::print() const {
  Result<Header> h = loadHeader();

  if (!h.ok()) {
    std::cerr << h.status().message() << "\n";
    return;
  }

  auto entries = readAll();
  if (!entries.ok()) {
    std::cerr << entries.status().message() << "\n";
    return;
  }

  h.value().print();

  const auto &entryVec = entries.value();
  std::cout << "WAL Entries (" << entryVec.size() << "):\n";
  for (const auto &entry : entryVec) {
    entry.print();
  }
}

Status WAL::truncate() {
  const std::string filename = "db.wal";

  // Open in truncate mode (not append)
  Result<BinaryWriter> res = OpenBinaryWriter(walPath_, filename, false);
  if (!res.ok()) {
    return res.status();
  }
  BinaryWriter &w = res.value();

  // Write fresh header
  Header header;
  header.magic = kWalMagic;
  header.creationTime = time(nullptr);
  header.headerCrc32 = header.computeCrc32();
  header.padding = 0;

  Status writeStatus = WriteHeader(header, w);
  if (!writeStatus.ok()) {
    return writeStatus;
  }

  w.flush();
  utils::syncFile((walPath_ / filename).string());

  return OkStatus();
}

} // namespace arrow::wal
