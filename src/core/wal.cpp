#include "arrow/wal.h"
#include "arrow/utils/crc32.h"
#include "arrow/utils/filesync.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <variant>
#include <vector>

namespace arrow {

static constexpr size_t FILECRC32SIZE = sizeof(uint32_t);

void WAL::Header::computeHeaderCrc32() {
  this->headerCrc32 =
      utils::crc32(reinterpret_cast<const void *>(this), 16);
}

WAL::Header::Header()
    : magic(0x41574C01), version(1), flags(0), creationTime(0), headerCrc32(0),
      _padding(0) {
  computeHeaderCrc32();
}

void WAL::Header::write(BinaryWriter &w) const {
  w.write(magic);
  w.write(version);
  w.write(flags);
  w.write(creationTime);
  w.write(headerCrc32);
  w.write(_padding);
}

void WAL::Header::read(BinaryReader &r) {
  if (!r.read(magic) || !r.read(version) || !r.read(flags) ||
      !r.read(creationTime) || !r.read(headerCrc32) || !r.read(_padding)) {
    magic = 0;
  }
}

utils::json WAL::Header::toJson() const {
  json j = json::object();
  j["magic"] = magic;
  j["version"] = version;
  j["flags"] = flags;
  j["creationTime"] = creationTime;
  j["headerCrc32"] = headerCrc32;
  return j;
}

void WAL::Header::print() const { std::cout << this->toJson().dump(2) << "\n"; }
WAL::Entry::Entry(BinaryReader &r) { read(r); }
WAL::Entry::Entry(std::ifstream &inFile) {
  BinaryReader r(inFile);
  read(r);
}

uint32_t WAL::Entry::computeHeaderCrc() const {
  uint32_t crc = 0;
  crc = utils::crc32(&type, sizeof(type), crc);
  crc = utils::crc32(&version, sizeof(version), crc);
  crc = utils::crc32(&lsn, sizeof(lsn), crc);
  crc = utils::crc32(&txid, sizeof(txid), crc);
  return crc;
}

uint32_t WAL::Entry::computePayloadCrc() const {
  uint32_t crc = 0;
  if (!embedding.empty()) {
    crc = utils::crc32(embedding.data(), embedding.size() * sizeof(float),
                            crc);
  }
  return crc;
}

WAL::Status WAL::Entry::write(BinaryWriter &w) {
  assert(dimension == embedding.size());
  w.write(type);
  w.write(version);
  w.write(lsn);
  w.write(txid);
  headerCrc = computeHeaderCrc();
  w.write(headerCrc);
  payloadLength = computePayloadLength();
  w.write(payloadLength);
  w.write(vectorId);
  w.write(dimension);
  w.write(padding);
  w.write(embedding);
  payloadCrc = computePayloadCrc();
  w.write(payloadCrc);
  return Status::SUCCESS;
}

WAL::Status WAL::Entry::read(BinaryReader &r) {
  if (!r.read(payloadLength) || !r.read(type) || !r.read(version) ||
      !r.read(lsn) || !r.read(txid)) {
    std::cerr << "Failed to read entry header fields\n";
    return Status::FAILURE;
  }

  if (!r.read(headerCrc) || !r.read(vectorId) || !r.read(dimension) ||
      !r.read(padding)) {
    std::cerr << "Failed to read entry metadata fields\n";
    return Status::FAILURE;
  }

  embedding.resize(dimension);
  if (!r.read(embedding)) {
    std::cerr << "Failed to read entry embedding data\n";
    return Status::FAILURE;
  }

  if (!r.read(payloadCrc)) {
    std::cerr << "Failed to read entry payload CRC\n";
    return Status::FAILURE;
  }

  uint32_t computedHeaderCrc = computeHeaderCrc();
  if (headerCrc != computedHeaderCrc) {
    std::cerr << "Header CRC mismatch: stored=" << headerCrc
              << ", computed=" << computedHeaderCrc << "\n";
    return Status::FAILURE;
  }

  uint32_t computedPayloadCrc = computePayloadCrc();
  if (payloadCrc != computedPayloadCrc) {
    std::cerr << "Payload CRC mismatch: stored=" << payloadCrc
              << ", computed=" << computedPayloadCrc << "\n";
    return Status::FAILURE;
  }

  return Status::SUCCESS;
}

std::string WAL::Entry::typeToString() const {
  switch (type) {
  case OperationType::COMMIT_TXN:
    return "COMMIT_TXN";
  case OperationType::ABORT_TXN:
    return "ABORT_TXN";
  case OperationType::INSERT:
    return "INSERT";
  case OperationType::DELETE:
    return "DELETE";
  case OperationType::UPDATE:
    return "UPDATE";
  case OperationType::BATCH_INSERT:
    return "BATCH_INSERT";
  default:
    return "INVALID OPERATION";
  }
}

utils::json WAL::Entry::toJson() const {
  json j = json::object();
  j["type"] = typeToString();
  j["lsn"] = lsn;
  j["txid"] = txid;
  j["vectorId"] = vectorId;
  j["dimension"] = dimension;
  j["embedding"] = embedding;
  return j;
}

void WAL::Entry::print() const { 
  std::cout << toJson().dump(2) << "\n"; 
}

WAL::Status WAL::log(const WAL::Entry &entry, std::string pathParam,
                     bool reset) {
  namespace fs = std::filesystem;
  fs::path path = walPath_;

  if (pathParam != "") {
    path = fs::path(pathParam);
  }

  if (!fs::exists(path)) {
    try {
      fs::create_directories(path);
    } catch (const std::exception &e) {
      std::cerr << "Failed to create WAL directory: " << path << " - "
                << e.what() << "\n";
      return Status::FAILURE;
    }
  } else if (!fs::is_directory(path)) {
    std::cerr << "WAL path exists but is not a directory: " << path << "\n";
    return Status::FAILURE;
  }
  const std::string filename = "db.wal";
  fs::path filePath = path / filename;

  if (reset) {
    std::ofstream walFile(filePath,
                          std::ios::out | std::ios::binary | std::ios::trunc);

    if (!walFile.is_open()) {
      std::cerr << "Failed to open WAL file: " << filePath << "\n";
      return Status::FAILURE;
    }

    BinaryWriter writer(walFile);

    Header header;
    header.creationTime = std::time(nullptr);
    header.computeHeaderCrc32();
    header.write(writer);

    Status writeStatus = const_cast<Entry &>(entry).write(writer);
    if (writeStatus == Status::FAILURE) {
      std::cerr << "Failed to write WAL entry to file: " << filePath << "\n";
      return Status::FAILURE;
    }

    utils::syncFile(walFile, filePath);
  } else {
    std::fstream walFile(filePath, std::ios::in | std::ios::out |
                                       std::ios::binary | std::ios::app);

    if (!walFile.is_open()) {
      std::cerr << "Failed to open WAL file: " << filePath << "\n";
      return Status::FAILURE;
    }

    walFile.seekp(0, std::ios::end);

    BinaryWriter writer(walFile);

    Status writeStatus = const_cast<Entry &>(entry).write(writer);
    if (writeStatus == Status::FAILURE) {
      std::cerr << "Failed to write WAL entry to file: " << filePath << "\n";
      return Status::FAILURE;
    }

    walFile.flush();
  }

  return Status::SUCCESS;
}

std::variant<WAL::Header, WAL::Status>
WAL::readHeader(std::string pathParam) const {
  namespace fs = std::filesystem;
  fs::path path = walPath_;

  if (pathParam != "") {
    path = fs::path(pathParam);
  }

  if (!fs::exists(path)) {
    std::cerr << "WAL directory does not exist: " << path << "\n";
    return Status::FAILURE;
  }

  if (!fs::is_directory(path)) {
    std::cerr << "WAL path exists but is not a directory: " << path << "\n";
    return Status::FAILURE;
  }

  const std::string filename = "db.wal";
  fs::path filePath = path / filename;
  std::ifstream walFile(filePath, std::ios::in | std::ios::binary);

  if (!walFile.is_open()) {
    std::cerr << "Failed to open WAL file: " << filePath << "\n";
    return Status::FAILURE;
  }

  walFile.seekg(0, std::ios::end);
  if (walFile.tellg() == 0) {
    Header defaultHeader;
    defaultHeader.creationTime = std::time(nullptr);
    defaultHeader.computeHeaderCrc32();
    return defaultHeader;
  }

  walFile.seekg(0, std::ios::beg);
  BinaryReader r(walFile);
  Header header;
  header.read(r);
  std::cout << "Reading default header\n:";

  if (!r.good() || header.magic == 0) {
    std::cerr << "Failed to read WAL header or header is invalid\n";
    return Status::FAILURE;
  }

  if (header.magic != 0x41574C01) {
    std::cerr << "Invalid WAL magic number: 0x" << std::hex << header.magic
              << std::dec << "\n";
    return Status::FAILURE;
  }

  uint32_t computedCrc =
      utils::crc32(reinterpret_cast<const void *>(&header), 16);
  if (header.headerCrc32 != computedCrc) {
    std::cerr << "Header CRC32 mismatch: stored=0x" << std::hex
              << header.headerCrc32 << ", computed=0x" << computedCrc
              << std::dec << "\n";
    return Status::FAILURE;
  }
  return header;
}

std::variant<WAL::Header, WAL::Status>
WAL::readHeader(std::ifstream &is) const {
  BinaryReader r(is);
  Header header;
  header.read(r);

  if (header.magic != 0x41574C01) {
    std::cerr << "Invalid WAL magic number: 0x" << std::hex << header.magic
              << std::dec << "\n";
    return Status::FAILURE;
  }

  uint32_t computedCrc =
      utils::crc32(reinterpret_cast<const void *>(&header), 16);
  if (header.headerCrc32 != computedCrc) {
    std::cerr << "Header CRC32 mismatch\n";
    return Status::FAILURE;
  }

  return header;
}

std::variant<WAL::EntryPtr, WAL::Status> WAL::read(std::string pathParam) {
  namespace fs = std::filesystem;
  fs::path path = walPath_;

  if (pathParam != "") {
    path = fs::path(pathParam);
  }

  if (!fs::exists(path) || !fs::is_directory(path)) {
    std::cerr << "WAL directory does not exist: " << path << "\n";
    return Status::FAILURE;
  }

  const std::string filename = "db.wal";
  fs::path filePath = path / filename;
  std::ifstream walFile(filePath, std::ios::in | std::ios::binary);

  if (!walFile.is_open()) {
    std::cerr << "Failed to open WAL file: " << filePath << "\n";
    return Status::FAILURE;
  }

  BinaryReader r(walFile);
  Header header;

  // Read header if file has content
  walFile.seekg(0, std::ios::end);
  size_t fileSize = walFile.tellg();

  // Calculate header size: magic(4) + version(2) + endianness(1) + crc32(4) =
  // 11 bytes

  if (fileSize == 0) {
    std::cerr << "WAL file is empty, no entries to read\n";
    return Status::FAILURE;
  }

  if (fileSize < HEADERSIZE) {
    std::cerr << "WAL file is too small to contain a valid header\n";
    return Status::FAILURE;
  }

  walFile.seekg(0, std::ios::beg);
  header.read(r);

  if (header.magic != 0x41574C01) {
    std::cerr << "Invalid WAL magic number: 0x" << std::hex << header.magic
              << std::dec << "\n";
    return Status::FAILURE;
  }

  size_t remainingSize = fileSize - HEADERSIZE;
  if (remainingSize == 0) {
    std::cerr << "WAL file contains only header, no entries\n";
    return Status::FAILURE;
  }

  EntryPtr pEntry = std::make_unique<Entry>(r);

  if (!r.good() && !walFile.eof()) {
    std::cerr << "Failed to read entry from WAL file\n";
    return Status::FAILURE;
  }

  if (pEntry->dimension != pEntry->embedding.size()) {
    std::cerr << "Entry dimension mismatch after read\n";
    return Status::FAILURE;
  }

  return pEntry;
}

std::variant<std::vector<WAL::EntryPtr>, WAL::Status>
WAL::readAll(std::string pathParam) const {
  namespace fs = std::filesystem;
  fs::path path = walPath_;

  if (pathParam != "") {
    path = fs::path(pathParam);
  }

  if (!fs::exists(path)) {
    std::cerr << "WAL directory does not exist: " << path << "\n";
    return Status::FAILURE;
  }

  if (!fs::is_directory(path)) {
    std::cerr << "WAL path exists but is not a directory: " << path << "\n";
    return Status::FAILURE;
  }

  const std::string &filename = "db.wal";
  fs::path filePath = path / filename;
  std::ifstream walFile(filePath, std::ios::in | std::ios::binary);

  if (!walFile.is_open()) {
    std::cerr << "Failed to open WAL file: " << filePath << "\n";
    return Status::FAILURE;
  }

  BinaryReader r(walFile);

  walFile.seekg(0, std::ios::end);
  size_t fileSize = walFile.tellg();
  Header header;

  if (fileSize == 0) {
    return std::vector<EntryPtr>{};
  }

  if (fileSize < HEADERSIZE) {
    std::cerr << "WAL file is too small to contain a valid header\n";
    return Status::FAILURE;
  }

  walFile.seekg(0, std::ios::beg);
  header.read(r);

  if (!r.good() || header.magic == 0) {
    std::cerr << "Failed to read WAL header or header is invalid\n";
    return Status::FAILURE;
  }

  if (header.magic != 0x41574C01) {
    std::cerr << "Invalid WAL magic number: 0x" << std::hex << header.magic
              << std::dec << "\n";
    return Status::FAILURE;
  }

  std::vector<EntryPtr> entries;

  walFile.seekg(HEADERSIZE, std::ios::beg);
  if (!walFile.good()) {
    std::cerr << "Failed to seek past header\n";
    return Status::FAILURE;
  }

  std::streampos lastPos = walFile.tellg();
  size_t consecutiveNoProgress = 0;

  while (walFile.good() && !walFile.eof()) {
    std::streampos currentPos = walFile.tellg();
    walFile.seekg(0, std::ios::end);
    std::streampos fileEnd = walFile.tellg();
    walFile.seekg(currentPos, std::ios::beg);

    if (currentPos >= fileEnd) {
      break;
    }

    if (walFile.peek() == EOF) {
      break;
    }

    EntryPtr pEntry = std::make_unique<Entry>(r);

    if (!r.good() && !walFile.eof()) {
      std::cerr << "Stream error while reading WAL entry\n";
      return Status::FAILURE;
    }

    if (pEntry->dimension != pEntry->embedding.size()) {
      std::cerr << "Entry dimension mismatch after read\n"
                << pEntry->toJson().dump(2) << "\n";
      continue;
    }

    currentPos = walFile.tellg();
    if (currentPos == lastPos) {
      consecutiveNoProgress++;
      if (consecutiveNoProgress > 10) {
        std::cerr << "No progress reading entries, possible infinite loop\n";
        return Status::FAILURE;
      }
    } else {
      consecutiveNoProgress = 0;
      lastPos = currentPos;
    }

    entries.push_back(std::move(pEntry));

    if (entries.size() > 1000000) {
      std::cerr << "Too many entries read, possible infinite loop\n";
      return Status::FAILURE;
    }
  }

  return entries;
}

void WAL::print() const {
  // Load and print header
  auto h = readHeader();
  if (std::holds_alternative<Header>(h)) {
    const auto &header = std::get<Header>(h);
    header.print();
  } else {
    std::cout << "WAL Header: (not available)\n\n";
  }

  // Print all entries
  auto entries = readAll();
  if (std::holds_alternative<std::vector<EntryPtr>>(entries)) {
    const auto &entryVec = std::get<std::vector<EntryPtr>>(entries);
    std::cout << "WAL Entries (" << entryVec.size() << "):\n";
    for (const auto &entry : entryVec) {
      entry->print();
    }
  } else {
    std::cout << "No entries found or error reading entries.\n";
  }
}

WAL::WAL() {
  namespace fs = std::filesystem;
  fs::path defaultPath = "wal";
  if (!fs::exists(defaultPath)) {
    fs::create_directories(defaultPath);
    std::cout << "Created WAL directory at " << fs::absolute(defaultPath)
              << "\n";
  }
  walPath_ = defaultPath;
}
} // namespace arrow
