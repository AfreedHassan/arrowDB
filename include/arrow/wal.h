#ifndef ARROW_WAL_H
#define ARROW_WAL_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "types.h"
#include "utils/binary.h"
#include "utils/result.h"

namespace arrow {
namespace wal {

using Status = utils::Status;
using StatusCode = utils::StatusCode;
template <typename T> using Result = utils::Result<T>;

/// Magic number for WAL file format: "AWL" + version byte (0x01)
static constexpr uint32_t kWalMagic = 0x41574C01;

/// Maximum allowed embedding dimension to prevent memory exhaustion attacks
static constexpr uint32_t kMaxDimension = 65536;

/// Minimum valid OperationType enum value
static constexpr uint16_t kMinOperationType = 1;

/// Maximum valid OperationType enum value
static constexpr uint16_t kMaxOperationType = 6;

struct Header {
  uint32_t magic = kWalMagic;
  uint16_t version = 1;
  uint16_t flags = 0;
  uint64_t creationTime = 0;
  uint32_t headerCrc32 = 0;
  uint32_t padding = 0;

  uint32_t computeCrc32() const noexcept;
  utils::json toJson() const;
  void print() const noexcept;
};

static_assert(sizeof(Header) >= 24, "Header wire size must be >= 24 bytes");
static constexpr std::size_t kHeaderWireSize = 24;

enum class OperationType : uint16_t {
  COMMIT_TXN = 1,
  ABORT_TXN = 2,
  INSERT = 3,
  DELETE = 4,
  UPDATE = 5,
  BATCH_INSERT = 6
};

struct Entry {
  OperationType type;
  uint16_t version;
  uint64_t lsn;
  uint64_t txid;
  uint32_t headerCRC;
  uint32_t payloadLength = 0;
  VectorID vectorID = 0;
  uint32_t dimension = 0;
  uint8_t padding;
  std::vector<float> embedding;
  uint32_t payloadCRC = 0;

  uint32_t computePayloadLength() const noexcept {
    return static_cast<uint32_t>(embedding.size() * sizeof(float));
  }
  uint32_t computePayloadCrc() const noexcept;
  uint32_t computeHeaderCrc() const noexcept;
  utils::json toJson() const;
  void print() const noexcept;
};

//////////////////////////////////////////////////////////////////////////
// Domain helpers / protocol: free functions only
//////////////////////////////////////////////////////////////////////////

Result<Header> ParseHeader(BinaryReader& r);
Status WriteHeader(const Header& h, BinaryWriter& w);
Status IsHeaderValid(const Header& h) noexcept;

Result<Entry> ParseEntry(BinaryReader& r);
Status WriteEntry(const Entry& e, BinaryWriter& w);
Status IsEntryValid(const Entry& e) noexcept;

//////////////////////////////////////////////////////////////////////////
// Filesystem helpers (domain namespace, not utils)
//////////////////////////////////////////////////////////////////////////

Result<BinaryReader> OpenBinaryReader(const std::filesystem::path& dir,
                                           const std::string& filename);
Result<BinaryWriter> OpenBinaryWriter(const std::filesystem::path& dir,
                                           const std::string& filename,
                                           bool append = true);

Result<Header> LoadHeader(const std::filesystem::path& dir,
                          const std::string& filename = "db.wal");

//////////////////////////////////////////////////////////////////////////
// WAL orchestration object (coordinator, no parsing internals)
//////////////////////////////////////////////////////////////////////////

class WAL {
 public:
  explicit WAL(std::filesystem::path dbPath);
  ~WAL();

  [[nodiscard]] Result<Header> loadHeader(const std::string& pathParam = "") const;
  [[nodiscard]] Status writeHeader(const Header& header,
                                   const std::string& pathParam = "") const;

  [[nodiscard]] Result<std::vector<Entry>> readAll(const std::string& pathParam = "") const;
  [[nodiscard]] Result<Entry> readNext(BinaryReader& r) const;
  [[nodiscard]] Status log(const Entry& entry, const std::string& pathParam = "",
                           bool reset = false);

  /// Log multiple entries in batch with single fsync.
  /// More efficient than calling log() multiple times (N-1 fewer fsyncs).
  [[nodiscard]] Status logBatch(const std::vector<Entry>& entries,
                                const std::string& pathParam = "");

  void print() const;

  /// Truncate WAL to header-only state (checkpoint operation).
  /// Creates a fresh WAL with only a header, discarding all entries.
  [[nodiscard]] Status truncate();

  Status ValidateOrCreatePath(const std::filesystem::path& basePath, const std::string& pathParam, std::filesystem::path& outPath) const;

 private:
  std::filesystem::path walPath_;
};

}  // namespace wal
}  // namespace arrow

#endif  // ARROW_WAL_H
