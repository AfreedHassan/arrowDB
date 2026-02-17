#ifndef ARROW_WAL_H
#define ARROW_WAL_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "arrow/types.h"
#include "arrow/utils/utils.h"
#include "internal/binary.h"
#include "arrow/utils/result.h"

namespace arrow {
namespace wal {

using Status = utils::Status;
using StatusCode = utils::StatusCode;
template <typename T> using Result = utils::Result<T>;

/// Magic number for WAL file format: "AWL" + version byte (0x01)
static constexpr uint32_t kWalMagic = 0x41574C01;

/// Maximum allowed embedding dimension to prevent memory exhaustion attacks
static constexpr uint32_t kMaxDimension = 65536;

/// Wire/struct size of vector ID field (128 bytes fixed)
static constexpr uint32_t kVectorIDSize = 128;

/// Maximum usable string length for vector ID (127 bytes, reserving 1 for null terminator)
static constexpr uint32_t kMaxVectorIDSize = 127;

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
  char vectorID[kVectorIDSize] = {};  // null-padded fixed-size array
  uint32_t dimension = 0;
  uint8_t padding;
  std::vector<float> embedding;
  uint32_t payloadCRC = 0;

  uint32_t computePayloadLength() const noexcept {
    uint32_t embeddingLen = static_cast<uint32_t>(embedding.size() * sizeof(float));
    // kVectorIDSize bytes for vectorID + 4 bytes for dimension + 1 byte for padding + embedding
    return kVectorIDSize + 4 + 1 + embeddingLen;
  }
  uint32_t computePayloadCrc() const noexcept;
  uint32_t computeHeaderCrc() const noexcept;
  utils::json toJson() const;
  void print() const noexcept;

  std::string getVectorID() const {
    // Find null terminator or use full buffer
    size_t len = 0;
    while (len < kVectorIDSize && vectorID[len] != '\0') {
      ++len;
    }
    return std::string(vectorID, len);
  }

  utils::Status setVectorID(const std::string& id) {
    if (id.size() > kMaxVectorIDSize) {
      return utils::Status(utils::StatusCode::kInvalidArgument,
                           "Vector ID exceeds maximum length of " +
                               std::to_string(kMaxVectorIDSize) + " bytes");
    }
    std::memset(vectorID, 0, kVectorIDSize);
    std::memcpy(vectorID, id.data(), id.size());
    return utils::OkStatus();
  }
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
// Entry Builder - Factory for correctly-populated WAL entries
//////////////////////////////////////////////////////////////////////////

/// Builds WAL entries with correct CRCs and manages LSN/TXID counters.
class EntryBuilder {
 public:
  explicit EntryBuilder(uint64_t startLsn = 1, uint64_t startTxid = 1)
      : lsn_(startLsn), txid_(startTxid) {}

  /// Build an INSERT entry.
  Result<Entry> buildInsert(const std::string& vectorID,
                            uint32_t dimension,
                            const std::vector<float>& embedding);

  /// Build a DELETE entry (no embedding needed).
  Result<Entry> buildDelete(const std::string& vectorID);

  /// Current LSN (for persistence metadata).
  uint64_t currentLsn() const noexcept { return lsn_; }
  uint64_t currentTxid() const noexcept { return txid_; }

  /// Restore counters (e.g., after loading from disk).
  void restoreCounters(uint64_t lsn, uint64_t txid) {
    lsn_ = lsn;
    txid_ = txid;
  }

  /// Roll back N entries worth of counter increments (for batch failure).
  void rollbackCounters(uint64_t count) {
    if (lsn_ >= count) lsn_ -= count;
    if (txid_ >= count) txid_ -= count;
  }

 private:
  uint64_t lsn_;
  uint64_t txid_;
};

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
// ReadAll free function - one-shot WAL reading
//////////////////////////////////////////////////////////////////////////

/// Contents of a WAL file (header + entries).
struct WALContents {
  Header header;
  std::vector<Entry> entries;
};

/// Read entire WAL file in one shot. Opens reader, parses header, parses entries, closes.
Result<WALContents> ReadAll(const std::filesystem::path& walFilePath);

//////////////////////////////////////////////////////////////////////////
// Recovery report
//////////////////////////////////////////////////////////////////////////

/// Information about what happened during WAL recovery.
struct RecoveryReport {
  uint64_t validEntries = 0;
  uint64_t discardedBytes = 0;
  bool truncationPerformed = false;
};

//////////////////////////////////////////////////////////////////////////
// Forward declaration for WALWriter
//////////////////////////////////////////////////////////////////////////

class WALWriter;

//////////////////////////////////////////////////////////////////////////
// WAL orchestration object (coordinator, no parsing internals)
//////////////////////////////////////////////////////////////////////////

class WAL {
 public:
  /// Factory method to open a WAL. Binds to its path at construction.
  static Result<WAL> open(std::filesystem::path walDir);

  // Destructor must be defined in .cpp where WALWriter is fully defined
  ~WAL();

  // Move-only type (unique_ptr member)
  WAL(WAL&&) noexcept;
  WAL& operator=(WAL&&) noexcept;
  WAL(const WAL&) = delete;
  WAL& operator=(const WAL&) = delete;

  /// Log a single entry with immediate fsync (delegates to writer_.append()).
  [[nodiscard]] Status log(const Entry& entry);

  /// Log a single entry without fsync (delegates to writer_.appendDeferred()).
  /// Caller must call sync() later to ensure durability.
  [[nodiscard]] Status logDeferred(const Entry& entry);

  /// Sync all deferred WAL entries to disk.
  [[nodiscard]] Status sync();

  /// Log multiple entries in batch with single fsync (delegates to writer_.appendBatch()).
  /// More efficient than calling log() multiple times (N-1 fewer fsyncs).
  [[nodiscard]] Status logBatch(std::span<const Entry> entries);

  /// Read all entries from the WAL (delegates to ReadAll()).
  [[nodiscard]] Result<WALContents> readAll() const;

  /// Truncate WAL to header-only state (checkpoint operation).
  /// Closes writer, truncates file, reopens writer.
  [[nodiscard]] Status truncate();

  /// Recover from corruption by truncating to last valid entry.
  /// Returns report of what was recovered/discarded.
  [[nodiscard]] Result<RecoveryReport> recover();

  void print() const;

 private:
  explicit WAL(std::filesystem::path walDir, std::unique_ptr<WALWriter>&& writer);

  std::filesystem::path walDir_;
  std::unique_ptr<WALWriter> writer_;
};

}  // namespace wal
}  // namespace arrow

#endif  // ARROW_WAL_H
