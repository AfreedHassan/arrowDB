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

struct Header {
  uint32_t magic = 0x41574C01;
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
  explicit WAL(std::filesystem::path db_path);
  ~WAL();

  [[nodiscard]] Result<Header> loadHeader(std::string path_param = "") const;
  [[nodiscard]] Status writeHeader(const Header& header,
                                   std::string path_param = "") const;

  [[nodiscard]] Result<std::vector<Entry>> readAll(std::string path_param = "") const;
  [[nodiscard]] Result<Entry> readNext(BinaryReader& r) const;
  [[nodiscard]] Status log(const Entry& entry, std::string path_param = "",
                           bool reset = false);

  void print() const;
  Status ValidateOrCreatePath(const std::filesystem::path& base_path, const std::string& path_param, std::filesystem::path& out_path) const;

 private:
  std::filesystem::path walPath_;
  uint64_t offset_ = 1;
};

}  // namespace wal
}  // namespace arrow

#endif  // ARROW_WAL_H
