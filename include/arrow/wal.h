#ifndef WAL_H
#define WAL_H

#include "types.h"
#include "utils/binary.h"
#include <filesystem>
#include <string>

namespace arrow {
class WAL {
  using json = arrow::utils::json;

public:
  struct Header {
    uint32_t magic = 0x41574C01;
    uint16_t version = 1;
    uint16_t flags = 0;
    uint64_t creationTime = 0;
    uint32_t headerCrc32 = 0;
    uint32_t _padding = 0;

    Header();
    void write(BinaryWriter &w) const;
    void read(BinaryReader &r);
    void print() const;

    void computeHeaderCrc32();
    json toJson() const;
  };

  static constexpr size_t HEADERSIZE = 24;
  enum class Status {
    SUCCESS,
    FAILURE,
  };

  enum class OperationType : uint16_t {
    COMMIT_TXN = 1,
    ABORT_TXN = 2,
    INSERT = 3,
    DELETE = 4,
    UPDATE = 5,
    BATCH_INSERT = 6
  };

  struct Entry {
    OperationType type; // uint16_t
    uint16_t version;
    uint64_t lsn;
    uint64_t txid;
    uint32_t headerCrc;

    uint32_t payloadLength;
    VectorID vectorId;
    uint32_t dimension;
    uint8_t padding;
    std::vector<float> embedding;
    uint32_t payloadCrc;

    // Constructors
    Entry(OperationType t, uint64_t seq, uint64_t tid, VectorID vid, uint32_t dims, const std::vector<float> &v)
        : payloadLength(0), type(t), version(1), lsn(seq), txid(tid), headerCrc(0), vectorId(vid), dimension(dims), padding(0), embedding(v), payloadCrc(0) {
      assert(dims == v.size());
    }

    Entry(BinaryReader &r);
    Entry(std::ifstream &inFile);

    Status write(BinaryWriter &w);
    Status read(BinaryReader &r);
    std::string typeToString() const;
    json toJson() const;
    void print() const;

    inline uint32_t computePayloadLength() const {
      return (embedding.size() * sizeof(float)); 
    }

    uint32_t computeHeaderCrc() const;
    uint32_t computePayloadCrc() const;

  };

  std::variant<Header, Status> readHeader(std::string pathParam = "") const;
  std::variant<Header, Status> readHeader(std::ifstream &is) const;

  Status log(const Entry &entry, std::string pathParam = "",
             bool reset = false);
  using EntryPtr = std::unique_ptr<Entry>;
  std::variant<EntryPtr, Status> read(std::string pathParam = "");
  std::variant<std::vector<EntryPtr>, Status>
  readAll(std::string pathParam = "") const;
  void print() const;

  WAL();
  ~WAL() = default;

private:
  std::filesystem::path walPath_;
  uint64_t offset = 1;
};

} // namespace arrow

#endif // WAL_H

