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
  uint32_t magic = 0x01; 
  uint16_t version = 1;
  uint8_t endianness = 0;  // 0 = little-endian, 1 = big-endian (default to little-endian)
  uint32_t crc32 = 0;

	void write(BinaryWriter& w) const;
	void read(BinaryReader& r);
	void print() const;
	
	// Detect host endianness
	static uint8_t detectEndianness();
	// Check if file endianness matches host endianness
	bool isEndiannessCompatible() const;
};

  static constexpr size_t HEADERSIZE = sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t);
  static constexpr size_t FILECRC32SIZE = sizeof(uint32_t);  // CRC32 at end of file
  enum class Status {
    SUCCESS,
    FAILURE,
  };
  enum class EntryType {
    INSERT = 1 << 0,
    DELETE = 1 << 1,
    UPDATE = 1 << 2,
  };
/*
 *
 * struct Operation { 
    OperationType type;
    VectorID id;
		uint32_t dimension;
		std::vector<float> embedding;
  };
 * struct Entry {
       uint64_t offset;        // Offset in WAL file
       Operation operation;    // The operation (insert, delete, update)
       uint32_t crc32;        // CRC32 checksum of the entry data
 * }
 *
 *
 * 
 */
 

struct Entry {
    EntryType type;
    VectorID id;
		uint32_t dimension;
		std::vector<float> embedding;

    // Constructors
    Entry(EntryType t, VectorID vid, uint32_t dims, const std::vector<float>& v)
        : type(t), id(vid), dimension(dims), embedding(v) {
					assert(dims == v.size());
				}
    
    Entry(BinaryReader& r);
    Entry(std::ifstream& inFile);

    // Methods (declarations only, implementations in .cpp)
    Status write(BinaryWriter& w) const;
    Status read(BinaryReader& r);
    std::string typeToString() const;
    json toJson() const;
    void print() const;
    
    // Compute CRC32 over all entry data in a single pass
    uint32_t computeCrc32() const;
  };
  // Helper to load header from file
	std::variant<Header, Status> readHeader(std::string pathParam = "") const;
  std::variant<Header, Status> readHeader(std::ifstream& is) const;

  Status log(const Entry &entry, std::string pathParam = "", bool reset = false); 
	using EntryPtr = std::unique_ptr<Entry>;
  std::variant<EntryPtr, Status> read(std::string pathParam = ""); 
  std::variant<std::vector<EntryPtr>, Status> readAll(std::string pathParam = "") const;
	void print() const;

  WAL(); 
  ~WAL() = default;

private:
  std::filesystem::path walPath_;
	uint64_t offset=1;
};

} // namespace arrow

#endif // WAL_H
