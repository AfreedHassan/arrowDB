#ifndef BINARY_H
#define BINARY_H

#include <fstream>
#include <sstream>
#include <iostream>

namespace arrow {
class BinaryWriter {
  using pStream_t = std::unique_ptr<std::ostream>;
  pStream_t pOstream_;
public:
  explicit BinaryWriter(pStream_t&& pFile) : pOstream_(std::move(pFile)) {}

  BinaryWriter(const BinaryWriter&) = delete;
  BinaryWriter& operator=(const BinaryWriter&) = delete;
  BinaryWriter(BinaryWriter&& other) noexcept = default;
  BinaryWriter& operator=(BinaryWriter&& other) noexcept = default;
  
  ~BinaryWriter() = default;

  template <typename T>
  void write(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    pOstream_->write(reinterpret_cast<const char*>(&v), sizeof(T));
  }

	template<typename T>
  void write(const std::vector<T>& v) {
		static_assert(std::is_trivially_copyable_v<T>);
		size_t n = v.size();
		if (n > 0) {
			pOstream_->write(reinterpret_cast<const char*>(v.data()),
					n * sizeof(T));
		}
  }

  void writeString(std::string_view sv) {
    uint64_t size = sv.size();
    write(size);
    pOstream_->write(sv.data(), size);
  }

  void flush() { pOstream_->flush(); }
  
  // ONLY FOR TESTING
  std::string str() const {
    auto* ss = dynamic_cast<std::stringstream*>(pOstream_.get());
    return ss ? ss->str() : "";
  }
};

class BinaryReader {
  using pStream_t = std::unique_ptr<std::istream>;
  pStream_t pIstream_;
public:
  explicit BinaryReader(pStream_t&& pFile) : pIstream_(std::move(pFile)) {}

  // Check if last read operation succeeded
  bool good() const { return pIstream_->good(); }
  bool fail() const { return pIstream_->fail(); }
  bool eof() const { return pIstream_->eof(); }

  //forwards args to is.seekg
  template <typename... Args>
  void seek(Args&&... args) { pIstream_->seekg(std::forward<Args>(args)...); }
  std::istream::pos_type tell() const { return pIstream_->tellg(); }
  std::istream::int_type peek() const { return pIstream_->peek(); }

  template <typename T>
  bool read(T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    pIstream_->read(reinterpret_cast<char*>(&v), sizeof(T));
    return pIstream_->good() && pIstream_->gcount() == sizeof(T);
  }

	template <typename T>
  bool read(std::vector<T>& v) {
    static_assert(std::is_trivially_copyable_v<T>);
		size_t n = v.size();
		if (n > 0) {
      pIstream_->read(reinterpret_cast<char*>(v.data()),
              n * sizeof(T));
      return pIstream_->good() && static_cast<size_t>(pIstream_->gcount()) == n * sizeof(T);
    }
    return true;  // Empty vector is valid
  }

  std::string& read(std::string& out) {
    uint64_t size;
    if (!read(size)) {
      out.clear();
      return out;
    }
		out.resize(size);
    if (size > 0) {
      pIstream_->read(out.data(), size);
      if (!pIstream_->good() || static_cast<size_t>(pIstream_->gcount()) != size) {
        out.clear();
      }
    }
    return out;
  }
};
} // namespace arrow
#endif // BINARY_H
