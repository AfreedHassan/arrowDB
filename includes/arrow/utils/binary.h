#ifndef BINARY_H
#define BINARY_H

#include <iostream>

namespace arrow {
class BinaryWriter {
  std::ostream& os;
public:
  explicit BinaryWriter(std::ostream& o) : os(o) {}

  template <typename T>
  void write(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
  }

	template<typename T>
  void write(const std::vector<T>& v) {
		static_assert(std::is_trivially_copyable_v<T>);
		size_t n = v.size();
		if (n > 0) {
			os.write(reinterpret_cast<const char*>(v.data()),
					n * sizeof(T));
		}
  }

  void writeString(std::string_view sv) {
    uint64_t size = sv.size();
    write(size);
    os.write(sv.data(), size);
  }
};

class BinaryReader {
  std::istream& is;
public:
  explicit BinaryReader(std::istream& i) : is(i) {}

  // Check if last read operation succeeded
  bool good() const { return is.good(); }
  bool fail() const { return is.fail(); }
  bool eof() const { return is.eof(); }

  template <typename T>
  bool read(T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return is.good() && is.gcount() == sizeof(T);
  }

	template <typename T>
  bool read(std::vector<T>& v) {
    static_assert(std::is_trivially_copyable_v<T>);
		size_t n = v.size();
		if (n > 0) {
      is.read(reinterpret_cast<char*>(v.data()),
              n * sizeof(T));
      return is.good() && static_cast<size_t>(is.gcount()) == n * sizeof(T);
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
      is.read(out.data(), size);
      if (!is.good() || static_cast<size_t>(is.gcount()) != size) {
        out.clear();
      }
    }
    return out;
  }
};
} // namespace arrow
#endif // BINARY_H
