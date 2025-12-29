#include <iostream>
#include <format>

/**
 * @brief Stream output operator for std::vector.
 * @tparam T The element type of the vector.
 * @param os The output stream.
 * @param vec The vector to output.
 * @return Reference to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
	os << std::format("{}", vec) << '\n';
	return os;
}

namespace utils {

}

