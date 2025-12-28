#include <iostream>
#include <format>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
	os << std::format("{}", vec) << '\n';
	return os;
}

namespace utils {

}
