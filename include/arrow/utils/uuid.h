#ifndef ARROW_UTILS_UUID_H
#define ARROW_UTILS_UUID_H

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <string>

namespace arrow::uuid {

struct uuid {
	std::array<uint8_t, 16> bytes;
};

inline std::string to_string(uuid id) {
	const char hex[] = "0123456789abcdef";
	std::string str(36, '-');

	for (size_t i = 0, idx = 0; i < 36; ++i) {
		if (i == 8 || i == 13 || i == 18 || i == 23) {
			continue;
		}
		str[i] = hex[id.bytes[idx] >> 4];
		str[++i] = hex[id.bytes[idx] & 0x0F];
		++idx;
	}
	return str;
}

inline std::string uuidv4() {
	static thread_local std::mt19937_64 gen(std::random_device{}());
	std::uniform_int_distribution<uint64_t> dist;

	alignas(uint64_t) uint8_t bytes[16];
	uint64_t high = dist(gen);
	uint64_t low = dist(gen);

	std::memcpy(bytes, &high, 8);
	std::memcpy(bytes + 8, &low, 8);

	bytes[6] = (bytes[6] & 0x0F) | 0x40;
	bytes[8] = (bytes[8] & 0x3F) | 0x80;

	uuid result;
	std::memcpy(result.bytes.data(), bytes, 16);
	return to_string(result);
}

inline std::string uuidv4(uint64_t seed) {
	std::mt19937_64 gen(seed);
	std::uniform_int_distribution<uint64_t> dist;

	alignas(uint64_t) uint8_t bytes[16];
	uint64_t high = dist(gen);
	uint64_t low = dist(gen);

	std::memcpy(bytes, &high, 8);
	std::memcpy(bytes + 8, &low, 8);

	bytes[6] = (bytes[6] & 0x0F) | 0x40;
	bytes[8] = (bytes[8] & 0x3F) | 0x80;

	uuid result;
	std::memcpy(result.bytes.data(), bytes, 16);
	return to_string(result);
}


inline bool operator==(uuid const& lhs, uuid const& rhs) noexcept {
	return lhs.bytes == rhs.bytes;
}

inline bool operator!=(uuid const& lhs, uuid const& rhs) noexcept {
	return !(lhs == rhs);
}

inline bool operator<(uuid const& lhs, uuid const& rhs) noexcept {
	return lhs.bytes < rhs.bytes;
}

}

namespace std {

template <>
struct hash<arrow::uuid::uuid> {
	using argument_type = arrow::uuid::uuid;
	using result_type = std::size_t;

	result_type operator()(argument_type const& uuid) const noexcept {
		uint64_t high =
			static_cast<uint64_t>(uuid.bytes[0]) << 56 |
			static_cast<uint64_t>(uuid.bytes[1]) << 48 |
			static_cast<uint64_t>(uuid.bytes[2]) << 40 |
			static_cast<uint64_t>(uuid.bytes[3]) << 32 |
			static_cast<uint64_t>(uuid.bytes[4]) << 24 |
			static_cast<uint64_t>(uuid.bytes[5]) << 16 |
			static_cast<uint64_t>(uuid.bytes[6]) << 8 |
			static_cast<uint64_t>(uuid.bytes[7]);

		uint64_t low =
			static_cast<uint64_t>(uuid.bytes[8]) << 56 |
			static_cast<uint64_t>(uuid.bytes[9]) << 48 |
			static_cast<uint64_t>(uuid.bytes[10]) << 40 |
			static_cast<uint64_t>(uuid.bytes[11]) << 32 |
			static_cast<uint64_t>(uuid.bytes[12]) << 24 |
			static_cast<uint64_t>(uuid.bytes[13]) << 16 |
			static_cast<uint64_t>(uuid.bytes[14]) << 8 |
			static_cast<uint64_t>(uuid.bytes[15]);

		if constexpr (sizeof(result_type) > 4) {
			return static_cast<result_type>(high ^ low);
		} else {
			uint64_t hash64 = high ^ low;
			return static_cast<result_type>(
				static_cast<uint32_t>(hash64 >> 32) ^ static_cast<uint32_t>(hash64)
			);
		}
	}
};

}

#endif
