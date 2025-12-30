#ifndef TYPES_H	
#define TYPES_H

#include <cstdint>
#include <variant>
#include <unordered_map>
#include <expected>

namespace arrow {
	using VectorID = uint64_t;
	using Timestamp = uint64_t;

	// Metadata value types
	using MetadataValue = std::variant<int64_t, double, std::string, bool>;
	using Metadata = std::unordered_map<std::string, MetadataValue>;

	// Error handling
	enum class ErrorCode { 
		DimensionMismatch, NotFound, DuplicateID, StorageError 
	};

	template<typename T>
	using Result = std::expected<T, ErrorCode>;

}

#endif // TYPES_H
