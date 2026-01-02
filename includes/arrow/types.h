#ifndef TYPES_H	
#define TYPES_H

#include <cstdint>
#include <nlohmann/json.hpp>
#include <variant>
#include <unordered_map>
#include <expected>

namespace arrow {
	using VectorID = uint64_t;
	using Timestamp = uint64_t;

	/**
	 * @brief Distance metrics for vector similarity computation.
	 */
	enum class DistanceMetric {
		Cosine,      ///< Cosine similarity (dot product of normalized vectors)
		L2,          ///< L2 (Euclidean) distance
		InnerProduct ///< Inner product (dot product)
	};

	/**
	 * @brief Data types for vector storage.
	 */
	enum class DataType {
		Int16,  ///< 16-bit signed integer
		Float16 ///< 16-bit floating point
	};

	/**
	 * @brief Index types for vector search.
	 */
	enum IndexType {
		HNSW ///< Hierarchical Navigable Small World graph index
	};
	// Metadata value types
	using MetadataValue = std::variant<int64_t, double, std::string, bool>;
	using Metadata = std::unordered_map<std::string, MetadataValue>;

	// Error handling
	enum class ErrorCode { 
		DimensionMismatch, NotFound, DuplicateID, StorageError 
	};

	template<typename T>
	using Result = std::expected<T, ErrorCode>;

	namespace utils {
	using json = nlohmann::json;
	}

	struct ArrowRecord {
		VectorID id;										///< Unique identifier for the record
		std::vector<float> embedding;		///< The vector embedding
	//Metadata metadata;							///< Metadata (not yet implemented)
	};
}


#endif // TYPES_H
