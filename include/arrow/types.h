#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <nlohmann/json.hpp>
#include <variant>
#include <unordered_map>
#include <expected>
#include <vector>
#include "utils/status.h"

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
		Int32,  ///< 32-bit signed integer
		Float32 ///< 32-bit floating point
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

	namespace utils {
	using json = nlohmann::json;
	}

	struct ArrowRecord {
		VectorID id;										///< Unique identifier for the record
		std::vector<float> embedding;		///< The vector embedding
	//Metadata metadata;							///< Metadata (not yet implemented)
	};

	/// Result of a single insert operation in a batch operation
	struct InsertResult {
		VectorID id;           ///< Vector ID that was attempted
		utils::Status status;  ///< Success or error status
	};

	/// Aggregate result of batch insert operation
	struct BatchInsertResult {
		std::vector<InsertResult> results;  ///< Per-vector results
		size_t successCount;                ///< Number of successful inserts
		size_t failureCount;                ///< Number of failed inserts
	};
}


#endif // TYPES_H
