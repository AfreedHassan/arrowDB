#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <unordered_map>
#include <expected>
#include <vector>
#include "utils/status.h"

namespace arrow {
  using VectorID = std::string;

	/**
	 * @brief Index spaces for vector similarity computation.
	 */
	enum class Space {
		Cosine,      ///< Cosine similarity (dot product of normalized vectors)
		L2,          ///< L2 (Euclidean) distance
		InnerProduct ///< Inner product (dot product)
	};

	/**
	 * @brief Index types for vector search.
	 */
	enum class IndexType : uint8_t {
		HNSW = 0 ///< Hierarchical Navigable Small World graph index
		// Future: IVF = 1, Flat = 2
	};

	/**
	 * @brief Vector quantization strategy.
	 */
	enum class Quantization : uint8_t {
		None = 0,    ///< Full float32 precision
		INT8 = 1     ///< 8-bit scalar quantization (~4x less memory in search)
		// Future: INT4 = 2, FP16 = 3
	};

	// Metadata value types
	using MetadataValue = std::variant<int64_t, double, std::string, bool>;
	using Metadata = std::unordered_map<std::string, MetadataValue>;

	/// Predicate for filtering search results by metadata.
	using MetadataFilter = std::function<bool(const Metadata&)>;

	// ─── Schema ───────────────────────────────────────────────

	/// Supported metadata field types for schema validation.
	enum class FieldType : uint8_t {
		Int64,
		Double,
		String,
		Bool
	};

	/// Definition of a single schema field.
	struct FieldDef {
		std::string name;
		FieldType type;
		bool required = false;
	};

	/// Schema for validating metadata on insert/update.
	/// An empty schema (no fields) disables validation (backward compat).
	struct Schema {
		std::vector<FieldDef> fields;

		/// Builder: add a field definition and return *this for chaining.
		Schema& field(std::string name, FieldType type, bool required = false) {
			fields.push_back({std::move(name), type, required});
			return *this;
		}

		bool empty() const noexcept { return fields.empty(); }
	};

	// ─── Document ─────────────────────────────────────────────

	/// A document combining vector ID, embedding, and metadata.
	/// If `id` is empty, a UUID will be auto-generated on insert.
	struct Document {
		VectorID id;
		std::vector<float> embedding;
		Metadata metadata;
	};

	/// Result of a single insert operation in a batch operation
	struct InsertResult {
		VectorID id;             ///< Vector ID that was attempted
		utils::Status status;    ///< Success or error status
	};

	/// Aggregate result of batch insert operation
	struct BatchInsertResult {
		std::vector<InsertResult> results;  ///< Per-vector results
		size_t successCount;                ///< Number of successful inserts
		size_t failureCount;                ///< Number of failed inserts
	};

	/// A document with its similarity score and metadata from a search result
	struct ScoredDocument {
		VectorID id;                        ///< Document/vector identifier
		float score;                        ///< Similarity score (higher = more similar)
		Metadata metadata;                  ///< Associated metadata
	};

	/// Result of a search operation
	struct SearchResult {
		std::vector<ScoredDocument> hits;   ///< Matching documents sorted by score
		// Future: uint64_t elapsed_ms;     ///< Query execution time
		// Future: size_t total_count;      ///< Total matches (for pagination)
	};

	/// Result from index search (id + score only, no metadata)
	struct IndexSearchResult {
		VectorID id;    ///< Vector identifier
		float score;    ///< Similarity score
	};
}


#endif // TYPES_H
