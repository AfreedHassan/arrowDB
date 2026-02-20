#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
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
	class MetadataValue {
	public:
		using Variant = std::variant<int64_t, double, std::string, bool>;

		// Implicit constructors (preserves all existing construction patterns)
		MetadataValue() : data_(int64_t{0}) {}
		MetadataValue(int64_t v) : data_(v) {}
		MetadataValue(int v) : data_(static_cast<int64_t>(v)) {}
		MetadataValue(double v) : data_(v) {}
		MetadataValue(std::string v) : data_(std::move(v)) {}
		MetadataValue(const char* v) : data_(std::string(v)) {}
		MetadataValue(bool v) : data_(v) {}

		// Typed accessors
		int64_t asInt64() const { return std::get<int64_t>(data_); }
		double asDouble() const { return std::get<double>(data_); }
		const std::string& asString() const { return std::get<std::string>(data_); }
		bool asBool() const { return std::get<bool>(data_); }

		// Implicit conversions (enables: std::string s = meta["key"];)
		operator int64_t() const { return asInt64(); }
		operator double() const { return asDouble(); }
		operator const std::string&() const { return asString(); }
		// No operator bool() — use .asBool() to avoid accidental if(value) conversions

		// Variant access for internal code (std::visit, std::get_if, etc.)
		const Variant& variant() const { return data_; }
		Variant& variant() { return data_; }

		// Equality (delegates to variant ==)
		bool operator==(const MetadataValue& o) const { return data_ == o.data_; }
		bool operator!=(const MetadataValue& o) const { return data_ != o.data_; }

	private:
		Variant data_;
	};

	using Metadata = std::unordered_map<std::string, MetadataValue>;

	// Forward declaration — full definition in arrow/filter.h.
	class MetadataFilter;

	// ─── Metadata Schema ─────────────────────────────────────

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

  constexpr bool kRequiredField = true;

	/// Schema for validating metadata on insert/update.
	/// An empty schema (no fields) disables validation (backward compat).
	struct MetadataSchema {
		std::vector<FieldDef> fields;

		/// Builder: add a field definition and return *this for chaining.
		MetadataSchema& field(std::string name, FieldType type, bool required = false) {
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
	// ── Backward-compatible free functions for MetadataValue ──
	// Enables: get<T>(mv), get_if<T>(&mv), holds_alternative<T>(mv), visit(fn, mv)
	// Found via ADL since MetadataValue lives in arrow::.

	template <typename T>
	decltype(auto) get(MetadataValue& v) { return std::get<T>(v.variant()); }
	template <typename T>
	decltype(auto) get(const MetadataValue& v) { return std::get<T>(v.variant()); }
	template <typename T>
	decltype(auto) get(MetadataValue&& v) { return std::get<T>(std::move(v.variant())); }

	template <typename T>
	auto get_if(MetadataValue* v) { return v ? std::get_if<T>(&v->variant()) : nullptr; }
	template <typename T>
	auto get_if(const MetadataValue* v) { return v ? std::get_if<T>(&v->variant()) : nullptr; }

	template <typename T>
	bool holds_alternative(const MetadataValue& v) { return std::holds_alternative<T>(v.variant()); }

	template <typename Fn>
	decltype(auto) visit(Fn&& fn, const MetadataValue& v) { return std::visit(std::forward<Fn>(fn), v.variant()); }
	template <typename Fn>
	decltype(auto) visit(Fn&& fn, MetadataValue& v) { return std::visit(std::forward<Fn>(fn), v.variant()); }

} // namespace arrow

// ── std:: overloads for full backward compatibility ──────────
// Allows existing std::get<T>(metadataValue) code to compile unchanged.

namespace std {
	template <typename T>
	decltype(auto) get(arrow::MetadataValue& v) { return std::get<T>(v.variant()); }
	template <typename T>
	decltype(auto) get(const arrow::MetadataValue& v) { return std::get<T>(v.variant()); }
	template <typename T>
	decltype(auto) get(arrow::MetadataValue&& v) { return std::get<T>(std::move(v.variant())); }

	template <typename T>
	auto get_if(arrow::MetadataValue* v) { return v ? std::get_if<T>(&v->variant()) : nullptr; }
	template <typename T>
	auto get_if(const arrow::MetadataValue* v) { return v ? std::get_if<T>(&v->variant()) : nullptr; }

	template <typename T>
	bool holds_alternative(const arrow::MetadataValue& v) { return std::holds_alternative<T>(v.variant()); }

	template <typename Fn>
	decltype(auto) visit(Fn&& fn, const arrow::MetadataValue& v) { return std::visit(std::forward<Fn>(fn), v.variant()); }
	template <typename Fn>
	decltype(auto) visit(Fn&& fn, arrow::MetadataValue& v) { return std::visit(std::forward<Fn>(fn), v.variant()); }
}

#endif // TYPES_H
