#ifndef  ARROWDB_H
#define  ARROWDB_H
#include <variant>
#include <vector>
#include <string>

namespace arrow {
	
	/**
	 * @brief Distance metrics for vector similarity computation.
	 */
	enum class DistanceMetric { 
		Cosine,        ///< Cosine similarity (dot product of normalized vectors)
		L2,            ///< L2 (Euclidean) distance
		InnerProduct   ///< Inner product (dot product)
	};

	/**
	 * @brief Data types for vector storage.
	 */
	enum class DataType { 
		Int16,    ///< 16-bit signed integer
		Float16   ///< 16-bit floating point
	};

	/**
	 * @brief Index types for vector search.
	 */
	enum IndexType { 
		HNSW   ///< Hierarchical Navigable Small World graph index
	};

	/**
	 * @brief Configuration for a vector collection.
	 * 
	 * CollectionConfig specifies the properties of a vector collection,
	 * including its name, vector dimensions, distance metric, and data type.
	 */
	class CollectionConfig {
	public:
		std::string name;        ///< Collection name
		size_t dimensions;       ///< Dimension of vectors in this collection
		DistanceMetric metric;   ///< Distance metric to use for similarity
		DataType dtype;         ///< Data type for vector storage
		IndexType idxType;      ///< Index type for search acceleration
	
		/**
		 * @brief Constructs a CollectionConfig with the specified parameters.
		 * @param name_ The name of the collection.
		 * @param dimensions_ The dimension of vectors in this collection. Must be > 0.
		 * @param metric_ The distance metric to use for similarity computation.
		 * @param dtype_ The data type for storing vectors.
		 * @throws std::invalid_argument if dimensions_ <= 0 or dtype_ is unsupported.
		 */
		CollectionConfig(
			std::string name_,
			size_t dimensions_,
			DistanceMetric metric_,
			DataType dtype_
		)
			: name(std::move(name_)),
			  dimensions(dimensions_),
			  metric(metric_),
			  dtype(dtype_) {

			if (dimensions <= 0) {
				throw std::invalid_argument("dimension must be > 0");
			}

			if (dtype != DataType::Float16 && dtype != DataType::Int16) {
				throw std::invalid_argument("only float16 and int16 supported");
			}
		}
	};

	/**
	 * @brief A collection of vectors with a specific configuration.
	 * 
	 * Collection represents a named group of vectors that share the same
	 * dimension, distance metric, and data type. It serves as the primary
	 * interface for vector database operations.
	 */
	class Collection {
		public:
			/**
			 * @brief Constructs a Collection with the given configuration.
			 * @param config The configuration for this collection.
			 */
			explicit Collection(CollectionConfig config) : config_(std::move(config)) {}

			/**
			 * @brief Gets the name of this collection.
			 * @return The collection name.
			 */
			const std::string& name() const {
				return config_.name;
			}

			/**
			 * @brief Gets the dimension of vectors in this collection.
			 * @return The vector dimension.
			 */
			size_t dimension() const {
				return config_.dimensions;
			}

			/**
			 * @brief Gets the distance metric used by this collection.
			 * @return The distance metric.
			 */
			DistanceMetric metric() const {
				return config_.metric;
			}

			/**
			 * @brief Gets the data type used for vector storage.
			 * @return The data type.
			 */
			DataType dtype() const {
				return config_.dtype;
			}
		private:
			const CollectionConfig config_;

			// ----- Internal state (added later) -----
			// VectorStore vector_store_;
			// HNSWIndex hnsw_;
			// WAL wal_;
	};
} // namespace arrow

/**
 * @brief A record containing a vector embedding and its ID.
 * @deprecated This struct appears to be legacy code and may be removed.
 */
struct arrowRecord {
	uint64_t id;                    ///< Unique identifier for the record
	std::vector<float> embedding;   ///< The vector embedding
	//Metadata metadata;             ///< Metadata (not yet implemented)
};

/**
 * @brief Legacy vector database class.
 * @deprecated This class appears to be legacy code and may be removed.
 */
class arrowDB {
	public : 
	std::vector<int> store;  ///< Internal storage
	
	/**
	 * @brief Constructs an arrowDB with initial capacity.
	 * @param n Initial capacity for the store.
	 */
	arrowDB(int n) { 
		store.reserve(n);
	};
	arrowDB(const arrowDB &) = default;
	arrowDB(arrowDB &&) = default;
	arrowDB &operator=(const arrowDB &) = default;
	arrowDB &operator=(arrowDB &&) = default;

	/**
	 * @brief Inserts a value into the store.
	 * @param n The value to insert.
	 */
	void insert(int n) {
		this->store.push_back(n);
	}
};

#endif
