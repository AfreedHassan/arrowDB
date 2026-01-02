#ifndef ARROWDB_H
#define ARROWDB_H
#include "hnsw_index.h"
#include "utils/utils.h"
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

namespace arrow {
	class CollectionConfig;
	class Collection;

	namespace utils {
		/**
		 * @brief Convert CollectionConfig to JSON object.
		 * 
		 * Example output:
		 * {
		 *   "name": "my_collection",
		 *   "dimensions": 128,
		 *   "metric": "Cosine",
		 *   "dtype": "Float16",
		 *   "idxType": "HNSW"
		 * }
		 */

		json collectionConfigToJson(const CollectionConfig& config); 

		/**
		 * @brief Convert JSON object to CollectionConfig.
		 */
		CollectionConfig jsonToCollectionConfig(const json& j); 

		/**
		 * @brief Convert HNSWConfig to JSON object.
		 */
		json hnswConfigToJson(const HNSWConfig& config); 

		/**
		 * @brief Convert JSON object to HNSWConfig.
		 */
		HNSWConfig jsonToHNSWConfig(const json& j); 

		/**
		 * @brief Export CollectionConfig to JSON file (for meta.json).
		 * 
		 * Format:
		 * {
		 *   "name": "my_collection",
		 *   "dimensions": 128,
		 *   "metric": "Cosine",
		 *   "dtype": "Float16",
		 *   "idxType": "HNSW",
		 *   "hnsw": {
		 *     "maxElements": 1000000,
		 *     "M": 64,
		 *     "efConstruction": 200
		 *   }
		 * }
		 */
		void exportCollectionConfigToJson(
				const CollectionConfig& config,
				const HNSWConfig& hnswConfig,
				const std::string& filepath
				); 

		/**
		 * @brief Import CollectionConfig and HNSWConfig from JSON file.
		 * 
		 * @param filepath Path to meta.json file
		 * @return Pair of (CollectionConfig, HNSWConfig)
		 * @throws std::runtime_error if file cannot be read or parsed
		 */
		std::pair<CollectionConfig, HNSWConfig> importConfigsFromJson(const std::string& filepath);
	} // namespace arrow::utils
	
	/**
	 * @brief Configuration for a vector collection.
	 *
	 * CollectionConfig specifies the properties of a vector collection,
	 * including its name, vector dimensions, distance metric, and data type.
	 */
	class CollectionConfig {
		public:
			std::string name;      ///< Collection name
			uint32_t dimensions;     ///< Dimension of vectors in this collection
			DistanceMetric metric; ///< Distance metric to use for similarity
			DataType dtype;        ///< Data type for vector storage
			IndexType idxType;     ///< Index type for search acceleration

			/**
			 * @brief Constructs a CollectionConfig with the specified parameters.
			 * @param name_ The name of the collection.
			 * @param dimensions_ The dimension of vectors in this collection. Must be >
			 * 0.
			 * @param metric_ The distance metric to use for similarity computation.
			 * @param dtype_ The data type for storing vectors.
			 * @throws std::invalid_argument if dimensions_ <= 0 or dtype_ is unsupported.
			 */
			CollectionConfig(std::string name_, uint32_t dimensions_,
					DistanceMetric metric_, DataType dtype_)
				: name(std::move(name_)), dimensions(dimensions_), metric(metric_),
				dtype(dtype_) {

					if (dimensions <= 0) {
						throw std::invalid_argument("dimension must be > 0");
					}

					if (dtype != DataType::Float16 && dtype != DataType::Int16) {
						throw std::invalid_argument("only float16 and int16 supported");
					}
				}
	};

} // namespace arrow

// Include utils.h after CollectionConfig is defined so collection-specific utilities can use it
// ARROWDB_H is already defined, so utils.h will compile collection-specific functions
namespace arrow {

/**
 * @brief A collection of vectors with a specific configuration.
 *
 * Collection represents a named group of vectors that share the same
 * dimension, distance metric, and data type. It serves as the primary
 * interface for vector database operations.
 *
 * Default HNSW parameters are optimized for large datasets (100K+ vectors):
 * - M=64: Provides 91-92% recall@10 for 100K vectors (vs 74-78% with M=32)
 * - efConstruction=200: Balanced build time and quality
 * - Default EF search=200: Provides ~91% recall@10 for 100K vectors
 *
 * For smaller datasets (<10K), consider custom HNSWConfig with M=32 to save memory:
 * - M=32: Provides 98.4% recall@10 for 10K vectors with lower memory usage
 * - efConstruction=200: Sufficient for small datasets
 */
class Collection {
public:
  /**
   * @brief Constructs a Collection with the given configuration and default HNSW parameters.
   * 
   * Uses benchmark-optimized defaults: M=64, efConstruction=200.
   * Optimized for large datasets (100K+ vectors) with 91-92% recall@10.
   * 
   * @param config The configuration for this collection.
   */
  explicit Collection(CollectionConfig config)
      : config_(std::move(config)), 
				hnswConfig_({}),
				index_(std::make_unique<HNSWIndex>(config_.dimensions, config_.metric)) {}

  /**
   * @brief Constructs a Collection with custom HNSW configuration.
   * 
   * Use this constructor to override default HNSW parameters for:
   * - Small datasets (<10K): Use M=32 for lower memory usage (still achieves 98% recall)
   * - Memory-constrained environments: Use M=16 or M=32 (lower recall but less memory)
   * - Very large datasets (1M+): Use M=64 with efConstruction=400
   * 
   * @param config The configuration for this collection.
   * @param hnsw_config Custom HNSW index configuration.
   */
  explicit Collection(CollectionConfig config, const HNSWConfig& hnsw_config)
      : config_(std::move(config)), 
				hnswConfig_(hnsw_config),
				index_(std::make_unique<HNSWIndex>(config_.dimensions, config_.metric, hnsw_config)) {}

  /**
   * @brief Gets the name of this collection.
   * @return The collection name.
   */
  const std::string &name() const { return config_.name; }

  /**
   * @brief Gets the dimension of vectors in this collection.
   * @return The vector dimension.
   */
  uint32_t dimension() const { return config_.dimensions; }

  /**
   * @brief Gets the distance metric used by this collection.
   * @return The distance metric.
   */
  DistanceMetric metric() const { return config_.metric; }

  /**
   * @brief Gets the data type used for vector storage.
   * @return The data type.
   */
  DataType dtype() const { return config_.dtype; }

  /**
   * @brief Gets the HNSW configuration used by this collection.
   * @return The HNSW configuration.
   */
  const HNSWConfig& hnswConfig() const { return hnswConfig_; }

  /// Insert a vector into the collection.
  /// 
  /// @param id Unique identifier for the vector
  /// @param vec Vector data (must match collection dimension)
  /// 
  /// @note Insert throughput: ~3-14k vectors/second depending on dataset size.
  ///       For large batch inserts, consider future batch API.
  void insert(VectorID id, const std::vector<float> &vec) {
    index_->insert(id, vec);
  }

  /// Set metadata for a vector.
  /// 
  /// @param id Vector identifier
  /// @param metadata Metadata to associate with the vector
  void setMetadata(VectorID id, const Metadata& metadata) {
    metadata_[id] = metadata;
  }

  /// Search for k nearest neighbors.
  /// 
  /// @param query Query vector (must match collection dimension)
  /// @param k Number of results to return
  /// @param ef Search beam width (higher = better recall, slower)
  /// @return Vector of search results (id, score pairs), sorted by score descending
  /// 
  /// @note With default M=64, EF=200 provides ~91% recall@10 for 100K vectors.
  ///       For >95% recall, use EF=300-400 or higher.
  /// @note Search latency: ~8-9ms for 100K vectors with EF=200, M=64.
  std::vector<SearchResult> search(const std::vector<float> &query, uint32_t k,
                                    uint32_t ef = 200) const {  // Optimized for 100K+ vectors
    return index_->search(query, k, ef);
  }

  /// Save the entire collection to disk.
  /// 
  /// Persists the collection to a directory containing:
  /// - meta.json: Collection config and HNSW config
  /// - index.bin: HNSW index structure (includes vector data)
  /// - metadata.json: Vector metadata
  /// 
  /// @param directoryPath Directory path where the collection will be saved.
  ///                       The directory will be created if it doesn't exist.
  /// @throws std::runtime_error if directory creation or file writing fails
	void save(const std::string& directoryPath) const {
		namespace fs = std::filesystem;
		
		// Create directory if it doesn't exist
		fs::create_directories(directoryPath);
		
		// Save collection config and HNSW config to meta.json
		std::string metaPath = (fs::path(directoryPath) / "meta.json").string();
		utils::exportCollectionConfigToJson(config_, hnswConfig_, metaPath);
		
		// Save HNSW index to index.bin
		std::string indexPath = (fs::path(directoryPath) / "index.bin").string();
		index_->saveIndex(indexPath);
		
		// Save metadata to metadata.json (only if not empty)
		if (!metadata_.empty()) {
			std::string metadataPath = (fs::path(directoryPath) / "metadata.json").string();
			utils::exportMetadataToJson(metadata_, metadataPath);
		}
	}

  /// Load a collection from disk.
  /// 
  /// Loads a collection from a directory containing:
  /// - meta.json: Collection config and HNSW config
  /// - index.bin: HNSW index structure (includes vector data)
  /// - metadata.json: Vector metadata (optional)
  /// 
  /// @param directoryPath Directory path where the collection is stored.
  /// @return A new Collection instance loaded from disk.
  /// @throws std::runtime_error if directory doesn't exist, files are missing, or loading fails
	static Collection load(const std::string& directoryPath) {
		namespace fs = std::filesystem;
		
		// Check directory exists
		if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
			throw std::runtime_error("Collection directory does not exist: " + directoryPath);
		}
		
		// Load collection config and HNSW config from meta.json
		std::string metaPath = (fs::path(directoryPath) / "meta.json").string();
		if (!fs::exists(metaPath)) {
			throw std::runtime_error("meta.json not found in collection directory: " + directoryPath);
		}
		auto [collectionCfg, hnswCfg] = utils::importConfigsFromJson(metaPath);
		
		// Create collection with loaded config
		Collection collection(collectionCfg, hnswCfg);
		
		// Load HNSW index from index.bin
		std::string indexPath = (fs::path(directoryPath) / "index.bin").string();
		if (!fs::exists(indexPath)) {
			throw std::runtime_error("index.bin not found in collection directory: " + directoryPath);
		}
		collection.index_->loadIndex(indexPath);
		
		// Load metadata from metadata.json (optional)
		std::string metadataPath = (fs::path(directoryPath) / "metadata.json").string();
		if (fs::exists(metadataPath)) {
			collection.metadata_ = utils::importMetadataFromJson(metadataPath);
		}
		
		return collection;
	}

  /// Get the number of vectors in the collection.
  uint32_t size() const { return index_->size(); }

	void printCollectionInfo() const {
		// use format instead
    std::cout << "Collection '" << this->name() << "' created:\n";
    std::cout << "  Dimension: " << this->dimension() << "\n";
    std::cout << "  Metric: Cosine\n";
    std::cout << "  Initial size: " << this->size() << "\n\n";
	}

	bool exportMetadataToJson(const std::string& filepath) const {
			try {
				utils::exportMetadataToJson(metadata_, filepath);
				return true;
			} catch (const std::exception& e) {
				std::cerr << "Error exporting metadata: " << e.what() << "\n";
				return false;
			}
	}

private:
  const CollectionConfig config_;
  HNSWConfig hnswConfig_;
  std::unique_ptr<HNSWIndex> index_;
	std::unordered_map<VectorID, Metadata> metadata_;

  // ----- Internal state (added later) -----
  // VectorStore vector_store_;
  // HNSWIndex hnsw_;
  // WAL wal_;
}; // class Collection

/**
 * @brief Legacy vector database class.
 * @deprecated This class appears to be legacy code and may be removed.
 */
class arrowDB {
public:
  std::vector<int> store; ///< Internal storage

  /**
   * @brief Constructs an arrowDB with initial capacity.
   * @param n Initial capacity for the store.
   */
  arrowDB(int n) { store.reserve(n); };
  arrowDB(const arrowDB &) = default;
  arrowDB(arrowDB &&) = default;
  arrowDB &operator=(const arrowDB &) = default;
  arrowDB &operator=(arrowDB &&) = default;

  /**
   * @brief Inserts a value into the store.
   * @param n The value to insert.
   */
  void insert(int n) { this->store.push_back(n);   }
};

} // namespace arrow
	//
namespace arrow::utils {
inline json collectionConfigToJson(const CollectionConfig& config) {
    json j = json::object();
    j["name"] = config.name;
    j["dimensions"] = config.dimensions;
    j["metric"] = distanceMetricToJson(config.metric);
    j["dtype"] = dataTypeToJson(config.dtype);
    j["idxType"] = "HNSW";  // Currently only HNSW is supported
    return j;
}

inline CollectionConfig jsonToCollectionConfig(const json& j) {
    return CollectionConfig(
        j["name"].get<std::string>(),
        j["dimensions"].get<uint32_t>(),
        jsonToDistanceMetric(j["metric"]),
        jsonToDataType(j["dtype"])
    );
}

inline json hnswConfigToJson(const HNSWConfig& config) {
    json j = json::object();
    j["maxElements"] = config.maxElements;
    j["M"] = config.M;
    j["efConstruction"] = config.efConstruction;
    return j;
}

inline HNSWConfig jsonToHNSWConfig(const json& j) {
    HNSWConfig config;
    if (j.contains("maxElements")) {
        config.maxElements = j["maxElements"].get<uint32_t>();
    }
    if (j.contains("M")) {
        config.M = j["M"].get<uint32_t>();
    }
    if (j.contains("efConstruction")) {
        config.efConstruction = j["efConstruction"].get<uint32_t>();
    }
    return config;
}

inline void exportCollectionConfigToJson(
    const CollectionConfig& config,
    const HNSWConfig& hnswConfig,
    const std::string& filepath
) {
    json j = collectionConfigToJson(config);
    j["hnsw"] = hnswConfigToJson(hnswConfig);
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    file << j.dump(2);  // Pretty print with 2-space indent
    file.close();
}

inline std::pair<CollectionConfig, HNSWConfig> importConfigsFromJson(
    const std::string& filepath
) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    json j;
    file >> j;
    file.close();
    
    CollectionConfig config = jsonToCollectionConfig(j);
    HNSWConfig hnswConfig = j.contains("hnsw") 
        ? jsonToHNSWConfig(j["hnsw"])
        : HNSWConfig{};  // Use defaults if not present
    
    return {config, hnswConfig};
}
} // namespace arrow::utils

#endif // ARROWDB_H
