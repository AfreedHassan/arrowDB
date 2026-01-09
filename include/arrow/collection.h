#ifndef ARROWDB_H
#define ARROWDB_H

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "hnsw_index.h"
#include "utils/utils.h"
#include "wal.h"

namespace arrow {
class CollectionConfig;
class Collection;

/// Recovery metadata persisted in meta.json for crash recovery.
struct RecoveryMetadata {
  uint64_t lastPersistedLsn = 0;   ///< LSN of last persisted state
  uint64_t lastPersistedTxid = 0;  ///< TXID of last persisted state
  bool cleanShutdown = true;       ///< Whether last shutdown was clean
};

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

json collectionConfigToJson(const CollectionConfig &config);

/**
 * @brief Convert JSON object to CollectionConfig.
 */
CollectionConfig jsonToCollectionConfig(const json &j);

/**
 * @brief Convert HNSWConfig to JSON object.
 */
json hnswConfigToJson(const HNSWConfig &config);

/**
 * @brief Convert JSON object to HNSWConfig.
 */
HNSWConfig jsonToHNSWConfig(const json &j);

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
 *   },
 *   "recovery": {
 *     "lastPersistedLsn": 42,
 *     "lastPersistedTxid": 42,
 *     "cleanShutdown": true
 *   }
 * }
 */
void exportCollectionConfigToJson(const CollectionConfig &config,
                                  const HNSWConfig &hnswConfig,
                                  const std::string &filepath,
                                  const RecoveryMetadata &recovery = {});

/**
 * @brief Import CollectionConfig, HNSWConfig, and RecoveryMetadata from JSON.
 *
 * @param filepath Path to meta.json file
 * @return Tuple of (CollectionConfig, HNSWConfig, RecoveryMetadata)
 * @throws std::runtime_error if file cannot be read or parsed
 */
std::tuple<CollectionConfig, HNSWConfig, RecoveryMetadata>
importConfigsFromJson(const std::string &filepath);
} // namespace utils

/**
 * @brief Configuration for a vector collection.
 *
 * CollectionConfig specifies the properties of a vector collection,
 * including its name, vector dimensions, distance metric, and data type.
 */
class CollectionConfig {
public:
  std::string name;      ///< Collection name
  uint32_t dimensions;   ///< Dimension of vectors in this collection
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

    if (dtype != DataType::Float32 && dtype != DataType::Int32) {
      throw std::invalid_argument("only float16 and int16 supported");
    }
  }
};

} // namespace arrow

// Include utils.h after CollectionConfig is defined so collection-specific
// utilities can use it ARROWDB_H is already defined, so utils.h will compile
// collection-specific functions
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
 * For smaller datasets (<10K), consider custom HNSWConfig with M=32 to save
 * memory:
 * - M=32: Provides 98.4% recall@10 for 10K vectors with lower memory usage
 * - efConstruction=200: Sufficient for small datasets
 */
class Collection {
public:
  /**
   * @brief Constructs a Collection with the given configuration and default
   * HNSW parameters.
   *
   * Uses benchmark-optimized defaults: M=64, efConstruction=200.
   * Optimized for large datasets (100K+ vectors) with 91-92% recall@10.
   *
   * @param config The configuration for this collection.
   */
  explicit Collection(CollectionConfig config)
      : config_(std::move(config)),
        hnswConfig_({}),
        pIndex_(std::make_unique<HNSWIndex>(config_.dimensions, config_.metric)) {}

  /**
   * @brief Constructs a Collection with custom HNSW configuration.
   *
   * Use this constructor to override default HNSW parameters for:
   * - Small datasets (<10K): Use M=32 for lower memory usage (still achieves
   * 98% recall)
   * - Memory-constrained environments: Use M=16 or M=32 (lower recall but less
   * memory)
   * - Very large datasets (1M+): Use M=64 with efConstruction=400
   *
   * @param config The configuration for this collection.
   * @param hnsw_config Custom HNSW index configuration.
   */
  explicit Collection(CollectionConfig config, const HNSWConfig &hnsw_config)
      : config_(std::move(config)), hnswConfig_(hnsw_config),
        pIndex_(std::make_unique<HNSWIndex>(config_.dimensions, config_.metric,
                                           hnsw_config)) {}

  /// Constructs a Collection with persistence path for WAL-backed durability.
  ///
  /// @param config The configuration for this collection.
  /// @param persistencePath Directory path for WAL storage.
  explicit Collection(CollectionConfig config,
                     const std::filesystem::path& persistencePath)
      : config_(std::move(config)),
        hnswConfig_({}),
        pIndex_(std::make_unique<HNSWIndex>(config_.dimensions, config_.metric)),
        persistencePath_(persistencePath) {
    initializeWal();
  }

  /// Constructs a Collection with custom HNSW config and persistence path.
  ///
  /// @param config The configuration for this collection.
  /// @param hnsw_config Custom HNSW index configuration.
  /// @param persistencePath Directory path for WAL storage.
  explicit Collection(CollectionConfig config, const HNSWConfig &hnsw_config,
                     const std::filesystem::path& persistencePath)
      : config_(std::move(config)), hnswConfig_(hnsw_config),
        pIndex_(std::make_unique<HNSWIndex>(config_.dimensions, config_.metric,
                                           hnsw_config)),
        persistencePath_(persistencePath) {
    initializeWal();
  }

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
  const HNSWConfig &hnswConfig() const { return hnswConfig_; }

  /// Insert a vector into the collection.
  ///
  /// @param id Unique identifier for the vector
  /// @param vec Vector data (must match collection dimension)
  ///
  /// @note Insert throughput: ~3-14k vectors/second depending on dataset size.
  utils::Status insert(VectorID id, const std::vector<float> &vec) {
    using namespace wal;
    if (vec.size() != config_.dimensions) {
      return utils::Status(
          StatusCode::kDimensionMismatch, 
          "Vector dimension mismatch: expected " + std::to_string(config_.dimensions) + ", got " + std::to_string(vec.size())
        );
    }

    Entry entry{
      .type = OperationType::INSERT,
      .version = 1,
      .lsn = lsnCounter++,
      .txid = txidCounter++,
      .headerCRC = 0,
      .payloadLength = 0,
      .vectorID = id,
      .dimension = config_.dimensions,
      .padding = 0,
      .embedding = vec,
      .payloadCRC = 0
    };
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    entry.payloadLength = entry.computePayloadLength();

    if (pWal_) {
      Status status = pWal_->log(entry);
      if (!status.ok()) {
        return status;
      }
    }
    if (!pIndex_->insert(id, vec)) {
      return utils::Status(StatusCode::kInternal, "Insert failed");
    }
    return utils::OkStatus();
  }

  /// Insert a batch of vectors with partial success semantics.
  ///
  /// Validates all dimensions upfront, writes all valid vectors to WAL in a
  /// single batch (single fsync), then inserts to HNSW. Returns per-vector
  /// results allowing caller to see which inserts succeeded and which failed.
  ///
  /// @param batch Vector of (id, vector) pairs to insert
  /// @return Result containing BatchInsertResult with per-vector status
  ///
  /// @note Performance: 10-100x faster than N individual inserts due to
  ///       single fsync for entire batch vs N fsyncs.
  utils::Result<BatchInsertResult> insertBatch(
      const std::vector<std::pair<VectorID, std::vector<float>>>& batch) {

    BatchInsertResult result;
    result.results.resize(batch.size());  // Pre-allocate indexed results
    result.successCount = 0;
    result.failureCount = 0;

    // Phase 1: Validate all dimensions upfront
    std::vector<bool> validDimensions(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      validDimensions[i] = (batch[i].second.size() == config_.dimensions);
    }

    // Phase 2: Create WAL entries for all valid vectors and mark invalid ones
    std::vector<wal::Entry> walEntries;
    walEntries.reserve(batch.size());

    for (size_t i = 0; i < batch.size(); ++i) {
      const auto& [id, vec] = batch[i];

      if (!validDimensions[i]) {
        result.results[i] = {
          id,
          utils::Status(utils::StatusCode::kDimensionMismatch,
                       "Vector dimension mismatch")
        };
        result.failureCount++;
        continue;
      }

      // Create WAL entry
      wal::Entry entry{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = lsnCounter++,
        .txid = txidCounter++,
        .headerCRC = 0,
        .payloadLength = 0,
        .vectorID = id,
        .dimension = config_.dimensions,
        .padding = 0,
        .embedding = vec,
        .payloadCRC = 0
      };
      entry.headerCRC = entry.computeHeaderCrc();
      entry.payloadCRC = entry.computePayloadCrc();
      entry.payloadLength = entry.computePayloadLength();

      walEntries.push_back(std::move(entry));
    }

    // Phase 3: Batch write to WAL (single fsync)
    if (pWal_ && !walEntries.empty()) {
      utils::Status walStatus = pWal_->logBatch(walEntries);
      if (!walStatus.ok()) {
        // WAL failure - rollback counters and return error
        lsnCounter -= walEntries.size();
        txidCounter -= walEntries.size();
        return walStatus;
      }
    }

    // Phase 4: Insert into HNSW index (partial success)
    for (size_t i = 0; i < batch.size(); ++i) {
      const auto& [id, vec] = batch[i];

      if (!validDimensions[i]) {
        continue;  // Already marked as failed
      }

      // Attempt HNSW insert
      bool insertSuccess = pIndex_->insert(id, vec);

      if (insertSuccess) {
        result.results[i] = {id, utils::OkStatus()};
        result.successCount++;
      } else {
        result.results[i] = {
          id,
          utils::Status(utils::StatusCode::kInternal, "HNSW insert failed")
        };
        result.failureCount++;
      }
    }

    return result;
  }

  /// Set metadata for a vector.
  ///
  /// @param id Vector identifier
  /// @param metadata Metadata to associate with the vector
  void setMetadata(VectorID id, const Metadata &metadata) {
    metadata_[id] = metadata;
  }

  /// Search for k nearest neighbors.
  ///
  /// @param query Query vector (must match collection dimension)
  /// @param k Number of results to return
  /// @param ef Search beam width (higher = better recall, slower)
  /// @return Vector of search results (id, score pairs), sorted by score
  /// descending
  ///
  /// @note With default M=64, EF=200 provides ~91% recall@10 for 100K vectors.
  ///       For >95% recall, use EF=300-400 or higher.
  /// @note Search latency: ~8-9ms for 100K vectors with EF=200, M=64.
  std::vector<SearchResult>
  search(const std::vector<float> &query, uint32_t k,
         uint32_t ef = 200) const { // Optimized for 100K+ vectors
    return pIndex_->search(query, k, ef);
  }

  /// Search for k nearest neighbors for multiple queries in parallel.
  ///
  /// Parallelizes searches across multiple queries using a simple thread pool.
  /// Thread-safe because hnswlib guarantees thread-safety for concurrent searches.
  ///
  /// @param queries Vector of query vectors (must match collection dimension)
  /// @param k Number of results per query
  /// @param ef Search beam width (higher = better recall, slower)
  /// @return Result containing vector of result vectors (one per query), or error
  ///
  /// @note Performance: ~6-8x speedup on 8-core machine for 8+ queries
  utils::Result<std::vector<std::vector<SearchResult>>> searchBatch(
      const std::vector<std::vector<float>>& queries,
      uint32_t k,
      uint32_t ef = 200) const {

    // Validate all query dimensions upfront
    for (size_t i = 0; i < queries.size(); ++i) {
      if (queries[i].size() != config_.dimensions) {
        return utils::Result<std::vector<std::vector<SearchResult>>>(
            utils::Status(utils::StatusCode::kDimensionMismatch,
                         "Query " + std::to_string(i) + " dimension mismatch: expected " +
                         std::to_string(config_.dimensions) + ", got " +
                         std::to_string(queries[i].size()))
        );
      }
    }

    // Delegate to parallel search implementation
    return parallelSearch(pIndex_.get(), queries, k, ef);
  }

  /// Save the entire collection to disk with WAL checkpoint.
  ///
  /// Persists the collection to a directory containing:
  /// - meta.json: Collection config, HNSW config, and recovery metadata
  /// - index.bin: HNSW index structure (includes vector data)
  /// - metadata.json: Vector metadata
  ///
  /// After saving, the WAL is truncated (checkpointed) since all state is now
  /// persisted.
  ///
  /// @param directoryPath Directory path where the collection will be saved.
  ///                       The directory will be created if it doesn't exist.
  /// @return Status indicating success or failure
  utils::Status save(const std::string &directoryPath) {
    namespace fs = std::filesystem;

    // Create directory if it doesn't exist
    fs::create_directories(directoryPath);

    // Prepare recovery metadata
    // lsnCounter is the NEXT LSN to use, so lastPersistedLsn = lsnCounter - 1
    RecoveryMetadata recovery{
        .lastPersistedLsn = (lsnCounter > 0) ? lsnCounter - 1 : 0,
        .lastPersistedTxid = (txidCounter > 0) ? txidCounter - 1 : 0,
        .cleanShutdown = true};

    // Save collection config, HNSW config, and recovery metadata to meta.json
    std::string metaPath = (fs::path(directoryPath) / "meta.json").string();
    utils::exportCollectionConfigToJson(config_, hnswConfig_, metaPath,
                                        recovery);

    // Save HNSW index to index.bin
    std::string indexPath = (fs::path(directoryPath) / "index.bin").string();
    pIndex_->saveIndex(indexPath);

    // Save metadata to metadata.json (only if not empty)
    if (!metadata_.empty()) {
      std::string metadataPath =
          (fs::path(directoryPath) / "metadata.json").string();
      utils::exportMetadataToJson(metadata_, metadataPath);
    }

    // Checkpoint WAL - truncate after successful save
    if (pWal_) {
      wal::Status status = pWal_->truncate();
      if (!status.ok()) {
        return status;
      }
    }

    // Update lastPersistedLsn_
    lastPersistedLsn_ = recovery.lastPersistedLsn;

    return utils::OkStatus();
  }

  /// Load a collection from disk with WAL recovery.
  ///
  /// Loads a collection from a directory containing:
  /// - meta.json: Collection config, HNSW config, and recovery metadata
  /// - index.bin: HNSW index structure (includes vector data)
  /// - metadata.json: Vector metadata (optional)
  /// - wal/db.wal: Write-ahead log (optional)
  ///
  /// If a WAL exists, entries with LSN > lastPersistedLsn are replayed to
  /// recover any uncommitted changes from before a crash.
  ///
  /// @param directoryPath Directory path where the collection is stored.
  /// @return Result containing the loaded Collection or an error status
  static utils::Result<Collection> load(const std::string &directoryPath) {
    namespace fs = std::filesystem;

    // Check directory exists
    if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
      return utils::Status(utils::StatusCode::kNotFound,
                           "Collection directory does not exist: " +
                               directoryPath);
    }

    // Load collection config and HNSW config from meta.json
    std::string metaPath = (fs::path(directoryPath) / "meta.json").string();
    if (!fs::exists(metaPath)) {
      return utils::Status(utils::StatusCode::kNotFound,
                           "meta.json not found in collection directory: " +
                               directoryPath);
    }
    auto [collectionCfg, hnswCfg, recoveryMeta] =
        utils::importConfigsFromJson(metaPath);

    // Create collection with persistence path for WAL support
    Collection collection(collectionCfg, hnswCfg, fs::path(directoryPath));

    // Load HNSW index from index.bin
    std::string indexPath = (fs::path(directoryPath) / "index.bin").string();
    if (!fs::exists(indexPath)) {
      return utils::Status(utils::StatusCode::kNotFound,
                           "index.bin not found in collection directory: " +
                               directoryPath);
    }
    collection.pIndex_->loadIndex(indexPath);

    // Load metadata from metadata.json (optional)
    std::string metadataPath =
        (fs::path(directoryPath) / "metadata.json").string();
    if (fs::exists(metadataPath)) {
      collection.metadata_ = utils::importMetadataFromJson(metadataPath);
    }

    // Initialize counters from recovery metadata
    collection.lastPersistedLsn_ = recoveryMeta.lastPersistedLsn;
    collection.lsnCounter = recoveryMeta.lastPersistedLsn + 1;
    collection.txidCounter = recoveryMeta.lastPersistedTxid + 1;

    // Replay WAL if it exists
    fs::path walPath = fs::path(directoryPath) / "wal" / "db.wal";
    if (fs::exists(walPath)) {
      utils::Status replayStatus =
          collection.replayWal(recoveryMeta.lastPersistedLsn);
      if (!replayStatus.ok()) {
        return replayStatus;
      }
    }

    return collection;
  }

  /// Get the number of vectors in the collection.
  uint32_t size() const { return pIndex_->size(); }

  void printCollectionInfo() const {
    // use format instead
    std::cout << "Collection '" << this->name() << "' created:\n";
    std::cout << "  Dimension: " << this->dimension() << "\n";
    std::cout << "  Metric: Cosine\n";
    std::cout << "  Initial size: " << this->size() << "\n\n";
  }

  bool exportMetadataToJson(const std::string &filepath) const {
    try {
      utils::exportMetadataToJson(metadata_, filepath);
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Error exporting metadata: " << e.what() << "\n";
      return false;
    }
  }

  /// Remove a vector from the collection.
  ///
  /// Logs DELETE to WAL, marks vector as deleted in HNSW index (lazy deletion),
  /// and removes associated metadata.
  ///
  /// @param id Vector identifier to remove
  /// @return Status indicating success or failure
  utils::Status remove(VectorID id) {
    using namespace wal;

    Entry entry{.type = OperationType::DELETE,
                .version = 1,
                .lsn = lsnCounter++,
                .txid = txidCounter++,
                .headerCRC = 0,
                .payloadLength = 0,
                .vectorID = id,
                .dimension = 0,
                .padding = 0,
                .embedding = {},
                .payloadCRC = 0};
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    entry.payloadLength = entry.computePayloadLength();

    if (pWal_) {
      Status status = pWal_->log(entry);
      if (!status.ok()) {
        return status;
      }
    }

    // Use HNSW lazy deletion
    Status delStatus = pIndex_->markDelete(id);
    if (!delStatus.ok()) {
      return delStatus;
    }
    // Remove metadata
    metadata_.erase(id);

    return utils::OkStatus();
  }

  /// Perform clean shutdown - saves state and marks clean shutdown in metadata.
  ///
  /// Call this before destroying the collection to ensure all data is persisted.
  /// If no persistence path was configured, this is a no-op.
  ///
  /// @return Status indicating success or failure
  utils::Status close() {
    if (persistencePath_) {
      return save(persistencePath_->string());
    }
    return utils::OkStatus();
  }

  /// Check if collection recovered from WAL on load.
  ///
  /// @return true if WAL replay occurred during load, false otherwise
  bool recoveredFromWal() const { return recoveredFromWal_; }

  /// Get current LSN counter.
  ///
  /// @return The next LSN that will be assigned to an operation
  uint64_t currentLsn() const { return lsnCounter; }

  /// Get current TXID counter.
  ///
  /// @return The next TXID that will be assigned to an operation
  uint64_t currentTxid() const { return txidCounter; }

  void showMetadata(VectorID id) {
    utils::json j = utils::metadataToJson(metadata_.at(id));
    std::cout << j.dump(2) << "\n";
  }

 private:
  const CollectionConfig config_;
  HNSWConfig hnswConfig_;
  std::unique_ptr<HNSWIndex> pIndex_;
  std::unique_ptr<wal::WAL> pWal_;
  std::unordered_map<VectorID, Metadata> metadata_;
  uint64_t lsnCounter = 1;
  uint64_t txidCounter = 1;
  std::optional<std::filesystem::path> persistencePath_;
  uint64_t lastPersistedLsn_ = 0;
  bool recoveredFromWal_ = false;

  /// Parallel search implementation using thread pool.
  /// Divides queries among threads for concurrent execution.
  /// hnswlib is thread-safe for concurrent searches.
  static std::vector<std::vector<SearchResult>> parallelSearch(
      const HNSWIndex* index,
      const std::vector<std::vector<float>>& queries,
      uint32_t k,
      uint32_t ef) {

    const size_t numQueries = queries.size();
    std::vector<std::vector<SearchResult>> results(numQueries);

    // Determine thread count (use hardware concurrency, max 8)
    const size_t hwConcurrency = std::thread::hardware_concurrency();
    const size_t numThreads = std::min(hwConcurrency, std::min(size_t(8), numQueries));

    // Sequential fallback for single-threaded or single-query case
    if (numThreads <= 1 || numQueries <= 1) {
      for (size_t i = 0; i < numQueries; ++i) {
        results[i] = index->search(queries[i], k, ef);
      }
      return results;
    }

    // Parallel execution
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    auto worker = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        results[i] = index->search(queries[i], k, ef);
      }
    };

    const size_t queriesPerThread = (numQueries + numThreads - 1) / numThreads;

    for (size_t t = 0; t < numThreads; ++t) {
      size_t start = t * queriesPerThread;
      size_t end = std::min(start + queriesPerThread, numQueries);
      if (start < end) {
        threads.emplace_back(worker, start, end);
      }
    }

    // Join all threads
    for (auto& thread : threads) {
      thread.join();
    }

    return results;
  }

  void initializeWal() {
    if (persistencePath_) {
      namespace fs = std::filesystem;
      fs::path walDir = *persistencePath_ / "wal";
      pWal_ = std::make_unique<wal::WAL>(walDir);

      // Write initial header if WAL file doesn't exist yet
      fs::path walFile = walDir / "db.wal";
      if (!fs::exists(walFile)) {
        wal::Header header;
        header.magic = wal::kWalMagic;
        header.creationTime = static_cast<uint64_t>(time(nullptr));
        header.headerCrc32 = header.computeCrc32();
        (void)pWal_->writeHeader(header);
      }
    }
  }

  /// Replay WAL entries with LSN > fromLsn to recover uncommitted changes.
  ///
  /// @param fromLsn Only replay entries with LSN strictly greater than this
  /// @return Status indicating success or failure (fail-fast on corruption)
  utils::Status replayWal(uint64_t fromLsn) {
    if (!pWal_) {
      return utils::OkStatus();
    }

    wal::Result<std::vector<wal::Entry>> entriesResult = pWal_->readAll();
    if (!entriesResult.ok()) {
      // If WAL is empty or doesn't exist yet, that's OK
      if (entriesResult.status().code() == utils::StatusCode::kEof ||
          entriesResult.status().code() == utils::StatusCode::kNotFound) {
        return utils::OkStatus();
      }
      // Fail-fast on corruption
      return entriesResult.status();
    }

    const std::vector<wal::Entry> &entries = entriesResult.value();
    uint64_t maxLsn = lsnCounter;
    uint64_t maxTxid = txidCounter;
    uint64_t replayedCount = 0;

    for (const wal::Entry &entry : entries) {
      // Skip entries already persisted
      if (entry.lsn <= fromLsn) {
        continue;
      }

      // Track max LSN/TXID for counter updates
      if (entry.lsn >= maxLsn) {
        maxLsn = entry.lsn + 1;
      }
      if (entry.txid >= maxTxid) {
        maxTxid = entry.txid + 1;
      }

      // Replay based on operation type
      switch (entry.type) {
      case wal::OperationType::INSERT:
        // Replay insert directly to index (bypass WAL logging)
        if (!pIndex_->insert(entry.vectorID, entry.embedding)) {
          return utils::Status(utils::StatusCode::kInternal,
                               "Failed to replay INSERT for vector " +
                                   std::to_string(entry.vectorID));
        }
        ++replayedCount;
        break;

      case wal::OperationType::DELETE:
        // Replay delete using lazy deletion
        pIndex_->markDelete(entry.vectorID);
        metadata_.erase(entry.vectorID);
        ++replayedCount;
        break;

      default:
        // Ignore other operation types for now (COMMIT_TXN, ABORT_TXN, etc.)
        break;
      }
    }

    // Update counters to continue from where we left off
    lsnCounter = maxLsn;
    txidCounter = maxTxid;

    if (replayedCount > 0) {
      recoveredFromWal_ = true;
    }

    return utils::OkStatus();
  }
};

} // namespace arrow
  //
namespace arrow::utils {
inline json collectionConfigToJson(const CollectionConfig &config) {
  json j = json::object();
  j["name"] = config.name;
  j["dimensions"] = config.dimensions;
  j["metric"] = distanceMetricToJson(config.metric);
  j["dtype"] = dataTypeToJson(config.dtype);
  j["idxType"] = "HNSW"; // Currently only HNSW is supported
  return j;
}

inline CollectionConfig jsonToCollectionConfig(const json &j) {
  return CollectionConfig(
      j["name"].get<std::string>(), j["dimensions"].get<uint32_t>(),
      jsonToDistanceMetric(j["metric"]), jsonToDataType(j["dtype"]));
}

inline json hnswConfigToJson(const HNSWConfig &config) {
  json j = json::object();
  j["maxElements"] = config.maxElements;
  j["M"] = config.M;
  j["efConstruction"] = config.efConstruction;
  return j;
}

inline HNSWConfig jsonToHNSWConfig(const json &j) {
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

inline void exportCollectionConfigToJson(const CollectionConfig &config,
                                         const HNSWConfig &hnswConfig,
                                         const std::string &filepath,
                                         const RecoveryMetadata &recovery) {
  json j = collectionConfigToJson(config);
  j["hnsw"] = hnswConfigToJson(hnswConfig);

  // Add recovery metadata
  json recoveryJson = json::object();
  recoveryJson["lastPersistedLsn"] = recovery.lastPersistedLsn;
  recoveryJson["lastPersistedTxid"] = recovery.lastPersistedTxid;
  recoveryJson["cleanShutdown"] = recovery.cleanShutdown;
  j["recovery"] = recoveryJson;

  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }
  file << j.dump(2); // Pretty print with 2-space indent
  file.close();
}

inline std::tuple<CollectionConfig, HNSWConfig, RecoveryMetadata>
importConfigsFromJson(const std::string &filepath) {
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
                              : HNSWConfig{}; // Use defaults if not present

  // Parse recovery metadata if present
  RecoveryMetadata recovery;
  if (j.contains("recovery")) {
    const auto &r = j["recovery"];
    if (r.contains("lastPersistedLsn")) {
      recovery.lastPersistedLsn = r["lastPersistedLsn"].get<uint64_t>();
    }
    if (r.contains("lastPersistedTxid")) {
      recovery.lastPersistedTxid = r["lastPersistedTxid"].get<uint64_t>();
    }
    if (r.contains("cleanShutdown")) {
      recovery.cleanShutdown = r["cleanShutdown"].get<bool>();
    }
  }

  return {config, hnswConfig, recovery};
}
} // namespace arrow::utils

#endif // ARROWDB_H
