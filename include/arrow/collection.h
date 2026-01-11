// Copyright 2025 ArrowDB
#ifndef ARROW_COLLECTION_H
#define ARROW_COLLECTION_H

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "arrow/options.h"
#include "arrow/types.h"
#include "arrow/utils/result.h"
#include "arrow/utils/status.h"

namespace arrow {

/**
 * @brief A collection of vectors with a specific configuration.
 *
 * Collection represents a named group of vectors that share the same
 * dimension, distance metric, and data type. It serves as the primary
 * interface for vector database operations.
 *
 * Default HNSW parameters are optimized for large datasets (100K+ vectors):
 * - M=64: Provides 91-92% recall@10 for 100K vectors
 * - efConstruction=200: Balanced build time and quality
 * - Default EF search=200: Provides ~91% recall@10 for 100K vectors
 */
class Collection {
public:
    /// Constructs a Collection with the given configuration.
    explicit Collection(const CollectionConfig& config);

    /// Constructs a Collection with custom index options.
    Collection(const CollectionConfig& config, const IndexOptions& indexOptions);

    /// Constructs a Collection with persistence path for durability.
    Collection(const CollectionConfig& config,
               const std::filesystem::path& persistencePath);

    /// Constructs a Collection with custom index options and persistence.
    Collection(const CollectionConfig& config,
               const IndexOptions& indexOptions,
               const std::filesystem::path& persistencePath);

    /// Destructor
    ~Collection();

    // Move operations
    Collection(Collection&&) noexcept;
    Collection& operator=(Collection&&) noexcept;

    // Non-copyable
    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;

    /// Get the collection name.
    const std::string& name() const;

    /// Get the vector dimension.
    uint32_t dimension() const;

    /// Get the distance metric.
    DistanceMetric metric() const;

    /// Get the number of vectors in the collection.
    size_t size() const;

    /// Insert a vector into the collection.
    ///
    /// @param id Unique identifier for the vector
    /// @param vec Vector data (must match collection dimension)
    /// @return Status indicating success or failure
    utils::Status insert(VectorID id, const std::vector<float>& vec);

    /// Insert a batch of vectors with partial success semantics.
    ///
    /// @param batch Vector of (id, vector) pairs to insert
    /// @return Result containing BatchInsertResult with per-vector status
    utils::Result<BatchInsertResult> insertBatch(
        const std::vector<std::pair<VectorID, std::vector<float>>>& batch);

    /// Set metadata for a vector.
    ///
    /// @param id Vector identifier
    /// @param metadata Metadata to associate with the vector
    void setMetadata(VectorID id, const Metadata& metadata);

    /// Search for k nearest neighbors.
    ///
    /// @param query Query vector (must match collection dimension)
    /// @param k Number of results to return
    /// @param ef Search beam width (higher = better recall, slower)
    /// @return Vector of search results (id, score pairs)
    std::vector<IndexSearchResult> search(const std::vector<float>& query,
                                          uint32_t k,
                                          uint32_t ef = 200) const;

    /// Query for k nearest neighbors with metadata.
    ///
    /// @param query Query vector (must match collection dimension)
    /// @param k Number of results to return
    /// @param ef Search beam width (higher = better recall, slower)
    /// @return SearchResult with hits containing id, score, and metadata
    SearchResult query(const std::vector<float>& query,
                       uint32_t k,
                       uint32_t ef = 200) const;

    /// Search for k nearest neighbors for multiple queries in parallel.
    ///
    /// @param queries Vector of query vectors
    /// @param k Number of results per query
    /// @param ef Search beam width
    /// @return Result containing vector of result vectors
    utils::Result<std::vector<std::vector<IndexSearchResult>>> searchBatch(
        const std::vector<std::vector<float>>& queries,
        uint32_t k,
        uint32_t ef = 200) const;

    /// Remove a vector from the collection.
    ///
    /// @param id Vector identifier to remove
    /// @return Status indicating success or failure
    utils::Status remove(VectorID id);

    /// Save the collection to disk.
    ///
    /// @param directoryPath Directory path where the collection will be saved
    /// @return Status indicating success or failure
    utils::Status save(const std::string& directoryPath);

    /// Load a collection from disk.
    ///
    /// @param directoryPath Directory path where the collection is stored
    /// @return Result containing the loaded Collection or error
    static utils::Result<Collection> load(const std::string& directoryPath);

    /// Close the collection and save state.
    utils::Status close();

    /// Check if collection recovered from WAL on load.
    bool recoveredFromWal() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;

    // Private constructor used by load()
    Collection(std::unique_ptr<Impl> impl);
};

} // namespace arrow

#endif // ARROW_COLLECTION_H
