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
 * dimension, distance space, and data type. It serves as the primary
 * interface for vector database operations.
 *
 * Default HNSW parameters are optimized for ≤100K vectors:
 * - M=16: Optimal for ≤100K vectors (benchmarked)
 * - efConstruction=200: Balanced build time and quality
 * - Default EF search=200: Provides ~91% recall@10
 */
class Collection {
public:
    /// Creates an in-memory Collection with the given configuration.
    explicit Collection(const CollectionConfig& config);

    /// Creates a persistent Collection with file locking and WAL.
    ///
    /// @param config Collection configuration
    /// @param persistencePath Directory for persistence data
    /// @return The Collection, or error (e.g., file lock held by another process)
    static utils::Result<Collection> create(
        const CollectionConfig& config,
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

    /// Get the distance space.
    Space space() const;

    /// Get the number of vectors in the collection.
    size_t size() const;

    /// Insert a vector with an auto-generated UUID.
    ///
    /// @param vec Vector data (must match collection dimension)
    /// @param metadata Optional metadata
    /// @return The generated VectorID, or error
    utils::Result<VectorID> insert(const std::vector<float>& vec, Metadata metadata = {});

    /// Insert a vector into the collection.
    ///
    /// @param id Unique identifier for the vector
    /// @param vec Vector data (must match collection dimension)
    /// @return Status indicating success or failure
    utils::Status insert(const VectorID& id, const std::vector<float>& vec, Metadata metadata = {});

    utils::Status insert(const std::vector<std::string>& data);

    /// Insert a document (vector + metadata + optional ID).
    ///
    /// @param doc Document with embedding, metadata, and optional ID
    /// @return The VectorID (generated if doc.id was empty), or error
    utils::Result<VectorID> insert(Document doc);

    /// Insert a batch of documents with partial success semantics.
    ///
    /// @param docs Vector of documents to insert
    /// @return Result containing BatchInsertResult with per-vector status
    utils::Result<BatchInsertResult> insertBatch(std::vector<Document> docs);

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
    /// @return Status indicating success or failure
    utils::Status setMetadata(const VectorID& id, const Metadata& metadata);

    /// Get metadata for a vector.
    ///
    /// @param id Vector identifier
    /// @return Result containing metadata or error if vector not found
    utils::Result<Metadata> getMetadata(const VectorID& id);

    /// Search for k nearest neighbors.
    ///
    /// @param query Query vector (must match collection dimension)
    /// @param k Number of results to return
    /// @param ef Search beam width (higher = better recall, slower)
    /// @return Vector of search results (id, score pairs)
    std::vector<IndexSearchResult> search(const std::vector<float>& query,
                                          uint32_t k,
                                          uint32_t ef = 200) const;

    SearchResult query(const std::string& query,
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

    /// Search for k nearest neighbors with metadata filtering.
    ///
    /// @param query Query vector (must match collection dimension)
    /// @param k Number of results to return
    /// @param filter Predicate applied to each candidate's metadata
    /// @param ef Search beam width (higher = better recall, slower)
    /// @return Vector of search results passing the filter
    std::vector<IndexSearchResult> search(const std::vector<float>& query,
                                          uint32_t k,
                                          MetadataFilter filter,
                                          uint32_t ef = 200) const;

    /// Retrieve a vector by ID.
    ///
    /// @param id Vector identifier
    /// @return Result containing the vector data or error
    utils::Result<std::vector<float>> get(const VectorID& id) const;

    /// Update vector data and/or metadata for an existing ID.
    ///
    /// @param id Vector identifier (must already exist)
    /// @param vec New vector data (must match collection dimension)
    /// @param metadata New metadata (replaces existing)
    /// @return Status indicating success or failure
    utils::Status update(const VectorID& id, const std::vector<float>& vec,
                         Metadata metadata = {});

    /// Insert or update a vector.
    ///
    /// @param id Vector identifier
    /// @param vec Vector data (must match collection dimension)
    /// @param metadata Metadata to associate with the vector
    /// @return Status indicating success or failure
    utils::Status upsert(const VectorID& id, const std::vector<float>& vec,
                         Metadata metadata = {});

    /// Remove a vector from the collection.
    ///
    /// @param id Vector identifier to remove
    /// @return Status indicating success or failure
    utils::Status remove(const VectorID& id);

    /// Apply post-build optimizations for best search performance.
    /// When quantization is enabled, switches to global quantization with
    /// integer-domain distance kernels. Also reorders the graph for cache locality.
    /// No-op if already optimized or quantization is disabled.
    utils::Status optimize();

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

    /// Collection statistics.
    struct Stats {
      size_t vectorCount = 0;
      size_t metadataCount = 0;
      size_t maxCapacity = 0;
      size_t dimensions = 0;
    };

    /// Get collection statistics.
    Stats stats() const;

    /// Print collection statistics as JSON to stdout.
    void printStats() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;

    // Private constructor used by load() and create()
    Collection(std::unique_ptr<Impl> impl);
};

} // namespace arrow

#endif // ARROW_COLLECTION_H
