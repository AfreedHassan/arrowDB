// Copyright 2025 ArrowDB
#ifndef ARROW_DB_H
#define ARROW_DB_H

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "options.h"
#include "utils/result.h"
#include "utils/status.h"

namespace arrow {

// Forward declaration
class Collection;

/// ArrowDB - Main database interface for managing vector collections.
///
/// ArrowDB manages multiple collections, handles persistence, and provides
/// the primary entry point for interacting with the vector database.
///
/// Example usage:
/// ```cpp
/// ClientOptions options{.data_dir = "/path/to/data"};
/// ArrowDB db(options);
///
/// auto collection = db.createCollection("my_collection", {
///     .name = "my_collection",
///     .dimensions = 384,
///     .metric = DistanceMetric::Cosine
/// });
///
/// if (collection.ok()) {
///     collection.value()->insert(1, embedding);
/// }
///
/// db.close();
/// ```
class ArrowDB {
public:
    /// Construct a new ArrowDB instance.
    ///
    /// @param options Client configuration options
    explicit ArrowDB(const ClientOptions& options);

    /// Destructor - closes all collections gracefully.
    ~ArrowDB();

    // Non-copyable
    ArrowDB(const ArrowDB&) = delete;
    ArrowDB& operator=(const ArrowDB&) = delete;

    // Movable
    ArrowDB(ArrowDB&&) noexcept;
    ArrowDB& operator=(ArrowDB&&) noexcept;

    /// Create a new collection.
    ///
    /// @param name Collection name (must be unique)
    /// @param config Collection configuration
    /// @return Pointer to the created collection, or error status
    utils::Result<Collection*> createCollection(const std::string& name,
                                                 const CollectionConfig& config);

    /// Create a new collection with custom index options.
    ///
    /// @param name Collection name (must be unique)
    /// @param config Collection configuration
    /// @param indexOptions Custom index configuration
    /// @return Pointer to the created collection, or error status
    utils::Result<Collection*> createCollection(const std::string& name,
                                                 const CollectionConfig& config,
                                                 const IndexOptions& indexOptions);

    /// Get an existing collection by name.
    ///
    /// @param name Collection name
    /// @return Pointer to the collection, or error if not found
    utils::Result<Collection*> getCollection(const std::string& name);

    /// Drop a collection.
    ///
    /// @param name Collection name
    /// @return Status indicating success or failure
    utils::Status dropCollection(const std::string& name);

    /// List all collection names.
    ///
    /// @return Vector of collection names
    std::vector<std::string> listCollections() const;

    /// Check if a collection exists.
    ///
    /// @param name Collection name
    /// @return true if collection exists
    bool hasCollection(const std::string& name) const;

    /// Close the database and all collections.
    ///
    /// Saves all collections to disk and releases resources.
    /// @return Status indicating success or failure
    utils::Status close();

    /// Get the data directory path.
    const std::filesystem::path& dataDir() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace arrow

#endif // ARROW_DB_H
