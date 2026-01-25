// Copyright 2025 ArrowDB
#ifndef ARROW_CLIENT_H
#define ARROW_CLIENT_H

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
/// Client client(options);
///
/// auto collection = client.createCollection("my_collection", {
///     .name = "my_collection",
///     .dimensions = 384,
///     .space = IndexSpace::Cosine,
/// });
///
/// if (collection.ok()) {
///     collection.value()->insert(1, embedding);
/// }
///
/// client.close();
/// ```
class Client {
public:
    /// Construct a new ArrowDB instance.
    ///
    /// @param options Client configuration options
    explicit Client(const ClientOptions& options);

    /// Destructor - closes all collections gracefully.
    ~Client();

    // Non-copyable
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Movable
    Client(Client&&) noexcept;
    Client& operator=(Client&&) noexcept;

    /// Create a new collection.
    /// IndexConfig is embedded in CollectionConfig.
    ///
    /// @param name Collection name (must be unique)
    /// @param config Collection configuration (includes index config)
    /// @return Pointer to the created collection, or error status
    utils::Result<Collection*> createCollection(const std::string& name,
                                                 const CollectionConfig& config);

    /// Get an existing collection by name.
    ///
    /// @param name Collection name
    /// @return Pointer to the collection, or error if not found
    utils::Result<Collection*> getCollection(const std::string& name);

    /// Create a collection if it doesn't exist, or get the existing one.
    ///
    /// This is a convenience method that:
    /// - Returns the existing collection if it already exists (in memory or on disk)
    /// - Creates a new collection with the given config if it doesn't exist
    ///
    /// @param name Collection name
    /// @param config Collection configuration (used only if creating new)
    /// @return Pointer to the collection (existing or newly created)
    utils::Result<Collection*> getOrCreateCollection(const std::string& name,
                                                      const CollectionConfig& config);

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

#endif // ARROW_CLIENT_H
