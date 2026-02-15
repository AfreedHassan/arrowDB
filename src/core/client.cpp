// Copyright 2025 ArrowDB
#include "arrow/client.h"
#include "arrow/collection.h"

#include <filesystem>
#include <unordered_map>

namespace arrow {

/// ArrowDB implementation
class Client::Impl {
public:
    explicit Impl(const ClientOptions& options)
        : options_(options) {
        // Create data directory if it doesn't exist
        if (!options_.dataDir.empty()) {
            std::filesystem::create_directories(options_.dataDir);
        }
        // Load existing collections from data directory
        loadExistingCollections();
    }

    ~Impl() {
        // Close all collections gracefully
        for (auto& [name, collection] : collections_) {
            (void)collection->close();
        }
    }

    utils::Result<Collection*> createCollection(const std::string& name,
                                                 const CollectionConfig& config) {
        // Check if collection already exists
        if (collections_.count(name) > 0) {
            return utils::Status(utils::StatusCode::kAlreadyExists,
                                "Collection already exists: " + name);
        }

        // Create config with name if not set
        CollectionConfig effectiveConfig = config;
        if (effectiveConfig.name.empty()) {
            effectiveConfig.name = name;
        }

        // Determine persistence path
        std::filesystem::path collectionPath;
        if (!options_.dataDir.empty()) {
            collectionPath = options_.dataDir / name;
        }

        // Create the collection (IndexConfig is embedded in CollectionConfig)
        std::unique_ptr<Collection> collection;
        if (collectionPath.empty()) {
            collection = std::make_unique<Collection>(effectiveConfig);
        } else {
            collection = std::make_unique<Collection>(effectiveConfig, collectionPath);
        }

        Collection* ptr = collection.get();
        collections_[name] = std::move(collection);

        return ptr;
    }

    utils::Result<Collection*> getOrCreateCollection(const std::string& name,
                                                     const CollectionConfig& config) {

        auto it = collections_.find(name);
        if (it != collections_.end()) {
            return it->second.get();
        }

        if (!options_.dataDir.empty()) {
            std::filesystem::path collectionPath = options_.dataDir / name;
            std::filesystem::path metaPath = collectionPath / "meta.json";
            if (std::filesystem::exists(metaPath)) {
                auto loadResult = Collection::load(collectionPath.string());
                if (loadResult.ok()) {
                    auto collection = std::make_unique<Collection>(std::move(loadResult.value()));
                    Collection* ptr = collection.get();
                    collections_[name] = std::move(collection);
                    return ptr;
                }
            }
        }
        return createCollection(name, config);
    }

    utils::Result<Collection*> getCollection(const std::string& name) {
        auto it = collections_.find(name);
        if (it == collections_.end()) {
            return utils::Status(utils::StatusCode::kNotFound,
                                "Collection not found: " + name);
        }
        return it->second.get();
    }

    utils::Status dropCollection(const std::string& name) {
        auto it = collections_.find(name);
        if (it == collections_.end()) {
            return utils::Status(utils::StatusCode::kNotFound,
                                "Collection not found: " + name);
        }

        // Close and remove the collection
        (void)it->second->close();
        collections_.erase(it);

        // Remove from disk if data_dir is set
        if (!options_.dataDir.empty()) {
            std::filesystem::path collectionPath = options_.dataDir / name;
            if (std::filesystem::exists(collectionPath)) {
                std::filesystem::remove_all(collectionPath);
            }
        }

        return utils::OkStatus();
    }

    std::vector<std::string> listCollections() const {
        std::vector<std::string> names;
        names.reserve(collections_.size());
        for (const auto& [name, _] : collections_) {
            names.push_back(name);
        }
        return names;
    }

    bool hasCollection(const std::string& name) const {
        return collections_.count(name) > 0;
    }

    utils::Status close() {
        for (auto& [name, collection] : collections_) {
            utils::Status status = collection->close();
            if (!status.ok()) {
                return status;
            }
        }
        collections_.clear();
        return utils::OkStatus();
    }

    const std::filesystem::path& dataDir() const {
        return options_.dataDir;
    }

private:
    ClientOptions options_;
    std::unordered_map<std::string, std::unique_ptr<Collection>> collections_;

    void loadExistingCollections() {
        if (options_.dataDir.empty() || !std::filesystem::exists(options_.dataDir)) {
            return;
        }

        for (const auto& entry : std::filesystem::directory_iterator(options_.dataDir)) {
            if (!entry.is_directory()) continue;

            std::filesystem::path metaPath = entry.path() / "meta.json";
            if (!std::filesystem::exists(metaPath)) continue;

            // Try to load the collection
            std::string name = entry.path().filename().string();
            auto result = Collection::load(entry.path().string());
            if (result.ok()) {
                collections_[name] = std::make_unique<Collection>(std::move(result.value()));
            }
        }
    }
};

// ArrowDB public interface implementation
Client::Client(const ClientOptions& options)
    : pImpl_(std::make_unique<Impl>(options)) {}

Client::Client(std::filesystem::path dataDir)
    : Client(ClientOptions{.dataDir = std::move(dataDir)}) {}

Client::~Client() = default;

Client::Client(Client&&) noexcept = default;
Client& Client::operator=(Client&&) noexcept = default;

utils::Result<Collection*> Client::createCollection(const std::string& name,
                                                      const CollectionConfig& config) {
    return pImpl_->createCollection(name, config);
}

utils::Result<Collection*> Client::getCollection(const std::string& name) {
    return pImpl_->getCollection(name);
}

utils::Result<Collection*> Client::getOrCreateCollection(const std::string& name,
                                                         const CollectionConfig& config) {
    return pImpl_->getOrCreateCollection(name, config);
}

utils::Result<Collection*> Client::getOrCreateCollection(const std::string& name) {
    static constexpr uint32_t kDefaultEmbeddingDim = 384;  // all-MiniLM-L6-v2
    return pImpl_->getOrCreateCollection(name, {.dimensions = kDefaultEmbeddingDim});
}

utils::Status Client::dropCollection(const std::string& name) {
    return pImpl_->dropCollection(name);
}

std::vector<std::string> Client::listCollections() const {
    return pImpl_->listCollections();
}

bool Client::hasCollection(const std::string& name) const {
    return pImpl_->hasCollection(name);
}

utils::Status Client::close() {
    return pImpl_->close();
}

const std::filesystem::path& Client::dataDir() const {
    return pImpl_->dataDir();
}

} // namespace arrow
