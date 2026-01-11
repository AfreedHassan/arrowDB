// Copyright 2025 ArrowDB
#include "arrow/db.h"
#include "arrow/collection.h"

#include <filesystem>
#include <unordered_map>

namespace arrow {

/// ArrowDB implementation
class ArrowDB::Impl {
public:
    explicit Impl(const ClientOptions& options)
        : options_(options) {
        // Create data directory if it doesn't exist
        if (!options_.data_dir.empty()) {
            std::filesystem::create_directories(options_.data_dir);
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
                                                 const CollectionConfig& config,
                                                 const IndexOptions& indexOptions) {
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
        if (!options_.data_dir.empty()) {
            collectionPath = options_.data_dir / name;
        }

        // Create the collection using new Pimpl-based API
        std::unique_ptr<Collection> collection;
        if (collectionPath.empty()) {
            collection = std::make_unique<Collection>(effectiveConfig, indexOptions);
        } else {
            collection = std::make_unique<Collection>(effectiveConfig, indexOptions, collectionPath);
        }

        Collection* ptr = collection.get();
        collections_[name] = std::move(collection);

        return ptr;
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
        if (!options_.data_dir.empty()) {
            std::filesystem::path collectionPath = options_.data_dir / name;
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
        return options_.data_dir;
    }

private:
    ClientOptions options_;
    std::unordered_map<std::string, std::unique_ptr<Collection>> collections_;

    void loadExistingCollections() {
        if (options_.data_dir.empty() || !std::filesystem::exists(options_.data_dir)) {
            return;
        }

        for (const auto& entry : std::filesystem::directory_iterator(options_.data_dir)) {
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
ArrowDB::ArrowDB(const ClientOptions& options)
    : pImpl_(std::make_unique<Impl>(options)) {}

ArrowDB::~ArrowDB() = default;

ArrowDB::ArrowDB(ArrowDB&&) noexcept = default;
ArrowDB& ArrowDB::operator=(ArrowDB&&) noexcept = default;

utils::Result<Collection*> ArrowDB::createCollection(const std::string& name,
                                                      const CollectionConfig& config) {
    return pImpl_->createCollection(name, config, IndexOptions{});
}

utils::Result<Collection*> ArrowDB::createCollection(const std::string& name,
                                                      const CollectionConfig& config,
                                                      const IndexOptions& indexOptions) {
    return pImpl_->createCollection(name, config, indexOptions);
}

utils::Result<Collection*> ArrowDB::getCollection(const std::string& name) {
    return pImpl_->getCollection(name);
}

utils::Status ArrowDB::dropCollection(const std::string& name) {
    return pImpl_->dropCollection(name);
}

std::vector<std::string> ArrowDB::listCollections() const {
    return pImpl_->listCollections();
}

bool ArrowDB::hasCollection(const std::string& name) const {
    return pImpl_->hasCollection(name);
}

utils::Status ArrowDB::close() {
    return pImpl_->close();
}

const std::filesystem::path& ArrowDB::dataDir() const {
    return pImpl_->dataDir();
}

} // namespace arrow
