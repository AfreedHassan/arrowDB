// Copyright 2025 ArrowDB
#include "arrow/collection.h"
#include "arrow/utils/utils.h"
#include "internal/hnsw_index.h"
#include "internal/wal.h"

#include <fstream>
#include <iostream>
#include <thread>
#include <unordered_map>

namespace arrow {

// Internal configuration (not exposed in public header)
struct InternalConfig {
    std::string name;
    uint32_t dimensions;
    DistanceMetric metric;
    DataType dtype = DataType::Float32;
};

// Recovery metadata for crash recovery
struct RecoveryMetadata {
    uint64_t lastPersistedLsn = 0;
    uint64_t lastPersistedTxid = 0;
    bool cleanShutdown = true;
};

// JSON conversion utilities (internal)
namespace {

utils::json internalConfigToJson(const InternalConfig& config) {
    utils::json j = utils::json::object();
    j["name"] = config.name;
    j["dimensions"] = config.dimensions;
    j["metric"] = utils::distanceMetricToJson(config.metric);
    j["dtype"] = utils::dataTypeToJson(config.dtype);
    j["idxType"] = "HNSW";
    return j;
}

InternalConfig jsonToInternalConfig(const utils::json& j) {
    InternalConfig config;
    config.name = j["name"].get<std::string>();
    config.dimensions = j["dimensions"].get<uint32_t>();
    config.metric = utils::jsonToDistanceMetric(j["metric"]);
    config.dtype = utils::jsonToDataType(j["dtype"]);
    return config;
}

utils::json hnswConfigToJson(const HNSWConfig& config) {
    utils::json j = utils::json::object();
    j["maxElements"] = config.maxElements;
    j["M"] = config.M;
    j["efConstruction"] = config.efConstruction;
    return j;
}

HNSWConfig jsonToHNSWConfig(const utils::json& j) {
    HNSWConfig config;
    if (j.contains("maxElements")) config.maxElements = j["maxElements"].get<uint32_t>();
    if (j.contains("M")) config.M = j["M"].get<uint32_t>();
    if (j.contains("efConstruction")) config.efConstruction = j["efConstruction"].get<uint32_t>();
    return config;
}

void exportConfigToJson(const InternalConfig& config,
                        const HNSWConfig& hnswConfig,
                        const std::string& filepath,
                        const RecoveryMetadata& recovery) {
    utils::json j = internalConfigToJson(config);
    j["hnsw"] = hnswConfigToJson(hnswConfig);

    utils::json recoveryJson = utils::json::object();
    recoveryJson["lastPersistedLsn"] = recovery.lastPersistedLsn;
    recoveryJson["lastPersistedTxid"] = recovery.lastPersistedTxid;
    recoveryJson["cleanShutdown"] = recovery.cleanShutdown;
    j["recovery"] = recoveryJson;

    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    file << j.dump(2);
    file.close();
}

std::tuple<InternalConfig, HNSWConfig, RecoveryMetadata>
importConfigFromJson(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    utils::json j;
    file >> j;
    file.close();

    InternalConfig config = jsonToInternalConfig(j);
    HNSWConfig hnswConfig = j.contains("hnsw") ? jsonToHNSWConfig(j["hnsw"]) : HNSWConfig{};

    RecoveryMetadata recovery;
    if (j.contains("recovery")) {
        const auto& r = j["recovery"];
        if (r.contains("lastPersistedLsn")) recovery.lastPersistedLsn = r["lastPersistedLsn"].get<uint64_t>();
        if (r.contains("lastPersistedTxid")) recovery.lastPersistedTxid = r["lastPersistedTxid"].get<uint64_t>();
        if (r.contains("cleanShutdown")) recovery.cleanShutdown = r["cleanShutdown"].get<bool>();
    }

    return {config, hnswConfig, recovery};
}

} // anonymous namespace

class Collection::Impl {
public:
    InternalConfig config_;
    HNSWConfig hnswConfig_;
    std::unique_ptr<HNSWIndex> pIndex_;
    std::unique_ptr<wal::WAL> pWal_;
    std::unordered_map<VectorID, Metadata> metadata_;
    uint64_t lsnCounter = 1;
    uint64_t txidCounter = 1;
    std::optional<std::filesystem::path> persistencePath_;
    uint64_t lastPersistedLsn_ = 0;
    bool recoveredFromWal_ = false;

    Impl(const CollectionConfig& config, const IndexOptions& indexOptions)
        : config_{config.name, config.dimensions, config.metric, DataType::Float32},
          hnswConfig_{indexOptions.max_elements, indexOptions.M, indexOptions.ef_construction},
          pIndex_(std::make_unique<HNSWIndex>(config.dimensions, config.metric, hnswConfig_)) {}

    Impl(const CollectionConfig& config, const IndexOptions& indexOptions,
         const std::filesystem::path& persistencePath)
        : config_{config.name, config.dimensions, config.metric, DataType::Float32},
          hnswConfig_{indexOptions.max_elements, indexOptions.M, indexOptions.ef_construction},
          pIndex_(std::make_unique<HNSWIndex>(config.dimensions, config.metric, hnswConfig_)),
          persistencePath_(persistencePath) {
        initializeWal();
    }

    void initializeWal() {
        if (persistencePath_) {
            namespace fs = std::filesystem;
            fs::path walDir = *persistencePath_ / "wal";
            pWal_ = std::make_unique<wal::WAL>(walDir);

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

    utils::Status replayWal(uint64_t fromLsn) {
        if (!pWal_) return utils::OkStatus();

        wal::Result<std::vector<wal::Entry>> entriesResult = pWal_->readAll();
        if (!entriesResult.ok()) {
            if (entriesResult.status().code() == utils::StatusCode::kEof ||
                entriesResult.status().code() == utils::StatusCode::kNotFound) {
                return utils::OkStatus();
            }
            return entriesResult.status();
        }

        const std::vector<wal::Entry>& entries = entriesResult.value();
        uint64_t maxLsn = lsnCounter;
        uint64_t maxTxid = txidCounter;
        uint64_t replayedCount = 0;

        for (const wal::Entry& entry : entries) {
            if (entry.lsn <= fromLsn) continue;

            if (entry.lsn >= maxLsn) maxLsn = entry.lsn + 1;
            if (entry.txid >= maxTxid) maxTxid = entry.txid + 1;

            switch (entry.type) {
            case wal::OperationType::INSERT:
                if (!pIndex_->insert(entry.vectorID, entry.embedding)) {
                    return utils::Status(utils::StatusCode::kInternal,
                                        "Failed to replay INSERT for vector " +
                                        std::to_string(entry.vectorID));
                }
                ++replayedCount;
                break;
            case wal::OperationType::DELETE:
                pIndex_->markDelete(entry.vectorID);
                metadata_.erase(entry.vectorID);
                ++replayedCount;
                break;
            default:
                break;
            }
        }

        lsnCounter = maxLsn;
        txidCounter = maxTxid;
        if (replayedCount > 0) recoveredFromWal_ = true;

        return utils::OkStatus();
    }

    static std::vector<std::vector<IndexSearchResult>> parallelSearch(
        const HNSWIndex* index,
        const std::vector<std::vector<float>>& queries,
        uint32_t k,
        uint32_t ef) {

        const size_t numQueries = queries.size();
        std::vector<std::vector<IndexSearchResult>> results(numQueries);

        const size_t hwConcurrency = std::thread::hardware_concurrency();
        const size_t numThreads = std::min(hwConcurrency, std::min(size_t(8), numQueries));

        if (numThreads <= 1 || numQueries <= 1) {
            for (size_t i = 0; i < numQueries; ++i) {
                results[i] = index->search(queries[i], k, ef);
            }
            return results;
        }

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

        for (auto& thread : threads) {
            thread.join();
        }

        return results;
    }
};

// ─────────────────────────────────────────────────────────────
// Collection public methods
// ─────────────────────────────────────────────────────────────

Collection::Collection(const CollectionConfig& config)
    : pImpl_(std::make_unique<Impl>(config, IndexOptions{})) {}

Collection::Collection(const CollectionConfig& config, const IndexOptions& indexOptions)
    : pImpl_(std::make_unique<Impl>(config, indexOptions)) {}

Collection::Collection(const CollectionConfig& config,
                       const std::filesystem::path& persistencePath)
    : pImpl_(std::make_unique<Impl>(config, IndexOptions{}, persistencePath)) {}

Collection::Collection(const CollectionConfig& config,
                       const IndexOptions& indexOptions,
                       const std::filesystem::path& persistencePath)
    : pImpl_(std::make_unique<Impl>(config, indexOptions, persistencePath)) {}

Collection::Collection(std::unique_ptr<Impl> impl) : pImpl_(std::move(impl)) {}

Collection::~Collection() = default;
Collection::Collection(Collection&&) noexcept = default;
Collection& Collection::operator=(Collection&&) noexcept = default;

const std::string& Collection::name() const { return pImpl_->config_.name; }
uint32_t Collection::dimension() const { return pImpl_->config_.dimensions; }
DistanceMetric Collection::metric() const { return pImpl_->config_.metric; }
size_t Collection::size() const { return pImpl_->pIndex_->size(); }
bool Collection::recoveredFromWal() const { return pImpl_->recoveredFromWal_; }

utils::Status Collection::insert(VectorID id, const std::vector<float>& vec) {
    if (vec.size() != pImpl_->config_.dimensions) {
        return utils::Status(
            utils::StatusCode::kDimensionMismatch,
            "Vector dimension mismatch: expected " + std::to_string(pImpl_->config_.dimensions) +
            ", got " + std::to_string(vec.size()));
    }

    wal::Entry entry{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = pImpl_->lsnCounter++,
        .txid = pImpl_->txidCounter++,
        .headerCRC = 0,
        .payloadLength = 0,
        .vectorID = id,
        .dimension = pImpl_->config_.dimensions,
        .padding = 0,
        .embedding = vec,
        .payloadCRC = 0
    };
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    entry.payloadLength = entry.computePayloadLength();

    if (pImpl_->pWal_) {
        wal::Status status = pImpl_->pWal_->log(entry);
        if (!status.ok()) return status;
    }

    if (!pImpl_->pIndex_->insert(id, vec)) {
        return utils::Status(utils::StatusCode::kInternal, "Insert failed");
    }
    return utils::OkStatus();
}

utils::Result<BatchInsertResult> Collection::insertBatch(
    const std::vector<std::pair<VectorID, std::vector<float>>>& batch) {

    BatchInsertResult result;
    result.results.resize(batch.size());
    result.successCount = 0;
    result.failureCount = 0;

    std::vector<bool> validDimensions(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        validDimensions[i] = (batch[i].second.size() == pImpl_->config_.dimensions);
    }

    std::vector<wal::Entry> walEntries;
    walEntries.reserve(batch.size());

    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& [id, vec] = batch[i];

        if (!validDimensions[i]) {
            result.results[i] = {
                id,
                utils::Status(utils::StatusCode::kDimensionMismatch, "Vector dimension mismatch")
            };
            result.failureCount++;
            continue;
        }

        wal::Entry entry{
            .type = wal::OperationType::INSERT,
            .version = 1,
            .lsn = pImpl_->lsnCounter++,
            .txid = pImpl_->txidCounter++,
            .headerCRC = 0,
            .payloadLength = 0,
            .vectorID = id,
            .dimension = pImpl_->config_.dimensions,
            .padding = 0,
            .embedding = vec,
            .payloadCRC = 0
        };
        entry.headerCRC = entry.computeHeaderCrc();
        entry.payloadCRC = entry.computePayloadCrc();
        entry.payloadLength = entry.computePayloadLength();
        walEntries.push_back(std::move(entry));
    }

    if (pImpl_->pWal_ && !walEntries.empty()) {
        utils::Status walStatus = pImpl_->pWal_->logBatch(walEntries);
        if (!walStatus.ok()) {
            pImpl_->lsnCounter -= walEntries.size();
            pImpl_->txidCounter -= walEntries.size();
            return walStatus;
        }
    }

    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& [id, vec] = batch[i];
        if (!validDimensions[i]) continue;

        if (pImpl_->pIndex_->insert(id, vec)) {
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

void Collection::setMetadata(VectorID id, const Metadata& metadata) {
    pImpl_->metadata_[id] = metadata;
}

std::vector<IndexSearchResult> Collection::search(
    const std::vector<float>& query, uint32_t k, uint32_t ef) const {
    return pImpl_->pIndex_->search(query, k, ef);
}

SearchResult Collection::query(
    const std::vector<float>& queryVec, uint32_t k, uint32_t ef) const {
    auto indexResults = pImpl_->pIndex_->search(queryVec, k, ef);
    SearchResult result;
    result.hits.reserve(indexResults.size());

    for (const auto& ir : indexResults) {
        ScoredDocument doc;
        doc.id = ir.id;
        doc.score = ir.score;

        auto metaIt = pImpl_->metadata_.find(ir.id);
        if (metaIt != pImpl_->metadata_.end()) {
            doc.metadata = utils::metadataToJson(metaIt->second);
        } else {
            doc.metadata = nlohmann::json::object();
        }
        result.hits.push_back(std::move(doc));
    }

    return result;
}

utils::Result<std::vector<std::vector<IndexSearchResult>>> Collection::searchBatch(
    const std::vector<std::vector<float>>& queries, uint32_t k, uint32_t ef) const {

    for (size_t i = 0; i < queries.size(); ++i) {
        if (queries[i].size() != pImpl_->config_.dimensions) {
            return utils::Result<std::vector<std::vector<IndexSearchResult>>>(
                utils::Status(utils::StatusCode::kDimensionMismatch,
                             "Query " + std::to_string(i) + " dimension mismatch"));
        }
    }

    return Impl::parallelSearch(pImpl_->pIndex_.get(), queries, k, ef);
}

utils::Status Collection::remove(VectorID id) {
    wal::Entry entry{
        .type = wal::OperationType::DELETE,
        .version = 1,
        .lsn = pImpl_->lsnCounter++,
        .txid = pImpl_->txidCounter++,
        .headerCRC = 0,
        .payloadLength = 0,
        .vectorID = id,
        .dimension = 0,
        .padding = 0,
        .embedding = {},
        .payloadCRC = 0
    };
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    entry.payloadLength = entry.computePayloadLength();

    if (pImpl_->pWal_) {
        wal::Status status = pImpl_->pWal_->log(entry);
        if (!status.ok()) return status;
    }

    wal::Status delStatus = pImpl_->pIndex_->markDelete(id);
    if (!delStatus.ok()) return delStatus;

    pImpl_->metadata_.erase(id);
    return utils::OkStatus();
}

utils::Status Collection::save(const std::string& directoryPath) {
    namespace fs = std::filesystem;

    fs::create_directories(directoryPath);

    RecoveryMetadata recovery{
        .lastPersistedLsn = (pImpl_->lsnCounter > 0) ? pImpl_->lsnCounter - 1 : 0,
        .lastPersistedTxid = (pImpl_->txidCounter > 0) ? pImpl_->txidCounter - 1 : 0,
        .cleanShutdown = true
    };

    std::string metaPath = (fs::path(directoryPath) / "meta.json").string();
    exportConfigToJson(pImpl_->config_, pImpl_->hnswConfig_, metaPath, recovery);

    std::string indexPath = (fs::path(directoryPath) / "index.bin").string();
    pImpl_->pIndex_->saveIndex(indexPath);

    if (!pImpl_->metadata_.empty()) {
        std::string metadataPath = (fs::path(directoryPath) / "metadata.json").string();
        utils::exportMetadataToJson(pImpl_->metadata_, metadataPath);
    }

    if (pImpl_->pWal_) {
        wal::Status status = pImpl_->pWal_->truncate();
        if (!status.ok()) return status;
    }

    pImpl_->lastPersistedLsn_ = recovery.lastPersistedLsn;
    return utils::OkStatus();
}

utils::Result<Collection> Collection::load(const std::string& directoryPath) {
    namespace fs = std::filesystem;

    if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
        return utils::Status(utils::StatusCode::kNotFound,
                            "Collection directory does not exist: " + directoryPath);
    }

    std::string metaPath = (fs::path(directoryPath) / "meta.json").string();
    if (!fs::exists(metaPath)) {
        return utils::Status(utils::StatusCode::kNotFound,
                            "meta.json not found in collection directory: " + directoryPath);
    }

    auto [internalCfg, hnswCfg, recoveryMeta] = importConfigFromJson(metaPath);

    CollectionConfig config{
        .name = internalCfg.name,
        .dimensions = internalCfg.dimensions,
        .metric = internalCfg.metric
    };
    IndexOptions indexOptions{
        .max_elements = hnswCfg.maxElements,
        .M = hnswCfg.M,
        .ef_construction = hnswCfg.efConstruction
    };

    auto impl = std::make_unique<Impl>(config, indexOptions, fs::path(directoryPath));

    std::string indexPath = (fs::path(directoryPath) / "index.bin").string();
    if (!fs::exists(indexPath)) {
        return utils::Status(utils::StatusCode::kNotFound,
                            "index.bin not found in collection directory: " + directoryPath);
    }
    impl->pIndex_->loadIndex(indexPath);

    std::string metadataPath = (fs::path(directoryPath) / "metadata.json").string();
    if (fs::exists(metadataPath)) {
        impl->metadata_ = utils::importMetadataFromJson(metadataPath);
    }

    impl->lastPersistedLsn_ = recoveryMeta.lastPersistedLsn;
    impl->lsnCounter = recoveryMeta.lastPersistedLsn + 1;
    impl->txidCounter = recoveryMeta.lastPersistedTxid + 1;

    fs::path walPath = fs::path(directoryPath) / "wal" / "db.wal";
    if (fs::exists(walPath)) {
        utils::Status replayStatus = impl->replayWal(recoveryMeta.lastPersistedLsn);
        if (!replayStatus.ok()) return replayStatus;
    }

    return Collection(std::move(impl));
}

utils::Status Collection::close() {
    if (pImpl_->persistencePath_) {
        return save(pImpl_->persistencePath_->string());
    }
    return utils::OkStatus();
}

} // namespace arrow
