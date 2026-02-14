// Copyright 2025 ArrowDB
#include "arrow/collection.h"
#include "arrow/utils/utils.h"
#include "embedder/embedder.h"
#include "internal/hnsw_index.h"
#include "internal/wal.h"
#include "internal/id_space.h"
#include "internal/collection_persistence.h"
#include "internal/file_lock.h"
#include "internal/log.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace arrow {

// ─────────────────────────────────────────────────────────────
// Input validation helpers
// ─────────────────────────────────────────────────────────────

static constexpr size_t kMaxBatchSize = 10000;
static constexpr size_t kMaxMetadataKeys = 256;
static constexpr size_t kMaxMetadataValueSize = 65536;  // 64KB per string value

static utils::Status validateVector(const std::vector<float>& vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    if (std::isnan(vec[i]) || std::isinf(vec[i])) {
      return utils::Status(utils::StatusCode::kInvalidArgument,
        "Vector contains NaN or Inf at index " + std::to_string(i));
    }
  }
  return utils::OkStatus();
}

static utils::Status validateMetadata(const Metadata& meta) {
  if (meta.size() > kMaxMetadataKeys)
    return utils::Status(utils::StatusCode::kInvalidArgument, "Too many metadata keys");
  for (const auto& [key, val] : meta) {
    if (key.size() > 256)
      return utils::Status(utils::StatusCode::kInvalidArgument, "Metadata key too long");
    if (auto* s = std::get_if<std::string>(&val); s && s->size() > kMaxMetadataValueSize)
      return utils::Status(utils::StatusCode::kInvalidArgument, "Metadata string value too large");
  }
  return utils::OkStatus();
}

// ─────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────

class Collection::Impl {
public:
  InternalConfig config_;
  HNSWConfig hnswConfig_;
  std::unique_ptr<HNSWIndex> pIndex_;
  std::unique_ptr<wal::WAL> pWal_;
  std::unordered_map<InternalID, Metadata> metadata_;
  IDSpace idSpace_;
  wal::EntryBuilder builder_;
  std::optional<std::filesystem::path> persistencePath_;
  uint64_t lastPersistedLsn_ = 0;
  bool recoveredFromWal_ = false;
  mutable std::shared_mutex mutex_;
  std::optional<FileLock> fileLock_;

  utils::Status writeDirtyShutdownMarker() {
    if (!persistencePath_) {
      return utils::OkStatus();
    }
    return CollectionPersistence::writeDirtyShutdownMarker(
      *persistencePath_,
      config_,
      hnswConfig_,
      builder_.currentLsn(),
      builder_.currentTxid()
    );
  }

  explicit Impl(const CollectionConfig &config)
      : config_{config.name, config.dimensions, config.space,
                DataType::Float32},
        hnswConfig_{config.index.max_elements, config.index.M,
                    config.index.ef_construction},
        pIndex_(std::make_unique<HNSWIndex>(config.dimensions, config.space,
                                            hnswConfig_)) {}

  Impl(const CollectionConfig &config,
       const std::filesystem::path &persistencePath)
      : config_{config.name, config.dimensions, config.space,
                DataType::Float32},
        hnswConfig_{config.index.max_elements, config.index.M,
                    config.index.ef_construction},
        pIndex_(std::make_unique<HNSWIndex>(config.dimensions, config.space,
                                            hnswConfig_)),
        persistencePath_(persistencePath) {
    // Acquire exclusive file lock
    auto lockResult = FileLock::acquire(persistencePath);
    if (lockResult.ok()) {
      fileLock_ = std::move(lockResult.value());
    } else {
      ARROW_LOG_ERROR("Collection", "Failed to acquire file lock: " +
        lockResult.status().message());
      throw std::runtime_error("Failed to acquire file lock on " +
        persistencePath.string() + ": " + lockResult.status().message());
    }

    initializeWal();
    auto markerStatus = writeDirtyShutdownMarker();
    if (!markerStatus.ok()) {
      ARROW_LOG_WARN("Collection", "Failed to write dirty shutdown marker: " +
        markerStatus.message());
    }
  }

  void initializeWal() {
    if (persistencePath_) {
      namespace fs = std::filesystem;
      fs::path walDir = *persistencePath_ / "wal";

      wal::Result<wal::WAL> walResult = wal::WAL::open(walDir);
      if (!walResult.ok()) {
        ARROW_LOG_ERROR("WAL", "Open failed: " + walResult.status().message());
        return;
      }
      pWal_ = std::make_unique<wal::WAL>(std::move(walResult.value()));

      wal::Result<wal::RecoveryReport> recoverResult = pWal_->recover();
      if (!recoverResult.ok()) {
        ARROW_LOG_ERROR("WAL", "Recovery failed: " + recoverResult.status().message());
      } else {
        const auto& report = recoverResult.value();
        if (report.truncationPerformed) {
          ARROW_LOG_WARN("WAL", "Recovery: truncated " +
            std::to_string(report.discardedBytes) + " corrupt bytes, recovered " +
            std::to_string(report.validEntries) + " entries");
        }
      }
    }
  }

  utils::Status replayWal(uint64_t fromLsn) {
    if (!pWal_)
      return utils::OkStatus();

    wal::Result<wal::WALContents> contentsResult = pWal_->readAll();
    if (!contentsResult.ok()) {
      if (contentsResult.status().code() == utils::StatusCode::kEof ||
          contentsResult.status().code() == utils::StatusCode::kNotFound) {
        return utils::OkStatus();
      }
      return contentsResult.status();
    }

    const std::vector<wal::Entry>& entries = contentsResult.value().entries;
    uint64_t maxLsn = builder_.currentLsn();
    uint64_t maxTxid = builder_.currentTxid();
    uint64_t replayedCount = 0;

    for (const wal::Entry &entry : entries) {
      if (entry.lsn <= fromLsn)
        continue;

      if (entry.lsn >= maxLsn)
        maxLsn = entry.lsn + 1;
      if (entry.txid >= maxTxid)
        maxTxid = entry.txid + 1;

      switch (entry.type) {
      case wal::OperationType::INSERT:
        {
          std::string vectorID = entry.getVectorID();
          auto existingResult = idSpace_.lookup(vectorID);
          if (existingResult.ok()) {
            InternalID existingID = existingResult.value();
            if (!pIndex_->insert(existingID, entry.embedding)) {
              return utils::Status(utils::StatusCode::kInternal,
                                 "Failed to replay INSERT for existing vector " + vectorID);
            }
          } else {
            auto internalIDResult = idSpace_.assign(vectorID);
            if (!internalIDResult.ok()) {
              return utils::Status(utils::StatusCode::kInternal,
                                 "Failed to replay INSERT for vector " + vectorID);
            }
            InternalID internalID = internalIDResult.value();
            if (!pIndex_->insert(internalID, entry.embedding)) {
              return utils::Status(utils::StatusCode::kInternal,
                                 "Failed to replay INSERT for vector " + vectorID);
            }
          }
        }
        ++replayedCount;
        break;
      case wal::OperationType::DELETE:
        {
          std::string vectorID = entry.getVectorID();
          auto internalIDResult = idSpace_.lookup(vectorID);
          if (internalIDResult.ok()) {
            InternalID internalID = internalIDResult.value();
            pIndex_->markDelete(internalID);
            metadata_.erase(internalID);
          }
        }
        ++replayedCount;
        break;
      default:
        break;
      }
    }

    builder_.restoreCounters(maxLsn, maxTxid);
    if (replayedCount > 0)
      recoveredFromWal_ = true;

    return utils::OkStatus();
  }

  static std::vector<std::vector<IndexSearchResult>>
  parallelSearch(const HNSWIndex *index,
                 const std::vector<std::vector<float>> &queries, uint32_t k,
                 uint32_t ef) {

    const size_t numQueries = queries.size();
    std::vector<std::vector<IndexSearchResult>> results(numQueries);

    const size_t hwConcurrency = std::thread::hardware_concurrency();
    const size_t numThreads =
        std::min(hwConcurrency, std::min(size_t(8), numQueries));

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

    for (auto &thread : threads) {
      thread.join();
    }

    return results;
  }

  // Internal insert without lock (caller must hold unique_lock)
  utils::Status insertLocked(const VectorID& id, const std::vector<float>& vec,
                             Metadata metadata) {
    if (vec.size() != config_.dimensions) {
      return utils::Status(utils::StatusCode::kDimensionMismatch,
                           "Vector dimension mismatch: expected " +
                               std::to_string(config_.dimensions) +
                               ", got " + std::to_string(vec.size()));
    }

    if (id.size() > wal::kMaxVectorIDSize) {
      return utils::Status(utils::StatusCode::kInvalidArgument,
                           "Vector ID exceeds maximum length of " +
                               std::to_string(wal::kMaxVectorIDSize) + " bytes");
    }

    auto vecStatus = validateVector(vec);
    if (!vecStatus.ok()) return vecStatus;

    if (!metadata.empty()) {
      auto metaStatus = validateMetadata(metadata);
      if (!metaStatus.ok()) return metaStatus;
    }

    auto internalIDResult = idSpace_.reserve(id);

    if (!internalIDResult.ok()) {
      return internalIDResult.status();
    }

    InternalID internalID = internalIDResult.value();

    if (!pIndex_->insert(internalID, vec)) {
      return utils::Status(utils::StatusCode::kInternal, "Insert failed");
    }

    idSpace_.commit(id, internalID);

    // Log to WAL after successful index insert
    if (pWal_) {
      auto entryResult = builder_.buildInsert(id, config_.dimensions, vec);
      if (entryResult.ok()) {
        wal::Status status = pWal_->log(entryResult.value());
        if (!status.ok()) {
          ARROW_LOG_ERROR("Collection", "WAL log failed for " + id + ": " + status.message());
          return status;
        }
      }
    }

    metadata_[internalID] = std::move(metadata);
    return utils::OkStatus();
  }

  // Internal update without lock (caller must hold unique_lock)
  utils::Status updateLocked(const VectorID& id, const std::vector<float>& vec,
                             Metadata metadata) {
    auto idResult = idSpace_.lookup(id);
    if (!idResult.ok())
      return utils::Status(utils::StatusCode::kNotFound, "Vector not found: " + id);

    if (vec.size() != config_.dimensions)
      return utils::Status(utils::StatusCode::kDimensionMismatch, "Dimension mismatch");

    auto vecStatus = validateVector(vec);
    if (!vecStatus.ok()) return vecStatus;

    if (!metadata.empty()) {
      auto metaStatus = validateMetadata(metadata);
      if (!metaStatus.ok()) return metaStatus;
    }

    InternalID internalID = idResult.value();

    // Custom HNSW handles duplicate labels via updatePoint internally
    if (!pIndex_->insert(internalID, vec))
      return utils::Status(utils::StatusCode::kInternal, "Update failed: HNSW insert error");

    if (!metadata.empty())
      metadata_[internalID] = std::move(metadata);

    // WAL: log as INSERT (idempotent on replay since addPoint handles duplicates)
    if (pWal_) {
      auto entry = builder_.buildInsert(id, config_.dimensions, vec);
      if (entry.ok()) {
        wal::Status walStatus = pWal_->log(entry.value());
        if (!walStatus.ok()) {
          ARROW_LOG_ERROR("Collection", "WAL log failed for update " + id + ": " + walStatus.message());
          return walStatus;
        }
      }
    }

    return utils::OkStatus();
  }
};

// ─────────────────────────────────────────────────────────────
// Collection public methods
// ─────────────────────────────────────────────────────────────

Collection::Collection(const CollectionConfig &config)
    : pImpl_(std::make_unique<Impl>(config)) {}

Collection::Collection(const CollectionConfig &config,
                       const std::filesystem::path &persistencePath)
    : pImpl_(std::make_unique<Impl>(config, persistencePath)) {}

Collection::Collection(std::unique_ptr<Impl> impl) : pImpl_(std::move(impl)) {}

Collection::~Collection() = default;
Collection::Collection(Collection &&) noexcept = default;
Collection &Collection::operator=(Collection &&) noexcept = default;

const std::string &Collection::name() const {
  std::shared_lock lock(pImpl_->mutex_);
  return pImpl_->config_.name;
}
uint32_t Collection::dimension() const {
  std::shared_lock lock(pImpl_->mutex_);
  return pImpl_->config_.dimensions;
}
Space Collection::space() const {
  std::shared_lock lock(pImpl_->mutex_);
  return pImpl_->config_.space;
}
size_t Collection::size() const {
  std::shared_lock lock(pImpl_->mutex_);
  return pImpl_->pIndex_->size();
}
bool Collection::recoveredFromWal() const {
  std::shared_lock lock(pImpl_->mutex_);
  return pImpl_->recoveredFromWal_;
}

utils::Status Collection::insert(const VectorID& id, const std::vector<float>& vec, Metadata metadata) {
  std::unique_lock lock(pImpl_->mutex_);
  return pImpl_->insertLocked(id, vec, std::move(metadata));
}

utils::Status Collection::insert(const std::vector<std::string> &text) {
  Embedder embedder;
  if (!embedder.ok()) {
    return utils::Status(utils::StatusCode::kInternal,
                         "Embedder not initialized");
  }
  for (size_t i = 0; i < text.size(); ++i) {
    std::vector<float> vec = embedder.embed(text[i].c_str());
    if (vec.empty()) {
      return utils::Status(utils::StatusCode::kInternal, "Embedding failed");
    }
    std::string id = "doc-" + std::to_string(i + 1);
    Metadata meta{{"text", text[i]}};
    auto status = insert(id, vec, std::move(meta));
    if (!status.ok())
      return status;
  }
  return utils::OkStatus();
}

utils::Result<BatchInsertResult> Collection::insertBatch(
    const std::vector<std::pair<VectorID, std::vector<float>>>& batch) {
  std::unique_lock lock(pImpl_->mutex_);

  if (batch.size() > kMaxBatchSize) {
    return utils::Status(utils::StatusCode::kInvalidArgument,
      "Batch size exceeds maximum of " + std::to_string(kMaxBatchSize));
  }

  BatchInsertResult result;
  result.results.resize(batch.size());
  result.successCount = 0;
  result.failureCount = 0;

  // Collect successful WAL entries to log as a batch at the end
  std::vector<wal::Entry> successfulWalEntries;
  successfulWalEntries.reserve(batch.size());

  for (size_t i = 0; i < batch.size(); ++i) {
    const auto &[vectorID, vec] = batch[i];

    // Validate dimensions
    if (vec.size() != pImpl_->config_.dimensions) {
      result.results[i].id = 0;
      result.results[i].status = utils::Status(utils::StatusCode::kDimensionMismatch,
                                         "Vector dimension mismatch");
      result.failureCount++;
      continue;
    }

    // Validate vector contents
    auto vecStatus = validateVector(vec);
    if (!vecStatus.ok()) {
      result.results[i].id = 0;
      result.results[i].status = vecStatus;
      result.failureCount++;
      continue;
    }

    auto internalIDResult = pImpl_->idSpace_.reserve(vectorID);
    if (!internalIDResult.ok()) {
      result.results[i].id = 0;
      result.results[i].status = internalIDResult.status();
      result.failureCount++;
      continue;
    }
    InternalID internalID = internalIDResult.value();

    // Try index insert first
    if (pImpl_->pIndex_->insert(internalID, vec)) {
      pImpl_->idSpace_.commit(vectorID, internalID);
      result.results[i].id = internalID;
      result.results[i].status = utils::OkStatus();
      result.successCount++;

      // Build WAL entry for successful inserts only
      if (pImpl_->pWal_) {
        auto entryResult = pImpl_->builder_.buildInsert(vectorID, pImpl_->config_.dimensions, vec);
        if (entryResult.ok()) {
          successfulWalEntries.push_back(std::move(entryResult.value()));
        }
      }
    } else {
      result.results[i].id = internalID;
      result.results[i].status = utils::Status(utils::StatusCode::kInternal,
                                             "HNSW insert failed");
      result.failureCount++;
    }
  }

  // Log all successful inserts to WAL as a batch (single fsync)
  if (pImpl_->pWal_ && !successfulWalEntries.empty()) {
    utils::Status walStatus = pImpl_->pWal_->logBatch(successfulWalEntries);
    if (!walStatus.ok()) {
      ARROW_LOG_ERROR("Collection", "WAL batch log failed: " + walStatus.message());
      return walStatus;
    }
  }

  return result;
}

utils::Status Collection::setMetadata(const VectorID& id, const Metadata& metadata) {
  std::unique_lock lock(pImpl_->mutex_);
  auto metaStatus = validateMetadata(metadata);
  if (!metaStatus.ok()) return metaStatus;

  auto internalIDResult = pImpl_->idSpace_.lookup(id);
  if (!internalIDResult.ok()) {
    return utils::Status(utils::StatusCode::kNotFound, "Vector not found: " + id);
  }
  pImpl_->metadata_[internalIDResult.value()] = metadata;
  return utils::OkStatus();
}

utils::Result<Metadata> Collection::getMetadata(const VectorID& id) {
  std::shared_lock lock(pImpl_->mutex_);
  auto internalIDResult = pImpl_->idSpace_.lookup(id);
  if (!internalIDResult.ok()) {
    return utils::Status(utils::StatusCode::kNotFound, "Vector not found: " + id);
  }
  auto it = pImpl_->metadata_.find(internalIDResult.value());
  if (it != pImpl_->metadata_.end())
    return it->second;
  return Metadata{};
}

std::vector<IndexSearchResult>
Collection::search(const std::vector<float> &query, uint32_t k,
                   uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);
  return pImpl_->pIndex_->search(query, k, ef);
}

std::vector<IndexSearchResult>
Collection::search(const std::vector<float>& query, uint32_t k,
                   MetadataFilter filter, uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);

  // Over-fetch to account for filtering
  const uint32_t fetchK = std::min(k * 4, static_cast<uint32_t>(pImpl_->pIndex_->size()));
  if (fetchK == 0) return {};

  auto candidates = pImpl_->pIndex_->search(query, fetchK, ef);

  std::vector<IndexSearchResult> results;
  results.reserve(k);

  for (const auto& candidate : candidates) {
    if (results.size() >= k) break;

    auto it = pImpl_->metadata_.find(candidate.id);
    Metadata meta = (it != pImpl_->metadata_.end()) ? it->second : Metadata{};

    if (filter(meta)) {
      results.push_back(candidate);
    }
  }
  return results;
}

std::vector<IndexSearchResult>
Collection::query(const std::string &query, uint32_t k, uint32_t ef) const {
  Embedder embedder;
  if (!embedder.ok()) {
    std::cout << "Error: Failed to create embedder\n";
    return {};
  }

  std::vector<float> vec = embedder.embed(query.c_str());
  if (vec.empty()) {
    std::cout << "Error: Failed to embed query\n";
    return {};
  }

  std::shared_lock lock(pImpl_->mutex_);
  return pImpl_->pIndex_->search(vec, k, ef);
}

SearchResult Collection::query(const std::vector<float> &queryVec, uint32_t k,
                               uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);
  auto indexResults = pImpl_->pIndex_->search(queryVec, k, ef);
  SearchResult result;
  result.hits.reserve(indexResults.size());

  for (const auto &ir : indexResults) {
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

utils::Result<std::vector<std::vector<IndexSearchResult>>>
Collection::searchBatch(const std::vector<std::vector<float>> &queries,
                        uint32_t k, uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);
  for (size_t i = 0; i < queries.size(); ++i) {
    if (queries[i].size() != pImpl_->config_.dimensions) {
      return utils::Result<std::vector<std::vector<IndexSearchResult>>>(
          utils::Status(utils::StatusCode::kDimensionMismatch,
                        "Query " + std::to_string(i) + " dimension mismatch"));
    }
  }

  return Impl::parallelSearch(pImpl_->pIndex_.get(), queries, k, ef);
}

utils::Result<std::vector<float>> Collection::get(const VectorID& id) const {
  std::shared_lock lock(pImpl_->mutex_);

  auto idResult = pImpl_->idSpace_.lookup(id);
  if (!idResult.ok())
    return utils::Status(utils::StatusCode::kNotFound, "Vector not found: " + id);

  InternalID internalID = idResult.value();
  const float* data = pImpl_->pIndex_->getVectorData(internalID);
  if (!data)
    return utils::Status(utils::StatusCode::kNotFound, "Vector data not found");

  return std::vector<float>(data, data + pImpl_->config_.dimensions);
}

utils::Status Collection::update(const VectorID& id,
                                  const std::vector<float>& vec,
                                  Metadata metadata) {
  std::unique_lock lock(pImpl_->mutex_);
  return pImpl_->updateLocked(id, vec, std::move(metadata));
}

utils::Status Collection::upsert(const VectorID& id,
                                  const std::vector<float>& vec,
                                  Metadata metadata) {
  std::unique_lock lock(pImpl_->mutex_);

  auto idResult = pImpl_->idSpace_.lookup(id);
  if (idResult.ok()) {
    return pImpl_->updateLocked(id, vec, std::move(metadata));
  } else {
    return pImpl_->insertLocked(id, vec, std::move(metadata));
  }
}

utils::Status Collection::remove(const VectorID& id) {
  std::unique_lock lock(pImpl_->mutex_);
  if (id.size() > wal::kMaxVectorIDSize) {
    return utils::Status(utils::StatusCode::kInvalidArgument,
                         "Vector ID exceeds maximum length of " +
                             std::to_string(wal::kMaxVectorIDSize) + " bytes");
  }

  auto internalIDResult = pImpl_->idSpace_.lookup(id);
  if (!internalIDResult.ok()) {
    return utils::Status(utils::StatusCode::kNotFound, "Vector ID not found");
  }
  InternalID internalID = internalIDResult.value();

  wal::Status delStatus = pImpl_->pIndex_->markDelete(internalID);
  if (!delStatus.ok())
    return delStatus;

  // Log to WAL after successful delete
  if (pImpl_->pWal_) {
    auto entryResult = pImpl_->builder_.buildDelete(id);
    if (entryResult.ok()) {
      wal::Status status = pImpl_->pWal_->log(entryResult.value());
      if (!status.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL log failed for delete " + id + ": " + status.message());
        return status;
      }
    }
  }

  pImpl_->metadata_.erase(internalID);
  return utils::OkStatus();
}

utils::Status Collection::save(const std::string &directoryPath) {
  std::unique_lock lock(pImpl_->mutex_);
  namespace fs = std::filesystem;

  RecoveryMetadata recovery{
    .lastPersistedLsn = (pImpl_->builder_.currentLsn() > 0) ? pImpl_->builder_.currentLsn() - 1 : 0,
    .lastPersistedTxid = (pImpl_->builder_.currentTxid() > 0) ? pImpl_->builder_.currentTxid() - 1 : 0,
    .cleanShutdown = true
  };

  utils::Status status = CollectionPersistence::save(
    fs::path(directoryPath),
    pImpl_->config_,
    pImpl_->hnswConfig_,
    *pImpl_->pIndex_,
    pImpl_->idSpace_,
    pImpl_->metadata_,
    recovery
  );
  if (!status.ok()) return status;

  if (pImpl_->pWal_) {
    wal::Status walStatus = pImpl_->pWal_->truncate();
    if (!walStatus.ok()) return walStatus;
  }

  pImpl_->lastPersistedLsn_ = recovery.lastPersistedLsn;
  return utils::OkStatus();
}

utils::Result<Collection> Collection::load(const std::string &directoryPath) {
  namespace fs = std::filesystem;

  auto loadResult = CollectionPersistence::load(fs::path(directoryPath));
  if (!loadResult.ok()) return loadResult.status();

  auto& [config, hnswConfig, recovery, index, idSpace, metadata] = loadResult.value();

  CollectionConfig collectionConfig{
    .name = config.name,
    .dimensions = config.dimensions,
    .space = config.space,
    .index = {
      .max_elements = hnswConfig.maxElements,
      .M = hnswConfig.M,
      .ef_construction = hnswConfig.efConstruction
    }
  };

  auto impl = std::make_unique<Impl>(collectionConfig, fs::path(directoryPath));
  impl->pIndex_ = std::move(index);
  impl->idSpace_ = std::move(idSpace);
  impl->metadata_ = std::move(metadata);
  impl->lastPersistedLsn_ = recovery.lastPersistedLsn;
  impl->builder_.restoreCounters(
    recovery.lastPersistedLsn + 1,
    recovery.lastPersistedTxid + 1
  );

  fs::path walPath = fs::path(directoryPath) / "wal" / "db.wal";
  if (fs::exists(walPath)) {
    if (!recovery.cleanShutdown) {
      ARROW_LOG_WARN("Collection",
        "Collection was not shut down cleanly. Performing full WAL replay...");
      utils::Status replayStatus = impl->replayWal(0);
      if (!replayStatus.ok()) return replayStatus;

      utils::Status saveStatus = Collection(std::move(impl)).save(directoryPath);
      if (!saveStatus.ok()) return saveStatus;

      impl = std::make_unique<Impl>(collectionConfig, fs::path(directoryPath));

      auto reloadResult = CollectionPersistence::load(fs::path(directoryPath));
      if (!reloadResult.ok()) return reloadResult.status();
      impl->pIndex_ = std::move(reloadResult.value().index);
      impl->idSpace_ = std::move(reloadResult.value().idSpace);
      impl->metadata_ = std::move(reloadResult.value().metadata);
      impl->lastPersistedLsn_ = recovery.lastPersistedLsn;
      impl->builder_.restoreCounters(
        recovery.lastPersistedLsn + 1,
        recovery.lastPersistedTxid + 1
      );
      impl->recoveredFromWal_ = true;
    } else {
      utils::Status replayStatus = impl->replayWal(recovery.lastPersistedLsn);
      if (!replayStatus.ok()) return replayStatus;
    }
  }

  return Collection(std::move(impl));
}

utils::Status Collection::close() {
  if (pImpl_->persistencePath_) {
    return save(pImpl_->persistencePath_->string());
  }
  return utils::OkStatus();
}

Collection::Stats Collection::stats() const {
  std::shared_lock lock(pImpl_->mutex_);
  return Stats{
    .vectorCount = pImpl_->pIndex_->size(),
    .metadataCount = pImpl_->metadata_.size(),
    .maxCapacity = pImpl_->pIndex_->capacity(),
    .dimensions = pImpl_->config_.dimensions,
  };
}

} // namespace arrow
