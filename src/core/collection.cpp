// Copyright 2025 ArrowDB
#include "arrow/collection.h"
#include "arrow/utils/uuid.h"
#include "utils/json_utils.h"
#include "embedder/embedder.h"
#include "index/hnsw_index.h"
#include <nlohmann/json.hpp>
#include "wal/wal.h"
#include "core/id_space.h"
#include "core/collection_persistence.h"
#include "utils/file_lock.h"
#include "utils/log.h"

#include <roaring/roaring.hh>

#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace arrow {

// ─────────────────────────────────────────────────────────────
// Vector normalization for Cosine space
// ─────────────────────────────────────────────────────────────

static void normalizeVector(std::vector<float>& vec) {
  float norm = 0.0f;
  for (float v : vec) norm += v * v;
  if (norm > 0.0f) {
    norm = 1.0f / std::sqrt(norm);
    for (float& v : vec) v *= norm;
  }
}

// ─────────────────────────────────────────────────────────────
// Input validation helpers
// ─────────────────────────────────────────────────────────────

static constexpr size_t kMaxBatchSize = 1000000;
static constexpr size_t kParallelInsertThreshold = 1000;
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

/// Validate metadata against schema. Empty schema = no validation.
static utils::Status validateMetadataSchema(const MetadataSchema& schema, const Metadata& meta) {
  if (schema.empty()) return utils::OkStatus();

  for (const auto& field : schema.fields) {
    auto it = meta.find(field.name);
    if (it == meta.end()) {
      if (field.required) {
        return utils::Status(utils::StatusCode::kInvalidArgument,
            "Required field missing: " + field.name);
      }
      continue;
    }
    // Check type matches
    const auto& val = it->second;
    bool typeOk = false;
    switch (field.type) {
      case FieldType::Int64:  typeOk = std::holds_alternative<int64_t>(val); break;
      case FieldType::Double: typeOk = std::holds_alternative<double>(val);  break;
      case FieldType::String: typeOk = std::holds_alternative<std::string>(val); break;
      case FieldType::Bool:   typeOk = std::holds_alternative<bool>(val);    break;
    }
    if (!typeOk) {
      return utils::Status(utils::StatusCode::kInvalidArgument,
          "Field '" + field.name + "' has wrong type");
    }
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

  using MetadataMap = std::unordered_map<InternalID, Metadata>;

  std::unique_ptr<HNSWIndex> pIndex_;
  std::unique_ptr<wal::WAL> pWal_;

  std::shared_ptr<MetadataMap> metadata_ = std::make_shared<MetadataMap>();

  IDSpace idSpace_;
  wal::EntryBuilder builder_;
  std::optional<std::filesystem::path> persistencePath_;
  uint64_t lastPersistedLsn_ = 0;
  bool recoveredFromWal_ = false;

  mutable std::shared_mutex mutex_;
  std::mutex saveMutex_;  // Serializes save operations (separate from data mutex_)
  std::optional<FileLock> fileLock_;

  static constexpr uint32_t kCompactionOpsThreshold = 5000;
  static constexpr uint32_t kWalSyncBatchSize = 64;
  uint32_t opsSinceLastSave_ = 0;
  uint32_t walPendingSyncs_ = 0;
  std::condition_variable_any cv_;
  std::jthread compactionThread_;

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

  ~Impl() {
    // Must notify the CV so the compaction thread can observe the stop request
    // from jthread's destructor. Without this, the thread blocks on cv_.wait()
    // forever since jthread::~jthread only calls request_stop() + join().
    compactionThread_.request_stop();
    cv_.notify_one();
  }

  explicit Impl(const CollectionConfig &config)
      : config_{config.name, config.dimensions, config.space,
                DataType::Float32, config.schema},
        hnswConfig_{config.index_config.max_elements, config.index_config.hnsw_params.M,
                    config.index_config.hnsw_params.ef_construction,
                    config.index_config.quantization != Quantization::None},
        pIndex_(std::make_unique<HNSWIndex>(config.dimensions, config.space,
                                            hnswConfig_)) {}

  static utils::Result<std::unique_ptr<Impl>> create(
      const CollectionConfig& config,
      const std::filesystem::path& persistencePath,
      bool launchCompaction = true) {
    auto impl = std::make_unique<Impl>(config);
    impl->persistencePath_ = persistencePath;

    auto lockResult = FileLock::acquire(persistencePath);
    if (!lockResult.ok()) {
      ARROW_LOG_ERROR("Collection", "Failed to acquire file lock: " +
          lockResult.status().message());
      return utils::Status(utils::StatusCode::kIoError,
          "Failed to acquire file lock on " + persistencePath.string() +
          ": " + lockResult.status().message());
    }
    impl->fileLock_ = std::move(lockResult.value());

    auto walStatus = impl->openWal();
    if (!walStatus.ok()) return walStatus;

    auto markerStatus = impl->writeDirtyShutdownMarker();
    if (!markerStatus.ok()) {
      ARROW_LOG_WARN("Collection", "Failed to write dirty shutdown marker: " +
          markerStatus.message());
    }

    if (launchCompaction) impl->startCompaction();

    return impl;
  }

  utils::Status openWal() {
    if (!persistencePath_) return utils::OkStatus();
    namespace fs = std::filesystem;
    fs::path walDir = *persistencePath_ / "wal";

    wal::Result<wal::WAL> walResult = wal::WAL::open(walDir);
    if (!walResult.ok()) {
      return utils::Status(utils::StatusCode::kIoError,
          "Failed to open WAL: " + walResult.status().message());
    }
    pWal_ = std::make_unique<wal::WAL>(std::move(walResult.value()));

    wal::Result<wal::RecoveryReport> recoverResult = pWal_->recover();
    if (!recoverResult.ok()) {
      return utils::Status(utils::StatusCode::kCorruption,
          "WAL recovery failed: " + recoverResult.status().message());
    }
    const auto& report = recoverResult.value();
    if (report.truncationPerformed) {
      ARROW_LOG_WARN("WAL", "Recovery: truncated " +
          std::to_string(report.discardedBytes) + " corrupt bytes, recovered " +
          std::to_string(report.validEntries) + " entries");
    }
    return utils::OkStatus();
  }

  void startCompaction() {
    if (persistencePath_ && pWal_) {
      compactionThread_ = std::jthread([this](std::stop_token st) {
          compactionLoop(st);
      });
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

          // Validate embedding dimensions match collection config
          if (entry.embedding.size() != config_.dimensions) {
            return utils::Status(utils::StatusCode::kCorruption,
              "WAL entry dimension mismatch for " + vectorID + ": expected " +
              std::to_string(config_.dimensions) + ", got " +
              std::to_string(entry.embedding.size()));
          }

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
            mutableMetadata().erase(internalID);
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

  static std::vector<std::vector<HNSWSearchResult>>
  parallelSearch(const HNSWIndex *index,
                 const std::vector<std::vector<float>> &queries, uint32_t k,
                 uint32_t ef) {

    const size_t numQueries = queries.size();
    std::vector<std::vector<HNSWSearchResult>> results(numQueries);

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

    auto schemaStatus = validateMetadataSchema(config_.schema, metadata);
    if (!schemaStatus.ok()) return schemaStatus;

    // L2-normalize for Cosine space (IP on unit vectors == cosine similarity)
    const std::vector<float>* vecPtr = &vec;
    std::vector<float> normalizedVec;
    if (config_.space == Space::Cosine) {
      normalizedVec = vec;
      normalizeVector(normalizedVec);
      vecPtr = &normalizedVec;
    }

    auto internalIDResult = idSpace_.assign(id);
    if (!internalIDResult.ok()) {
      return internalIDResult.status();
    }

    InternalID internalID = internalIDResult.value();

    if (!pIndex_->insert(internalID, *vecPtr)) {
      idSpace_.remove(id);
      return utils::Status(utils::StatusCode::kInternal, "Insert failed");
    }

    // Log to WAL after successful index insert (deferred sync for throughput).
    // WAL failures are logged but not propagated — the insert already
    // succeeded in-memory and is immediately searchable. The only risk
    // is losing this entry on crash (it won't be replayed from WAL).
    if (pWal_) {
      auto entryResult = builder_.buildInsert(id, config_.dimensions, *vecPtr);
      if (!entryResult.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL entry build failed for " + id +
            ": " + entryResult.status().message());
      } else {
        wal::Status walStatus = pWal_->logDeferred(entryResult.value());
        if (!walStatus.ok()) {
          ARROW_LOG_ERROR("Collection", "WAL log failed for " + id +
              ": " + walStatus.message());
        } else {
          walPendingSyncs_++;
          maybeFlushWal();
        }
      }
    }

    opsSinceLastSave_++;
    mutableMetadata()[internalID] = std::move(metadata);
    requestCheckpoint();
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

    auto schemaStatus = validateMetadataSchema(config_.schema, metadata);
    if (!schemaStatus.ok()) return schemaStatus;

    // L2-normalize for Cosine space
    const std::vector<float>* vecPtr = &vec;
    std::vector<float> normalizedVec;
    if (config_.space == Space::Cosine) {
      normalizedVec = vec;
      normalizeVector(normalizedVec);
      vecPtr = &normalizedVec;
    }

    InternalID internalID = idResult.value();

    // Custom HNSW handles duplicate labels via updatePoint internally
    if (!pIndex_->insert(internalID, *vecPtr))
      return utils::Status(utils::StatusCode::kInternal, "Update failed: HNSW insert error");

    // Always update metadata (even if empty — caller may intend to clear it)
    mutableMetadata()[internalID] = std::move(metadata);

    // WAL: log as INSERT (idempotent on replay since addPoint handles duplicates)
    if (pWal_) {
      auto entry = builder_.buildInsert(id, config_.dimensions, *vecPtr);
      if (!entry.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL entry build failed for update " + id + ": " + entry.status().message());
        return entry.status();
      }
      wal::Status walStatus = pWal_->logDeferred(entry.value());
      if (!walStatus.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL log failed for update " + id + ": " + walStatus.message());
        return walStatus;
      }
      walPendingSyncs_++;
      maybeFlushWal();
    }

    opsSinceLastSave_++;
    requestCheckpoint();
    return utils::OkStatus();
  }

  // Internal remove without lock (caller must hold unique_lock)
  utils::Status removeLocked(const VectorID& id) {
    if (id.size() > wal::kMaxVectorIDSize) {
      return utils::Status(utils::StatusCode::kInvalidArgument,
          "Vector ID exceeds maximum length of " +
          std::to_string(wal::kMaxVectorIDSize) + " bytes");
    }

    auto internalIDResult = idSpace_.lookup(id);
    if (!internalIDResult.ok()) {
      return utils::Status(utils::StatusCode::kNotFound, "Vector ID not found");
    }
    InternalID internalID = internalIDResult.value();

    wal::Status delStatus = pIndex_->markDelete(internalID);
    if (!delStatus.ok())
      return delStatus;

    // Log to WAL after successful delete
    if (pWal_) {
      auto entryResult = builder_.buildDelete(id);
      if (!entryResult.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL entry build failed for delete " + id + ": " + entryResult.status().message());
        return entryResult.status();
      }
      wal::Status status = pWal_->logDeferred(entryResult.value());
      if (!status.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL log failed for delete " + id + ": " + status.message());
        return status;
      }
      walPendingSyncs_++;
      maybeFlushWal();
    }

    opsSinceLastSave_++;
    idSpace_.remove(id);
    mutableMetadata().erase(internalID);
    requestCheckpoint();
    return utils::OkStatus();
  }

  void compactionLoop(std::stop_token st) {
    std::unique_lock lock(mutex_);
    while (!st.stop_requested()) {
      cv_.wait(lock, [&] {
          return opsSinceLastSave_ >= kCompactionOpsThreshold || st.stop_requested();
          });
      if (st.stop_requested()) break;

      // Flush deferred WAL entries before snapshotting state
      auto flushStatus = flushWal();
      if (!flushStatus.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL flush failed during compaction: " + flushStatus.message());
      }

      // Snapshot mutable state under lock, then release.
      // Metadata: O(1) shared_ptr copy; COW on next mutation if needed.
      // IDSpace: full copy (smaller than metadata typically).
      auto metadataSnap = metadata_;
      IDSpace idSpaceCopy = idSpace_;
      uint64_t lsn = builder_.currentLsn();
      uint64_t txid = builder_.currentTxid();
      opsSinceLastSave_ = 0;
      lock.unlock();

      // Save without holding collection lock.
      // saveMutex_ prevents concurrent saves (from Collection::save()).
      // config_/hnswConfig_ are immutable; pIndex_ has internal thread safety.
      std::lock_guard saveLock(saveMutex_);

      RecoveryMetadata recovery{
        .lastPersistedLsn = (lsn > 0) ? lsn - 1 : 0,
        .lastPersistedTxid = (txid > 0) ? txid - 1 : 0,
        .cleanShutdown = true
      };
      auto status = CollectionPersistence::save(
          *persistencePath_, config_, hnswConfig_,
          *pIndex_, idSpaceCopy, *metadataSnap, recovery);

      lock.lock();

      if (!status.ok()) {
        ARROW_LOG_ERROR("Collection", "Background compaction save failed: " +
            status.message());
        opsSinceLastSave_ += kCompactionOpsThreshold;  // ensure retry
      } else {
        lastPersistedLsn_ = recovery.lastPersistedLsn;
        if (pWal_) {
          wal::Status walStatus = pWal_->truncate();
          if (!walStatus.ok()) {
            ARROW_LOG_ERROR("Collection", "WAL truncate failed: " +
                walStatus.message());
          }
        }
      }
    }
  }

  void requestCheckpoint() {
    if (opsSinceLastSave_ >= kCompactionOpsThreshold) {
      cv_.notify_one();
    }
  }

  // COW accessor: returns a mutable reference to the metadata map.
  // If another thread holds a snapshot (use_count > 1), copies first.
  // Caller must hold unique_lock on mutex_.
  MetadataMap& mutableMetadata() {
    if (metadata_.use_count() > 1) {
      metadata_ = std::make_shared<MetadataMap>(*metadata_);
    }
    return *metadata_;
  }

  // Sync deferred WAL entries when batch threshold is reached.
  // Caller must hold unique_lock on mutex_.
  void maybeFlushWal() {
    if (walPendingSyncs_ >= kWalSyncBatchSize && pWal_) {
      wal::Status status = pWal_->sync();
      if (!status.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL sync failed: " + status.message());
      }
      walPendingSyncs_ = 0;
    }
  }

  // Force-sync all deferred WAL entries to disk.
  // Caller must hold unique_lock on mutex_.
  utils::Status flushWal() {
    if (walPendingSyncs_ > 0 && pWal_) {
      wal::Status status = pWal_->sync();
      walPendingSyncs_ = 0;
      if (!status.ok()) return status;
    }
    return utils::OkStatus();
  }

  utils::Status saveLocked(const std::string &directoryPath) {
    namespace fs = std::filesystem;

    // Flush any deferred WAL entries before persisting
    auto flushStatus = flushWal();
    if (!flushStatus.ok()) {
      ARROW_LOG_ERROR("Collection", "WAL flush failed during save: " + flushStatus.message());
    }

    RecoveryMetadata recovery{
      .lastPersistedLsn = (builder_.currentLsn() > 0) ? builder_.currentLsn() - 1 : 0,
        .lastPersistedTxid = (builder_.currentTxid() > 0) ? builder_.currentTxid() - 1 : 0,
        .cleanShutdown = true
    };

    utils::Status status = CollectionPersistence::save(
        fs::path(directoryPath),
        config_,
        hnswConfig_,
        *pIndex_,
        idSpace_,
        *metadata_,
        recovery
        );
    if (!status.ok()) return status;

    if (pWal_) {
      wal::Status walStatus = pWal_->truncate();
      if (!walStatus.ok()) return walStatus;
    }

    opsSinceLastSave_ = 0;
    lastPersistedLsn_ = recovery.lastPersistedLsn;
    return utils::OkStatus();
  }

  // Phased save that minimizes lock contention:
  // Phase 1 (unique_lock): flush WAL, snapshot mutable state
  // Phase 2 (no lock): persist snapshots + index to disk
  //   (HNSW saveIndex is safe without collection lock — it has internal thread safety,
  //    and concurrent inserts are blocked by unique_lock holders only)
  // Phase 3 (unique_lock): truncate WAL, update counters
  utils::Status savePhased() {
    if (!persistencePath_) {
      return utils::Status(utils::StatusCode::kInvalidArgument, "No persistence path set");
    }

    // saveMutex_ serializes concurrent save() and compaction saves
    std::lock_guard saveLock(saveMutex_);

    // Phase 1: snapshot under exclusive lock
    std::shared_ptr<MetadataMap> metadataSnap;
    IDSpace idSpaceCopy;
    uint64_t lsn, txid;
    {
      std::unique_lock lock(mutex_);
      auto flushStatus = flushWal();
      if (!flushStatus.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL flush failed during save: " + flushStatus.message());
      }
      metadataSnap = metadata_;
      idSpaceCopy = idSpace_;
      lsn = builder_.currentLsn();
      txid = builder_.currentTxid();
    }

    // Phase 2: persist to disk without holding collection lock.
    // config_/hnswConfig_ are immutable; pIndex_ has internal thread safety.
    RecoveryMetadata recovery{
      .lastPersistedLsn = (lsn > 0) ? lsn - 1 : 0,
      .lastPersistedTxid = (txid > 0) ? txid - 1 : 0,
      .cleanShutdown = true
    };
    utils::Status status = CollectionPersistence::save(
        *persistencePath_, config_, hnswConfig_,
        *pIndex_, idSpaceCopy, *metadataSnap, recovery);
    if (!status.ok()) return status;

    // Phase 3: post-save cleanup under exclusive lock
    {
      std::unique_lock lock(mutex_);
      if (pWal_) {
        wal::Status walStatus = pWal_->truncate();
        if (!walStatus.ok()) return walStatus;
      }
      opsSinceLastSave_ = 0;
      lastPersistedLsn_ = recovery.lastPersistedLsn;
    }

    return utils::OkStatus();
  }

  utils::Status close() {
    compactionThread_.request_stop();
    cv_.notify_one();

    if (compactionThread_.joinable()) {
      compactionThread_.join();
    }
    if (persistencePath_) {
      std::unique_lock lock(mutex_);
      // Flush any deferred WAL entries before final save
      auto flushStatus = flushWal();
      if (!flushStatus.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL flush failed during close: " + flushStatus.message());
      }
      if (opsSinceLastSave_ > 0) {
        return saveLocked(persistencePath_->string());
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

utils::Result<Collection> Collection::create(
    const CollectionConfig& config,
    const std::filesystem::path& persistencePath) {
  auto result = Impl::create(config, persistencePath);
  if (!result.ok()) {
    return result.status();
  }
  return Collection(std::move(result.value()));
}

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

utils::Result<VectorID> Collection::insert(const std::vector<float>& vec, Metadata metadata) {
  VectorID id = arrow::uuid::uuidv4();
  auto status = insert(id, vec, std::move(metadata));
  if (!status.ok()) return status;
  return id;
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

utils::Result<BatchInsertResult> Collection::insertBatch(const std::vector<std::string>& texts) {
  Embedder embedder;
  if (!embedder.ok()) {
    return utils::Status(utils::StatusCode::kInternal, "Embedder not initialized");
  }

  std::vector<Document> docs;
  docs.reserve(texts.size());
  for (size_t i = 0; i < texts.size(); ++i) {
    std::vector<float> vec = embedder.embed(texts[i].c_str());
    if (vec.empty()) {
      return utils::Status(utils::StatusCode::kInternal,
          "Embedding failed for text at index " + std::to_string(i));
    }
    docs.push_back({
      .id = "doc-" + std::to_string(i + 1),
      .embedding = std::move(vec),
      .metadata = {{"text", texts[i]}},
    });
  }

  return insertBatch(std::move(docs));
}

utils::Result<VectorID> Collection::insert(Document doc) {
  if (doc.id.empty()) doc.id = arrow::uuid::uuidv4();
  auto status = insert(doc.id, doc.embedding, std::move(doc.metadata));
  if (!status.ok()) return status;
  return doc.id;
}

utils::Result<BatchInsertResult> Collection::insertBatch(std::vector<Document> docs) {
  std::unique_lock lock(pImpl_->mutex_);

  if (docs.size() > kMaxBatchSize) {
    return utils::Status(utils::StatusCode::kInvalidArgument,
      "Batch size exceeds maximum of " + std::to_string(kMaxBatchSize));
  }

  BatchInsertResult result;
  result.results.resize(docs.size());
  result.successCount = 0;
  result.failureCount = 0;

  // Phase 1: Validate all entries
  for (size_t i = 0; i < docs.size(); ++i) {
    auto& doc = docs[i];
    if (doc.id.empty()) doc.id = arrow::uuid::uuidv4();

    result.results[i].id = doc.id;

    if (doc.embedding.size() != pImpl_->config_.dimensions) {
      result.results[i].status = utils::Status(utils::StatusCode::kDimensionMismatch,
                                         "Vector dimension mismatch");
      result.failureCount++;
      continue;
    }

    auto vecStatus = validateVector(doc.embedding);
    if (!vecStatus.ok()) {
      result.results[i].status = vecStatus;
      result.failureCount++;
      continue;
    }

    if (!doc.metadata.empty()) {
      auto metaStatus = validateMetadata(doc.metadata);
      if (!metaStatus.ok()) {
        result.results[i].status = metaStatus;
        result.failureCount++;
        continue;
      }
    }

    auto schemaStatus = validateMetadataSchema(pImpl_->config_.schema, doc.metadata);
    if (!schemaStatus.ok()) {
      result.results[i].status = schemaStatus;
      result.failureCount++;
      continue;
    }

    result.results[i].status = utils::OkStatus();
  }

  // Normalize validated vectors for Cosine space
  if (pImpl_->config_.space == Space::Cosine) {
    for (size_t i = 0; i < docs.size(); ++i) {
      if (result.results[i].status.ok()) {
        normalizeVector(docs[i].embedding);
      }
    }
  }

  // Phase 2: Assign IDSpace for all validated entries (sequential, not thread-safe)
  struct PendingInsert {
    size_t batchIdx;
    InternalID internalID;
  };
  std::vector<PendingInsert> pendingInserts;
  pendingInserts.reserve(docs.size());

  for (size_t i = 0; i < docs.size(); ++i) {
    if (!result.results[i].status.ok()) continue;

    auto internalIDResult = pImpl_->idSpace_.assign(docs[i].id);
    if (!internalIDResult.ok()) {
      result.results[i].status = internalIDResult.status();
      result.failureCount++;
      continue;
    }
    pendingInserts.push_back({i, internalIDResult.value()});
  }

  // Phase 3: Pre-size HNSW to avoid auto-grow races during parallel insert
  size_t needed = pImpl_->pIndex_->size() + pendingInserts.size();
  if (needed > pImpl_->pIndex_->capacity()) {
    size_t newCap = pImpl_->pIndex_->capacity();
    while (newCap < needed) newCap *= 2;
    pImpl_->pIndex_->reserve(newCap);
  }

  // Phase 4: Parallel HNSW insert
  // Each slot in hnswSucceeded tracks whether that pending insert succeeded.
  std::vector<bool> hnswSucceeded(pendingInserts.size(), false);
  const size_t numPending = pendingInserts.size();

  if (numPending >= kParallelInsertThreshold) {
    unsigned nThreads = std::min(
        static_cast<unsigned>(std::thread::hardware_concurrency()),
        static_cast<unsigned>(numPending / 500));
    nThreads = std::max(nThreads, 2u);

    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    size_t perThread = numPending / nThreads;

    for (unsigned t = 0; t < nThreads; ++t) {
      size_t begin = t * perThread;
      size_t end = (t == nThreads - 1) ? numPending : begin + perThread;
      threads.emplace_back([&, begin, end]() {
        for (size_t j = begin; j < end; ++j) {
          auto& p = pendingInserts[j];
          if (pImpl_->pIndex_->insert(p.internalID, docs[p.batchIdx].embedding)) {
            hnswSucceeded[j] = true;
          }
        }
      });
    }
    for (auto& t : threads) t.join();
  } else {
    // Serial path for small batches
    for (size_t j = 0; j < numPending; ++j) {
      auto& p = pendingInserts[j];
      if (pImpl_->pIndex_->insert(p.internalID, docs[p.batchIdx].embedding)) {
        hnswSucceeded[j] = true;
      }
    }
  }

  // Phase 5: Collect results, tombstone IDSpace for failures, build WAL entries
  std::vector<wal::Entry> successfulWalEntries;
  successfulWalEntries.reserve(numPending);

  for (size_t j = 0; j < numPending; ++j) {
    auto& p = pendingInserts[j];
    if (!hnswSucceeded[j]) {
      pImpl_->idSpace_.remove(docs[p.batchIdx].id);
      result.results[p.batchIdx].status = utils::Status(utils::StatusCode::kInternal,
                                                         "HNSW insert failed");
      result.failureCount++;
      continue;
    }

    result.successCount++;

    if (pImpl_->pWal_) {
      auto entryResult = pImpl_->builder_.buildInsert(
          docs[p.batchIdx].id, pImpl_->config_.dimensions, docs[p.batchIdx].embedding);
      if (!entryResult.ok()) {
        ARROW_LOG_ERROR("Collection", "WAL entry build failed for batch insert "
            + docs[p.batchIdx].id + ": " + entryResult.status().message());
        // HNSW insert succeeded but WAL failed — vector is searchable but not durable
      } else {
        successfulWalEntries.push_back(std::move(entryResult.value()));
      }
    }
  }

  // Phase 6: WAL batch fsync
  if (pImpl_->pWal_ && !successfulWalEntries.empty()) {
    utils::Status walStatus = pImpl_->pWal_->logBatch(successfulWalEntries);
    if (!walStatus.ok()) {
      ARROW_LOG_ERROR("Collection", "WAL batch log failed: " + walStatus.message()
          + " (" + std::to_string(result.successCount) + " vectors inserted but not durable)");
    }
  }

  // Phase 7: Assign metadata
  for (size_t j = 0; j < numPending; ++j) {
    if (!hnswSucceeded[j]) continue;
    auto& p = pendingInserts[j];
    const auto& docMeta = docs[p.batchIdx].metadata;
    if (!docMeta.empty()) {
      pImpl_->mutableMetadata()[p.internalID] = docMeta;
    } else {
      pImpl_->mutableMetadata()[p.internalID];
    }
  }
  pImpl_->opsSinceLastSave_ += result.successCount;
  pImpl_->requestCheckpoint();
  return result;
}

utils::Result<BatchInsertResult> Collection::insertBatch(
    const std::vector<std::pair<VectorID, std::vector<float>>>& batch) {
  std::vector<Document> docs;
  docs.reserve(batch.size());
  for (const auto& [id, vec] : batch) {
    docs.push_back({id, vec, {}});
  }
  return insertBatch(std::move(docs));
}

utils::Status Collection::setMetadata(const VectorID& id, const Metadata& metadata) {
  std::unique_lock lock(pImpl_->mutex_);
  auto metaStatus = validateMetadata(metadata);
  if (!metaStatus.ok()) return metaStatus;

  auto schemaStatus = validateMetadataSchema(pImpl_->config_.schema, metadata);
  if (!schemaStatus.ok()) return schemaStatus;

  auto internalIDResult = pImpl_->idSpace_.lookup(id);
  if (!internalIDResult.ok()) {
    return utils::Status(utils::StatusCode::kNotFound, "Vector not found: " + id);
  }
  pImpl_->mutableMetadata()[internalIDResult.value()] = metadata;

  // Metadata changes must trigger a checkpoint to be durable.
  // (WAL entries don't carry metadata, so only checkpoints persist it.)
  pImpl_->opsSinceLastSave_++;
  pImpl_->requestCheckpoint();
  return utils::OkStatus();
}

utils::Result<Metadata> Collection::getMetadata(const VectorID& id) {
  std::shared_lock lock(pImpl_->mutex_);
  auto internalIDResult = pImpl_->idSpace_.lookup(id);
  if (!internalIDResult.ok()) {
    return utils::Status(utils::StatusCode::kNotFound, "Vector not found: " + id);
  }
  auto it = pImpl_->metadata_->find(internalIDResult.value());
  if (it != pImpl_->metadata_->end())
    return it->second;
  return Metadata{};
}

std::vector<IndexSearchResult>
Collection::search(const std::vector<float> &query, uint32_t k,
                   uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);
  if (query.size() != pImpl_->config_.dimensions) {
    ARROW_LOG_ERROR("Collection", "Query dimension mismatch: expected " +
        std::to_string(pImpl_->config_.dimensions) + ", got " + std::to_string(query.size()));
    return {};
  }
  const std::vector<float>* qPtr = &query;
  std::vector<float> normalizedQuery;
  if (pImpl_->config_.space == Space::Cosine) {
    normalizedQuery = query;
    normalizeVector(normalizedQuery);
    qPtr = &normalizedQuery;
  }
  auto internal = pImpl_->pIndex_->search(*qPtr, k, ef);
  std::vector<IndexSearchResult> results;
  results.reserve(internal.size());
  for (const auto &r : internal) {
    auto vid = pImpl_->idSpace_.resolve(r.id);
    if (vid.ok())
      results.push_back({std::string(vid.value()), r.score});
  }
  return results;
}

std::vector<IndexSearchResult>
Collection::search(const std::vector<float>& query, uint32_t k,
                   const MetadataFilter& filter, uint32_t ef) const {
  // O(1) snapshot via shared_ptr copy; COW ensures writers don't invalidate it.
  std::shared_ptr<const Impl::MetadataMap> metadataSnap;
  {
    std::shared_lock lock(pImpl_->mutex_);
    if (query.size() != pImpl_->config_.dimensions) {
      ARROW_LOG_ERROR("Collection", "Query dimension mismatch: expected " +
          std::to_string(pImpl_->config_.dimensions) + ", got " + std::to_string(query.size()));
      return {};
    }
    if (pImpl_->pIndex_->size() == 0) return {};
    metadataSnap = pImpl_->metadata_;
  }
  // Lock released. HNSW search is thread-safe via internal locks.
  const std::vector<float>* qPtr = &query;
  std::vector<float> normalizedQuery;
  if (pImpl_->config_.space == Space::Cosine) {
    normalizedQuery = query;
    normalizeVector(normalizedQuery);
    qPtr = &normalizedQuery;
  }
  HNSWIndex::IDFilter idFilter = [&metadataSnap, &filter](InternalID id) -> bool {
    auto it = metadataSnap->find(id);
    if (it != metadataSnap->end()) return filter(it->second);
    static const Metadata empty;
    return filter(empty);
  };
  auto internal = pImpl_->pIndex_->search(*qPtr, k, idFilter, ef);
  std::vector<IndexSearchResult> results;
  results.reserve(internal.size());
  for (const auto &r : internal) {
    auto vid = pImpl_->idSpace_.resolve(r.id);
    if (vid.ok())
      results.push_back({std::string(vid.value()), r.score});
  }
  return results;
}

SearchResult
Collection::query(const std::string &queryText, uint32_t k, uint32_t ef) const {
  Embedder embedder;
  if (!embedder.ok()) return {};

  std::vector<float> vec = embedder.embed(queryText.c_str());
  if (vec.empty()) return {};

  return query(vec, k, ef);
}

SearchResult Collection::query(const std::vector<float> &queryVec, uint32_t k,
                               uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);
  if (queryVec.size() != pImpl_->config_.dimensions) {
    ARROW_LOG_ERROR("Collection", "Query dimension mismatch: expected " +
        std::to_string(pImpl_->config_.dimensions) + ", got " + std::to_string(queryVec.size()));
    return {};
  }
  const std::vector<float>* qPtr = &queryVec;
  std::vector<float> normalizedQuery;
  if (pImpl_->config_.space == Space::Cosine) {
    normalizedQuery = queryVec;
    normalizeVector(normalizedQuery);
    qPtr = &normalizedQuery;
  }
  auto indexResults = pImpl_->pIndex_->search(*qPtr, k, ef);
  SearchResult result;
  result.hits.reserve(indexResults.size());

  for (const auto &ir : indexResults) {
    auto vid = pImpl_->idSpace_.resolve(ir.id);
    if (!vid.ok()) continue;

    ScoredDocument doc;
    doc.id = std::string(vid.value());
    doc.score = ir.score;

    auto metaIt = pImpl_->metadata_->find(ir.id);
    if (metaIt != pImpl_->metadata_->end()) {
      doc.metadata = metaIt->second;
    }
    result.hits.push_back(std::move(doc));
  }

  return result;
}

SearchResult Collection::query(const std::vector<float>& queryVec, uint32_t k,
                               const MetadataFilter& filter, uint32_t ef) const {
  // Snapshot metadata, then search with filter applied during graph traversal.
  std::shared_ptr<const Impl::MetadataMap> metadataSnap;
  {
    std::shared_lock lock(pImpl_->mutex_);
    if (queryVec.size() != pImpl_->config_.dimensions) {
      ARROW_LOG_ERROR("Collection", "Query dimension mismatch: expected " +
          std::to_string(pImpl_->config_.dimensions) + ", got " + std::to_string(queryVec.size()));
      return {};
    }
    if (pImpl_->pIndex_->size() == 0) return {};
    metadataSnap = pImpl_->metadata_;
  }

  const std::vector<float>* qPtr = &queryVec;
  std::vector<float> normalizedQuery;
  if (pImpl_->config_.space == Space::Cosine) {
    normalizedQuery = queryVec;
    normalizeVector(normalizedQuery);
    qPtr = &normalizedQuery;
  }

  HNSWIndex::IDFilter idFilter = [&metadataSnap, &filter](InternalID id) -> bool {
    auto it = metadataSnap->find(id);
    if (it != metadataSnap->end()) return filter(it->second);
    static const Metadata empty;
    return filter(empty);
  };

  auto indexResults = pImpl_->pIndex_->search(*qPtr, k, idFilter, ef);

  SearchResult result;
  result.hits.reserve(indexResults.size());
  for (const auto& ir : indexResults) {
    auto vid = pImpl_->idSpace_.resolve(ir.id);
    if (!vid.ok()) continue;

    ScoredDocument doc;
    doc.id = std::string(vid.value());
    doc.score = ir.score;

    auto metaIt = metadataSnap->find(ir.id);
    if (metaIt != metadataSnap->end()) {
      doc.metadata = metaIt->second;
    }
    result.hits.push_back(std::move(doc));
  }
  return result;
}

// ─────────────────────────────────────────────────────────────
// PreparedFilter (CRoaring bitmap, build once, search many)
// ─────────────────────────────────────────────────────────────

struct PreparedFilter::Impl {
  roaring::Roaring bitmap;
};

PreparedFilter::PreparedFilter(std::unique_ptr<Impl> impl) : pImpl_(std::move(impl)) {}
PreparedFilter::~PreparedFilter() = default;
PreparedFilter::PreparedFilter(PreparedFilter&&) noexcept = default;
PreparedFilter& PreparedFilter::operator=(PreparedFilter&&) noexcept = default;

PreparedFilter
Collection::prepareFilter(const MetadataFilter& filter) const {
  std::shared_ptr<const Impl::MetadataMap> metadataSnap;
  size_t indexCapacity;
  {
    std::shared_lock lock(pImpl_->mutex_);
    metadataSnap = pImpl_->metadata_;
    indexCapacity = pImpl_->pIndex_->capacity();
  }

  auto pfImpl = std::make_unique<PreparedFilter::Impl>();
  static const Metadata empty;
  bool emptyPasses = filter(empty);

  // InternalIDs are sequential from 0, bounded by HNSW's tableidx_t (uint32_t).
  // Roaring bitmaps use uint32_t, which is safe given this invariant.
  // IDs exceeding uint32 range are skipped defensively.

  if (emptyPasses) {
    pfImpl->bitmap.addRange(0, static_cast<uint64_t>(indexCapacity));
    std::vector<uint32_t> failingIds;
    failingIds.reserve(metadataSnap->size());
    for (const auto& [id, meta] : *metadataSnap) {
      if (id > std::numeric_limits<uint32_t>::max()) continue;  // skip out-of-range IDs
      if (!filter(meta)) {
        failingIds.push_back(static_cast<uint32_t>(id));
      }
    }
    if (!failingIds.empty()) {
      std::sort(failingIds.begin(), failingIds.end());
      pfImpl->bitmap -= roaring::Roaring(failingIds.size(), failingIds.data());
    }
  } else {
    std::vector<uint32_t> passingIds;
    passingIds.reserve(metadataSnap->size());
    for (const auto& [id, meta] : *metadataSnap) {
      if (id > std::numeric_limits<uint32_t>::max()) continue;  // skip out-of-range IDs
      if (filter(meta)) {
        passingIds.push_back(static_cast<uint32_t>(id));
      }
    }
    std::sort(passingIds.begin(), passingIds.end());
    pfImpl->bitmap = roaring::Roaring(passingIds.size(), passingIds.data());
  }

  return PreparedFilter(std::move(pfImpl));
}

std::vector<IndexSearchResult>
Collection::search(const std::vector<float>& query, uint32_t k,
                   const PreparedFilter& filter, uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);
  if (query.size() != pImpl_->config_.dimensions) {
    ARROW_LOG_ERROR("Collection", "Query dimension mismatch: expected " +
        std::to_string(pImpl_->config_.dimensions) + ", got " + std::to_string(query.size()));
    return {};
  }
  const std::vector<float>* qPtr = &query;
  std::vector<float> normalizedQuery;
  if (pImpl_->config_.space == Space::Cosine) {
    normalizedQuery = query;
    normalizeVector(normalizedQuery);
    qPtr = &normalizedQuery;
  }
  auto internal = pImpl_->pIndex_->searchBitmap(*qPtr, k, filter.pImpl_->bitmap, ef);
  std::vector<IndexSearchResult> results;
  results.reserve(internal.size());
  for (const auto &r : internal) {
    auto vid = pImpl_->idSpace_.resolve(r.id);
    if (vid.ok())
      results.push_back({std::string(vid.value()), r.score});
  }
  return results;
}

SearchResult
Collection::query(const std::vector<float>& queryVec, uint32_t k,
                  const PreparedFilter& filter, uint32_t ef) const {
  std::shared_lock lock(pImpl_->mutex_);
  if (queryVec.size() != pImpl_->config_.dimensions) {
    ARROW_LOG_ERROR("Collection", "Query dimension mismatch: expected " +
        std::to_string(pImpl_->config_.dimensions) + ", got " + std::to_string(queryVec.size()));
    return {};
  }
  const std::vector<float>* qPtr2 = &queryVec;
  std::vector<float> normalizedQuery2;
  if (pImpl_->config_.space == Space::Cosine) {
    normalizedQuery2 = queryVec;
    normalizeVector(normalizedQuery2);
    qPtr2 = &normalizedQuery2;
  }
  auto indexResults = pImpl_->pIndex_->searchBitmap(*qPtr2, k, filter.pImpl_->bitmap, ef);

  // Use metadata directly under lock (COW ensures consistency)
  const auto& metadataSnap = pImpl_->metadata_;

  SearchResult result;
  result.hits.reserve(indexResults.size());
  for (const auto& ir : indexResults) {
    auto vid = pImpl_->idSpace_.resolve(ir.id);
    if (!vid.ok()) continue;

    ScoredDocument doc;
    doc.id = std::string(vid.value());
    doc.score = ir.score;

    auto metaIt = metadataSnap->find(ir.id);
    if (metaIt != metadataSnap->end()) {
      doc.metadata = metaIt->second;
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

  const std::vector<std::vector<float>>* qsPtr = &queries;
  std::vector<std::vector<float>> normalizedQueries;
  if (pImpl_->config_.space == Space::Cosine) {
    normalizedQueries = queries;
    for (auto& q : normalizedQueries) normalizeVector(q);
    qsPtr = &normalizedQueries;
  }
  auto internalBatch = Impl::parallelSearch(pImpl_->pIndex_.get(), *qsPtr, k, ef);
  std::vector<std::vector<IndexSearchResult>> mapped(internalBatch.size());
  for (size_t i = 0; i < internalBatch.size(); ++i) {
    mapped[i].reserve(internalBatch[i].size());
    for (const auto &r : internalBatch[i]) {
      auto vid = pImpl_->idSpace_.resolve(r.id);
      if (vid.ok())
        mapped[i].push_back({std::string(vid.value()), r.score});
    }
  }
  return mapped;
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
  return pImpl_->removeLocked(id);
}

utils::Status Collection::optimize() {
  std::unique_lock lock(pImpl_->mutex_);
  if (!pImpl_->hnswConfig_.quantize) {
    return utils::OkStatus();  // No-op when quantization is disabled
  }
  if (pImpl_->pIndex_->isGlobalSQ()) {
    return utils::OkStatus();  // Already optimized
  }
  if (pImpl_->pIndex_->size() == 0) {
    return utils::OkStatus();  // Nothing to optimize
  }
  ARROW_LOG_INFO("Collection", "Optimizing index: computing global SQ + BFS reorder");
  pImpl_->pIndex_->computeGlobalSQ();
  pImpl_->pIndex_->reorderBFS();
  return utils::OkStatus();
}

utils::Status Collection::save(const std::string &directoryPath) {
  namespace fs = std::filesystem;
  // Use phased save if saving to the configured persistence path
  if (pImpl_->persistencePath_ && fs::path(directoryPath) == *pImpl_->persistencePath_) {
    return pImpl_->savePhased();
  }
  // Fallback to locked save for ad-hoc save paths
  std::unique_lock lock(pImpl_->mutex_);
  return pImpl_->saveLocked(directoryPath);
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
    .index_config = {
      .index_type = IndexType::HNSW,
      .max_elements = hnswConfig.maxElements,
      .quantization = hnswConfig.quantize ? Quantization::INT8 : Quantization::None,
      .hnsw_params = {
        .M = hnswConfig.M,
        .ef_construction = hnswConfig.efConstruction,
        .ef_search = 200,  // Not persisted; use default
      }
    },
    .schema = config.schema
  };

  // Consistency check: IDSpace (live entries only) must equal index size minus deleted
  size_t idSpaceSize = idSpace.size();
  size_t indexSize = index->size();
  size_t deletedCount = index->deletedCount();
  size_t liveIndexSize = indexSize - deletedCount;
  if (idSpaceSize != 0 && idSpaceSize != liveIndexSize) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Inconsistent state after load: IDSpace has " + std::to_string(idSpaceSize) +
      " entries but index has " + std::to_string(liveIndexSize) +
      " live vectors (" + std::to_string(indexSize) + " total, " +
      std::to_string(deletedCount) + " deleted)");
  }

  auto implResult = Impl::create(collectionConfig, fs::path(directoryPath), false);
  if (!implResult.ok()) return implResult.status();
  auto impl = std::move(implResult.value());
  impl->pIndex_ = std::move(index);
  impl->idSpace_ = std::move(idSpace);
  impl->metadata_ = std::make_shared<Impl::MetadataMap>(std::move(metadata));
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

      // Save the recovered state directly using CollectionPersistence,
      // keeping the same impl (and its file lock) alive throughout.
      RecoveryMetadata recoveredMeta{
        .lastPersistedLsn = (impl->builder_.currentLsn() > 0) ? impl->builder_.currentLsn() - 1 : 0,
        .lastPersistedTxid = (impl->builder_.currentTxid() > 0) ? impl->builder_.currentTxid() - 1 : 0,
        .cleanShutdown = true
      };
      utils::Status saveStatus = CollectionPersistence::save(
        fs::path(directoryPath),
        impl->config_,
        impl->hnswConfig_,
        *impl->pIndex_,
        impl->idSpace_,
        *impl->metadata_,
        recoveredMeta
      );
      if (!saveStatus.ok()) return saveStatus;

      // Truncate WAL after successful save
      if (impl->pWal_) {
        wal::Status walStatus = impl->pWal_->truncate();
        if (!walStatus.ok()) {
          ARROW_LOG_WARN("Collection", "Failed to truncate WAL after recovery save: " + walStatus.message());
        }
      }

      impl->lastPersistedLsn_ = recoveredMeta.lastPersistedLsn;
      impl->recoveredFromWal_ = true;
    } else {
      utils::Status replayStatus = impl->replayWal(recovery.lastPersistedLsn);
      if (!replayStatus.ok()) return replayStatus;
    }
  }

  // Auto-optimize: if quantization is enabled and not already globally optimized,
  // apply global SQ + BFS reorder for best search performance.
  if (impl->hnswConfig_.quantize && impl->pIndex_->size() > 0
      && !impl->pIndex_->isGlobalSQ()) {
    ARROW_LOG_INFO("Collection", "Auto-optimizing index on load: global SQ + BFS reorder");
    impl->pIndex_->computeGlobalSQ();
    impl->pIndex_->reorderBFS();
  }

  impl->startCompaction();
  return Collection(std::move(impl));
}

utils::Status Collection::close() {
  return pImpl_->close();
}

Collection::Stats Collection::stats() const {
  std::shared_lock lock(pImpl_->mutex_);
  return Stats{
    .vectorCount = pImpl_->pIndex_->size(),
    .metadataCount = pImpl_->metadata_->size(),
    .maxCapacity = pImpl_->pIndex_->capacity(),
    .dimensions = pImpl_->config_.dimensions,
  };
}

void Collection::printStats() const {
  std::shared_lock lock(pImpl_->mutex_);
  nlohmann::json j = {
    {"vectorCount", pImpl_->pIndex_->size()},
    {"metadataCount", pImpl_->metadata_->size()},
    {"maxCapacity", pImpl_->pIndex_->capacity()},
    {"dimensions", pImpl_->config_.dimensions},
  };
  std::cout << "collection: "<< pImpl_->config_.name << " = ";
  std::cout << j.dump(2) << "\n";
}

} // namespace arrow
