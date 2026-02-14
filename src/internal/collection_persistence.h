#ifndef ARROW_COLLECTION_PERSISTENCE_H
#define ARROW_COLLECTION_PERSISTENCE_H

#include <filesystem>
#include <unordered_map>
#include "arrow/types.h"
#include "arrow/utils/result.h"
#include "arrow/utils/status.h"
#include "internal/hnsw_index.h"
#include "internal/id_space.h"

namespace arrow {

struct InternalConfig {
  std::string name;
  uint32_t dimensions;
  Space space;
  DataType dtype = DataType::Float32;
};

struct RecoveryMetadata {
  uint64_t lastPersistedLsn = 0;
  uint64_t lastPersistedTxid = 0;
  bool cleanShutdown = true;
};

class CollectionPersistence {
public:
  struct LoadResult {
    InternalConfig config;
    HNSWConfig hnswConfig;
    RecoveryMetadata recovery;
    std::unique_ptr<HNSWIndex> index;
    IDSpace idSpace;
    std::unordered_map<InternalID, Metadata> metadata;
  };

  static utils::Status save(
    const std::filesystem::path& dir,
    const InternalConfig& config,
    const HNSWConfig& hnswConfig,
    const HNSWIndex& index,
    const IDSpace& idSpace,
    const std::unordered_map<InternalID, Metadata>& metadata,
    const RecoveryMetadata& recovery
  );

  static utils::Result<LoadResult> load(const std::filesystem::path& dir);

  static utils::Status writeDirtyShutdownMarker(
    const std::filesystem::path& dir,
    const InternalConfig& config,
    const HNSWConfig& hnswConfig,
    uint64_t currentLsn,
    uint64_t currentTxid
  );
};

} // namespace arrow

#endif // ARROW_COLLECTION_PERSISTENCE_H
