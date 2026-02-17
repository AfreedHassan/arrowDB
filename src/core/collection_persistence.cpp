#include "core/collection_persistence.h"
#include "utils/filesync.h"
#include "utils/json_utils.h"
#include <fstream>

namespace arrow {

static constexpr int kSchemaVersion = 3;

namespace {

utils::json fieldTypeToJson(FieldType ft) {
  switch (ft) {
    case FieldType::Int64:  return "Int64";
    case FieldType::Double: return "Double";
    case FieldType::String: return "String";
    case FieldType::Bool:   return "Bool";
  }
  return "String"; // unreachable
}

FieldType jsonToFieldType(const utils::json& j) {
  std::string s = j.get<std::string>();
  if (s == "Int64")  return FieldType::Int64;
  if (s == "Double") return FieldType::Double;
  if (s == "String") return FieldType::String;
  if (s == "Bool")   return FieldType::Bool;
  return FieldType::String; // fallback
}

utils::json schemaToJson(const Schema& schema) {
  utils::json arr = utils::json::array();
  for (const auto& f : schema.fields) {
    utils::json fj = utils::json::object();
    fj["name"] = f.name;
    fj["type"] = fieldTypeToJson(f.type);
    if (f.required) fj["required"] = true;
    arr.push_back(fj);
  }
  return arr;
}

Schema jsonToSchema(const utils::json& j) {
  Schema schema;
  if (!j.is_array()) return schema;
  for (const auto& fj : j) {
    FieldDef fd;
    fd.name = fj["name"].get<std::string>();
    fd.type = jsonToFieldType(fj["type"]);
    fd.required = fj.value("required", false);
    schema.fields.push_back(std::move(fd));
  }
  return schema;
}

utils::json internalConfigToJson(const InternalConfig& config) {
  utils::json j = utils::json::object();
  j["name"] = config.name;
  j["dimensions"] = config.dimensions;
  j["space"] = utils::spaceToJson(config.space);
  j["dtype"] = utils::dataTypeToJson(config.dtype);
  j["idxType"] = "HNSW";
  j["schemaVersion"] = kSchemaVersion;
  if (!config.schema.empty()) {
    j["schema"] = schemaToJson(config.schema);
  }
  return j;
}

InternalConfig jsonToInternalConfig(const utils::json& j) {
  InternalConfig config;
  config.name = j["name"].get<std::string>();
  config.dimensions = j["dimensions"].get<uint32_t>();
  config.space = utils::jsonToSpace(j["space"]);
  config.dtype = utils::jsonToDataType(j["dtype"]);
  if (j.contains("schema")) {
    config.schema = jsonToSchema(j["schema"]);
  }
  return config;
}

utils::json hnswConfigToJson(const HNSWConfig& config) {
  utils::json j = utils::json::object();
  j["maxElements"] = config.maxElements;
  j["M"] = config.M;
  j["efConstruction"] = config.efConstruction;
  if (config.quantize) j["quantize"] = true;
  return j;
}

HNSWConfig jsonToHNSWConfig(const utils::json& j) {
  HNSWConfig config;
  if (j.contains("maxElements"))
    config.maxElements = j["maxElements"].get<uint32_t>();
  if (j.contains("M"))
    config.M = j["M"].get<uint32_t>();
  if (j.contains("efConstruction"))
    config.efConstruction = j["efConstruction"].get<uint32_t>();
  if (j.contains("quantize"))
    config.quantize = j["quantize"].get<bool>();
  return config;
}

/// Atomic write: write to .tmp, fsync, rename over target.
utils::Status writeConfigToJson(
  const std::filesystem::path& metaPath,
  const InternalConfig& config,
  const HNSWConfig& hnswConfig,
  const RecoveryMetadata& recovery
) {
  utils::json j = internalConfigToJson(config);
  j["hnsw"] = hnswConfigToJson(hnswConfig);

  utils::json recoveryJson = utils::json::object();
  recoveryJson["lastPersistedLsn"] = recovery.lastPersistedLsn;
  recoveryJson["lastPersistedTxid"] = recovery.lastPersistedTxid;
  recoveryJson["cleanShutdown"] = recovery.cleanShutdown;
  j["recovery"] = recoveryJson;

  std::filesystem::path tmpPath = metaPath;
  tmpPath += ".tmp";

  std::ofstream file(tmpPath.string());
  if (!file.is_open()) {
    return utils::Status(utils::StatusCode::kIoError,
      "Failed to open file for writing: " + tmpPath.string());
  }
  file << j.dump(2);
  file.close();

  if (file.fail()) {
    return utils::Status(utils::StatusCode::kIoError,
      "Failed writing to: " + tmpPath.string());
  }

  if (!utils::syncFile(tmpPath.string())) {
    return utils::Status(utils::StatusCode::kIoError,
      "Failed to fsync: " + tmpPath.string());
  }

  std::error_code ec;
  std::filesystem::rename(tmpPath, metaPath, ec);
  if (ec) {
    return utils::Status(utils::StatusCode::kIoError,
      "Failed to rename " + tmpPath.string() + " -> " + metaPath.string());
  }

  return utils::OkStatus();
}

utils::Result<std::tuple<InternalConfig, HNSWConfig, RecoveryMetadata>>
readConfigFromJson(const std::filesystem::path& metaPath) {
  std::ifstream file(metaPath.string());
  if (!file.is_open()) {
    return utils::Status(utils::StatusCode::kIoError,
      "Failed to open file for reading: " + metaPath.string());
  }

  utils::json j;
  try {
    file >> j;
  } catch (const utils::json::parse_error& e) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Failed to parse meta.json: " + std::string(e.what()));
  } catch (const std::exception& e) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Error reading meta.json: " + std::string(e.what()));
  }
  file.close();

  int version = j.value("schemaVersion", 1);
  if (version > kSchemaVersion) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Collection was created with a newer version of ArrowDB (schema v" +
      std::to_string(version) + ", supported up to v" +
      std::to_string(kSchemaVersion) + ")");
  }

  InternalConfig config = jsonToInternalConfig(j);
  HNSWConfig hnswConfig =
    j.contains("hnsw") ? jsonToHNSWConfig(j["hnsw"]) : HNSWConfig{};

  // Validate config values
  if (config.dimensions == 0 || config.dimensions > 65536) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Invalid dimensions in meta.json: " + std::to_string(config.dimensions));
  }
  if (hnswConfig.maxElements == 0) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Invalid maxElements in meta.json: 0");
  }
  if (hnswConfig.M == 0 || hnswConfig.M > 10000) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Invalid M in meta.json: " + std::to_string(hnswConfig.M));
  }
  if (hnswConfig.efConstruction == 0) {
    return utils::Status(utils::StatusCode::kCorruption,
      "Invalid efConstruction in meta.json: 0");
  }

  RecoveryMetadata recovery;
  if (j.contains("recovery")) {
    const auto& r = j["recovery"];
    if (r.contains("lastPersistedLsn"))
      recovery.lastPersistedLsn = r["lastPersistedLsn"].get<uint64_t>();
    if (r.contains("lastPersistedTxid"))
      recovery.lastPersistedTxid = r["lastPersistedTxid"].get<uint64_t>();
    if (r.contains("cleanShutdown"))
      recovery.cleanShutdown = r["cleanShutdown"].get<bool>();
  }

  return std::make_tuple(config, hnswConfig, recovery);
}

} // anonymous namespace

utils::Status CollectionPersistence::save(
  const std::filesystem::path& dir,
  const InternalConfig& config,
  const HNSWConfig& hnswConfig,
  const HNSWIndex& index,
  const IDSpace& idSpace,
  const std::unordered_map<InternalID, Metadata>& metadata,
  const RecoveryMetadata& recovery
) {
  std::filesystem::create_directories(dir);

  // Write data files FIRST (index, id_space, metadata).
  // meta.json with cleanShutdown=true is written LAST as the commit point.
  // This ensures a crash mid-save leaves cleanShutdown=false, triggering
  // WAL replay on next load rather than silently using stale data.

  // Atomic index.bin write: save to .tmp, fsync, rename
  std::filesystem::path indexPath = dir / "index.bin";
  std::filesystem::path tmpIndexPath = dir / "index.bin.tmp";
  utils::Status indexStatus = index.saveIndex(tmpIndexPath.string());
  if (!indexStatus.ok()) return indexStatus;

  if (!utils::syncFile(tmpIndexPath.string())) {
    return utils::Status(utils::StatusCode::kIoError, "Failed to fsync index.bin.tmp");
  }
  std::error_code ec;
  std::filesystem::rename(tmpIndexPath, indexPath, ec);
  if (ec) {
    return utils::Status(utils::StatusCode::kIoError, "Failed to rename index.bin.tmp");
  }

  // Atomic id_space.bin write
  std::filesystem::path idSpacePath = dir / "id_space.bin";
  utils::Status idSpaceStatus = idSpace.save(idSpacePath);
  if (!idSpaceStatus.ok()) return idSpaceStatus;

  // Atomic metadata.json write (always write, even if empty, to clear stale data)
  std::filesystem::path metadataPath = dir / "metadata.json";
  std::filesystem::path tmpMetadataPath = dir / "metadata.json.tmp";
  if (!utils::exportMetadataToJson(metadata, tmpMetadataPath.string())) {
    return utils::Status(utils::StatusCode::kIoError, "Failed to write metadata.json.tmp");
  }

  if (!utils::syncFile(tmpMetadataPath.string())) {
    return utils::Status(utils::StatusCode::kIoError, "Failed to fsync metadata.json.tmp");
  }
  std::filesystem::rename(tmpMetadataPath, metadataPath, ec);
  if (ec) {
    return utils::Status(utils::StatusCode::kIoError, "Failed to rename metadata.json.tmp");
  }

  // Fsync the directory to ensure all renames are durable before writing the commit point
  utils::syncDir(dir.string());

  // Atomic meta.json write LAST — this is the commit point.
  // Only after all data files are durably written do we mark cleanShutdown=true.
  std::filesystem::path metaPath = dir / "meta.json";
  utils::Status status = writeConfigToJson(metaPath, config, hnswConfig, recovery);
  if (!status.ok()) return status;

  // Final directory fsync to ensure meta.json rename is durable
  utils::syncDir(dir.string());

  return utils::OkStatus();
}

utils::Result<CollectionPersistence::LoadResult>
CollectionPersistence::load(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return utils::Status(utils::StatusCode::kNotFound,
      "Collection directory does not exist: " + dir.string());
  }

  std::filesystem::path metaPath = dir / "meta.json";
  if (!std::filesystem::exists(metaPath)) {
    return utils::Status(utils::StatusCode::kNotFound,
      "meta.json not found in collection directory: " + dir.string());
  }

  auto configResult = readConfigFromJson(metaPath);
  if (!configResult.ok()) return configResult.status();

  auto [internalCfg, hnswCfg, recoveryMeta] = configResult.value();

  LoadResult result;
  result.config = internalCfg;
  result.hnswConfig = hnswCfg;
  result.recovery = recoveryMeta;

  std::filesystem::path indexPath = dir / "index.bin";
  if (!std::filesystem::exists(indexPath)) {
    return utils::Status(utils::StatusCode::kNotFound,
      "index.bin not found in collection directory: " + dir.string());
  }
  result.index = std::make_unique<HNSWIndex>(
    internalCfg.dimensions, internalCfg.space, hnswCfg);
  utils::Status loadStatus = result.index->loadIndex(indexPath.string());
  if (!loadStatus.ok()) return loadStatus;

  std::filesystem::path idSpacePath = dir / "id_space.bin";
  if (std::filesystem::exists(idSpacePath)) {
    auto idSpaceResult = IDSpace::load(idSpacePath);
    if (!idSpaceResult.ok()) {
      return utils::Status(utils::StatusCode::kCorruption,
        "Failed to load id_space.bin: " + idSpaceResult.status().message());
    }
    result.idSpace = std::move(idSpaceResult.value());
  }

  std::filesystem::path metadataPath = dir / "metadata.json";
  if (std::filesystem::exists(metadataPath)) {
    result.metadata = utils::importMetadataFromJson(metadataPath.string());
  }

  return result;
}

utils::Status CollectionPersistence::writeDirtyShutdownMarker(
  const std::filesystem::path& dir,
  const InternalConfig& config,
  const HNSWConfig& hnswConfig,
  uint64_t currentLsn,
  uint64_t currentTxid
) {
  std::filesystem::create_directories(dir);
  RecoveryMetadata recovery{
    .lastPersistedLsn = (currentLsn > 0) ? currentLsn - 1 : 0,
    .lastPersistedTxid = (currentTxid > 0) ? currentTxid - 1 : 0,
    .cleanShutdown = false
  };
  std::filesystem::path metaPath = dir / "meta.json";
  return writeConfigToJson(metaPath, config, hnswConfig, recovery);
}

} // namespace arrow
