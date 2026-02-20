#include "arrow/collection.h"
#include "core/collection_persistence.h"
#include "wal/wal.h"
#include "test_util.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace arrow;
using namespace arrow::testing;

class CollectionPersistenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_persistence_test";
    std::filesystem::create_directories(testDir);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
  std::string GetTestPath(const std::string& dirname) {
    return (testDir / dirname).string();
  }

  InternalConfig CreateTestConfig() {
    InternalConfig config;
    config.name = "test_collection";
    config.dimensions = 128;
    config.space = Space::Cosine;
    config.dtype = DataType::Float32;
    return config;
  }

  HNSWConfig CreateTestHNSWConfig() {
    HNSWConfig config;
    config.maxElements = 10000;
    config.M = 64;
    config.efConstruction = 200;
    return config;
  }
};

TEST_F(CollectionPersistenceTest, SaveAndLoadBasicCollection) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;
  recovery.lastPersistedLsn = 10;
  recovery.lastPersistedTxid = 5;
  recovery.cleanShutdown = true;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::mt19937 gen(42);
  for (size_t i = 0; i < 100; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    std::string vectorID = std::string("vec-") + std::to_string(i);
    auto result = idSpace.assign(vectorID);
    ASSERT_TRUE(result.ok());
    index.insert(result.value(), vec);

    Metadata meta;
    meta["index"] = static_cast<int64_t>(i);
    meta["label"] = std::string("test");
    metadata[result.value()] = meta;
  }

  std::string savePath = GetTestPath("test_save");
  utils::Status status = CollectionPersistence::save(
    savePath, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  auto loadResult = CollectionPersistence::load(savePath);
  ASSERT_TRUE(loadResult.ok());

  auto& loaded = loadResult.value();
  EXPECT_EQ(loaded.config.name, config.name);
  EXPECT_EQ(loaded.config.dimensions, config.dimensions);
  EXPECT_EQ(loaded.config.space, config.space);
  EXPECT_EQ(loaded.hnswConfig.maxElements, hnswConfig.maxElements);
  EXPECT_EQ(loaded.hnswConfig.M, hnswConfig.M);
  EXPECT_EQ(loaded.hnswConfig.efConstruction, hnswConfig.efConstruction);
  EXPECT_EQ(loaded.recovery.lastPersistedLsn, recovery.lastPersistedLsn);
  EXPECT_EQ(loaded.recovery.lastPersistedTxid, recovery.lastPersistedTxid);
  EXPECT_EQ(loaded.recovery.cleanShutdown, recovery.cleanShutdown);
  EXPECT_EQ(loaded.index->size(), 100);
  EXPECT_EQ(loaded.idSpace.size(), 100);
  EXPECT_EQ(loaded.metadata.size(), 100);
}

TEST_F(CollectionPersistenceTest, SaveCreatesRequiredFiles) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::string savePath = GetTestPath("test_files");
  utils::Status status = CollectionPersistence::save(
    savePath, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  std::filesystem::path path(savePath);
  EXPECT_TRUE(std::filesystem::exists(path / "meta.json"));
  EXPECT_TRUE(std::filesystem::exists(path / "index.bin"));
  EXPECT_TRUE(std::filesystem::exists(path / "id_space.bin"));
}

TEST_F(CollectionPersistenceTest, SaveWithEmptyMetadataStillCreatesFile) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::string savePath = GetTestPath("test_no_metadata");
  utils::Status status = CollectionPersistence::save(
    savePath, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  // metadata.json is always written (even when empty) to prevent stale data on disk
  std::filesystem::path path(savePath);
  EXPECT_TRUE(std::filesystem::exists(path / "metadata.json"));
}

TEST_F(CollectionPersistenceTest, SaveWithMetadataCreatesFile) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  auto result = idSpace.assign("test-id");
  ASSERT_TRUE(result.ok());
  Metadata meta;
  meta["key"] = std::string("value");
  metadata[result.value()] = meta;

  std::string savePath = GetTestPath("test_with_metadata");
  utils::Status status = CollectionPersistence::save(
    savePath, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  std::filesystem::path path(savePath);
  EXPECT_TRUE(std::filesystem::exists(path / "metadata.json"));
}

TEST_F(CollectionPersistenceTest, LoadReturnsErrorOnInvalidDirectory) {
  auto result = CollectionPersistence::load(std::filesystem::path("/nonexistent/path"));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionPersistenceTest, LoadReturnsErrorOnMissingMetaJson) {
  std::string path = GetTestPath("missing_meta");
  std::filesystem::create_directories(path);

  auto result = CollectionPersistence::load(path);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionPersistenceTest, LoadReturnsErrorOnMissingIndexBin) {
  std::string path = GetTestPath("missing_index");
  std::filesystem::create_directories(path);

  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  utils::Status status = CollectionPersistence::save(
    path, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  std::filesystem::path filePath(path);
  std::filesystem::remove(filePath / "index.bin");

  auto result = CollectionPersistence::load(filePath);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionPersistenceTest, WriteDirtyShutdownMarker) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery{};

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::string path = GetTestPath("dirty_shutdown");

  utils::Status saveStatus = CollectionPersistence::save(
    path, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(saveStatus.ok());

  utils::Status status = CollectionPersistence::writeDirtyShutdownMarker(
    path, config, hnswConfig, 5, 3);
  ASSERT_TRUE(status.ok());

  auto loadResult = CollectionPersistence::load(path);
  ASSERT_TRUE(loadResult.ok());

  auto& loaded = loadResult.value();
  EXPECT_EQ(loaded.recovery.lastPersistedLsn, 4);
  EXPECT_EQ(loaded.recovery.lastPersistedTxid, 2);
  EXPECT_FALSE(loaded.recovery.cleanShutdown);
}

TEST_F(CollectionPersistenceTest, WriteDirtyShutdownMarkerHandlesZeroCounters) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery{};

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::string path = GetTestPath("dirty_shutdown_zero");

  utils::Status saveStatus = CollectionPersistence::save(
    path, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(saveStatus.ok());

  utils::Status status = CollectionPersistence::writeDirtyShutdownMarker(
    path, config, hnswConfig, 0, 0);
  ASSERT_TRUE(status.ok());

  auto loadResult = CollectionPersistence::load(path);
  ASSERT_TRUE(loadResult.ok());

  auto& loaded = loadResult.value();
  EXPECT_EQ(loaded.recovery.lastPersistedLsn, 0);
  EXPECT_EQ(loaded.recovery.lastPersistedTxid, 0);
  EXPECT_FALSE(loaded.recovery.cleanShutdown);
}

TEST_F(CollectionPersistenceTest, RoundTripPreservesVectorData) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::mt19937 gen(42);
  std::vector<std::pair<std::string, std::vector<float>>> insertedVectors;

  for (size_t i = 0; i < 50; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    std::string vectorID = std::string("vec-") + std::to_string(i);
    auto result = idSpace.assign(vectorID);
    ASSERT_TRUE(result.ok());
    index.insert(result.value(), vec);
    insertedVectors.push_back({vectorID, vec});
  }

  std::string savePath = GetTestPath("roundtrip_data");
  utils::Status status = CollectionPersistence::save(
    savePath, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  auto loadResult = CollectionPersistence::load(savePath);
  ASSERT_TRUE(loadResult.ok());

  auto& loaded = loadResult.value();

  std::vector<float> query = RandomVector(128, gen);
  auto results = loaded.index->search(query, 10, 200);
  EXPECT_EQ(results.size(), 10);
}

TEST_F(CollectionPersistenceTest, RoundTripPreservesIDSpace) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::mt19937 gen(42);
  std::vector<std::string> vectorIDs;

  for (size_t i = 0; i < 100; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    std::string vectorID = std::string("test-id-") + std::to_string(i);
    auto result = idSpace.assign(vectorID);
    ASSERT_TRUE(result.ok());
    index.insert(result.value(), vec);
    vectorIDs.push_back(vectorID);
  }

  std::string savePath = GetTestPath("roundtrip_idspace");
  utils::Status status = CollectionPersistence::save(
    savePath, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  auto loadResult = CollectionPersistence::load(savePath);
  ASSERT_TRUE(loadResult.ok());

  auto& loaded = loadResult.value();

  for (const auto& vectorID : vectorIDs) {
    auto result = loaded.idSpace.lookup(vectorID);
    ASSERT_TRUE(result.ok()) << "Failed to lookup " << vectorID;
  }

  EXPECT_EQ(loaded.idSpace.size(), 100);
}

TEST_F(CollectionPersistenceTest, RoundTripPreservesMetadata) {
  InternalConfig config = CreateTestConfig();
  HNSWConfig hnswConfig = CreateTestHNSWConfig();
  RecoveryMetadata recovery;

  HNSWIndex index(128, Space::Cosine, hnswConfig);
  IDSpace idSpace;
  std::unordered_map<InternalID, Metadata> metadata;

  std::mt19937 gen(42);
  std::vector<std::pair<InternalID, Metadata>> originalMetadata;

  for (size_t i = 0; i < 50; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    std::string vectorID = std::string("meta-") + std::to_string(i);
    auto result = idSpace.assign(vectorID);
    ASSERT_TRUE(result.ok());
    InternalID internalID = result.value();
    index.insert(internalID, vec);

    Metadata meta;
    meta["index"] = static_cast<int64_t>(i);
    meta["label"] = std::string("test-") + std::to_string(i);
    meta["score"] = 0.95 + i * 0.001;
    meta["active"] = (i % 2 == 0);
    metadata[internalID] = meta;
    originalMetadata.push_back({internalID, meta});
  }

  std::string savePath = GetTestPath("roundtrip_metadata");
  utils::Status status = CollectionPersistence::save(
    savePath, config, hnswConfig, index, idSpace, metadata, recovery);
  ASSERT_TRUE(status.ok());

  auto loadResult = CollectionPersistence::load(savePath);
  ASSERT_TRUE(loadResult.ok());

  auto& loaded = loadResult.value();
  EXPECT_EQ(loaded.metadata.size(), 50);

  for (const auto& [internalID, originalMeta] : originalMetadata) {
    auto it = loaded.metadata.find(internalID);
    ASSERT_NE(it, loaded.metadata.end()) << "Missing metadata for internal ID " << internalID;

    const auto& loadedMeta = it->second;
    EXPECT_EQ(loadedMeta.size(), originalMeta.size());
    EXPECT_EQ(std::get<int64_t>(loadedMeta.at("index")), std::get<int64_t>(originalMeta.at("index")));
    EXPECT_EQ(std::get<std::string>(loadedMeta.at("label")), std::get<std::string>(originalMeta.at("label")));
    EXPECT_DOUBLE_EQ(std::get<double>(loadedMeta.at("score")), std::get<double>(originalMeta.at("score")));
    EXPECT_EQ(std::get<bool>(loadedMeta.at("active")), std::get<bool>(originalMeta.at("active")));
  }
}
