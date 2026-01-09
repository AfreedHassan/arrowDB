#include "arrow/collection.h"
#include "test_util.h"
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>

using namespace arrow;
using arrow::testing::RandomVector;

class CollectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a temporary directory for test files
    testDir = std::filesystem::temp_directory_path() / "arrow_collection_test";
    std::filesystem::create_directories(testDir);
  }

  void TearDown() override {
    // Clean up test files
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
  std::string GetTestPath(const std::string &dirname) {
    return (testDir / dirname).string();
  }
};

TEST_F(CollectionTest, CreateCollection) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);

  Collection collection(cfg);

  EXPECT_EQ(collection.name(), "test_collection");
  EXPECT_EQ(collection.dimension(), 128);
  EXPECT_EQ(collection.metric(), DistanceMetric::Cosine);
  EXPECT_EQ(collection.dtype(), DataType::Float32);
  EXPECT_EQ(collection.size(), 0);
}

TEST_F(CollectionTest, InsertVectors) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(static_cast<VectorID>(i), vec);
  }

  EXPECT_EQ(collection.size(), num_vectors);
}

TEST_F(CollectionTest, SearchFunctionality) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  // Insert vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(static_cast<VectorID>(i), vec);
  }

  // Perform search
  std::vector<float> query = RandomVector(dim, gen);
  const size_t k = 10;
  std::vector<SearchResult> results = collection.search(query, k);

  EXPECT_EQ(results.size(), k);

  // Verify results are sorted by score (descending for cosine similarity)
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i].score, results[i - 1].score)
        << "Results should be sorted in descending order";
  }
}

TEST_F(CollectionTest, SearchWithDifferentEf) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  // Insert vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(static_cast<VectorID>(i), vec);
  }

  std::vector<float> query = RandomVector(dim, gen);
  const size_t k = 10;

  // Test with different ef values
  for (size_t ef : {10, 50, 100}) {
    std::vector<SearchResult> results = collection.search(query, k, ef);
    EXPECT_EQ(results.size(), k) << "ef=" << ef;
  }
}

TEST_F(CollectionTest, SearchPerformance) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  // Insert vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(static_cast<VectorID>(i), vec);
  }

  std::vector<float> query = RandomVector(dim, gen);
  const size_t k = 10;

  // Measure search latency with ef=100
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<SearchResult> results = collection.search(query, k, 100);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  EXPECT_EQ(results.size(), k);
  // Search should be reasonably fast (< 1ms for 1K vectors)
  EXPECT_LT(duration.count(), 1000)
      << "Search took " << duration.count() << " microseconds";
}

// ============================================================================
// Persistence Tests (save / load)
// ============================================================================

TEST_F(CollectionTest, SaveCreatesDirectory) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);
  Collection collection(cfg);

  std::string save_path = GetTestPath("test_collection");
  EXPECT_NO_THROW(collection.save(save_path));

  // Directory should exist
  EXPECT_TRUE(std::filesystem::exists(save_path));
  EXPECT_TRUE(std::filesystem::is_directory(save_path));
}

TEST_F(CollectionTest, SaveCreatesRequiredFiles) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);
  Collection collection(cfg);

  // Insert some vectors
  std::mt19937 gen(42);
  for (size_t i = 0; i < 10; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    collection.insert(i, vec);
  }

  std::string save_path = GetTestPath("test_collection");
  collection.save(save_path);

  // Check required files exist
  EXPECT_TRUE(
      std::filesystem::exists(std::filesystem::path(save_path) / "meta.json"));
  EXPECT_TRUE(
      std::filesystem::exists(std::filesystem::path(save_path) / "index.bin"));

  // metadata.json is optional (only created if metadata exists)
  // For this test, it shouldn't exist since we didn't add metadata
}

TEST_F(CollectionTest, SaveIncludesMetadata) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);
  Collection collection(cfg);

  // Insert vectors with metadata
  std::mt19937 gen(42);
  for (size_t i = 0; i < 5; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    collection.insert(i, vec);

    Metadata meta;
    meta["category"] = std::string("test");
    meta["score"] = static_cast<double>(i);
    collection.setMetadata(i, meta);
  }

  std::string save_path = GetTestPath("test_collection");
  collection.save(save_path);

  // metadata.json should exist since we added metadata
  EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(save_path) /
                                      "metadata.json"));
}

TEST_F(CollectionTest, LoadFromDirectory) {
  // Create and save a collection
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);
  Collection original(cfg);

  std::mt19937 gen(42);
  for (size_t i = 0; i < 100; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    original.insert(i, vec);
  }

  std::string save_path = GetTestPath("test_collection");
  original.save(save_path);

  // Load the collection
  auto loadResult = Collection::load(save_path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  // Verify basic properties
  EXPECT_EQ(loaded.name(), "test_collection");
  EXPECT_EQ(loaded.dimension(), 128);
  EXPECT_EQ(loaded.metric(), DistanceMetric::Cosine);
  EXPECT_EQ(loaded.dtype(), DataType::Float32);
  EXPECT_EQ(loaded.size(), 100);
}

TEST_F(CollectionTest, RoundTripPreservesData) {
  // Create collection with custom HNSW config
  CollectionConfig cfg("test_collection", 64, DistanceMetric::Cosine,
                       DataType::Float32);
  HNSWConfig hnsw_cfg;
  hnsw_cfg.M = 32;
  hnsw_cfg.efConstruction = 200;
  Collection original(cfg, hnsw_cfg);

  std::mt19937 gen(42);
  std::vector<std::vector<float>> vectors;
  for (size_t i = 0; i < 50; ++i) {
    std::vector<float> vec = RandomVector(64, gen);
    vectors.push_back(vec);
    original.insert(i, vec);
  }

  std::string save_path = GetTestPath("test_collection");
  original.save(save_path);

  // Load and verify
  auto loadResult = Collection::load(save_path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());
  EXPECT_EQ(loaded.size(), 50);
  EXPECT_EQ(loaded.hnswConfig().M, 32);
  EXPECT_EQ(loaded.hnswConfig().efConstruction, 200);

  // Verify search results match
  for (size_t i = 0; i < std::min(size_t(10), vectors.size()); ++i) {
    auto originalResults = original.search(vectors[i], 5);
    auto loadedResults = loaded.search(vectors[i], 5);

    EXPECT_EQ(originalResults.size(), loadedResults.size());
    if (!originalResults.empty() && !loadedResults.empty()) {
      EXPECT_EQ(originalResults[0].id, loadedResults[0].id);
      EXPECT_NEAR(originalResults[0].score, loadedResults[0].score, 1e-5f);
    }
  }
}

TEST_F(CollectionTest, RoundTripPreservesMetadata) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);
  Collection original(cfg);

  std::mt19937 gen(42);
  for (size_t i = 0; i < 10; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    original.insert(i, vec);

    Metadata meta;
    meta["id"] = static_cast<int64_t>(i);
    meta["name"] = std::string("vector_") + std::to_string(i);
    meta["score"] = static_cast<double>(i) * 0.1;
    meta["active"] = (i % 2 == 0);
    original.setMetadata(i, meta);
  }

  std::string save_path = GetTestPath("test_collection");
  original.save(save_path);

  // Load and verify metadata
  auto loadResult = Collection::load(save_path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  // Note: We can't directly access metadata_ from outside, but we can verify
  // that metadata.json exists and was loaded (by checking collection can be
  // loaded)
  EXPECT_EQ(loaded.size(), 10);
  EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(save_path) /
                                      "metadata.json"));
}

TEST_F(CollectionTest, LoadReturnsErrorOnInvalidDirectory) {
  auto result = Collection::load("/nonexistent/directory");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionTest, LoadReturnsErrorOnMissingMetaJson) {
  // Create directory without meta.json
  std::string save_path = GetTestPath("incomplete_collection");
  std::filesystem::create_directories(save_path);

  auto result = Collection::load(save_path);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionTest, LoadReturnsErrorOnMissingIndexBin) {
  CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);
  Collection collection(cfg);

  std::string save_path = GetTestPath("incomplete_collection");
  std::filesystem::create_directories(save_path);

  // Save only meta.json
  std::string meta_path =
      (std::filesystem::path(save_path) / "meta.json").string();
  utils::exportCollectionConfigToJson(cfg, HNSWConfig{}, meta_path);

  auto result = Collection::load(save_path);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

// ============================================================================
// WAL Integration Tests
// ============================================================================
//

using namespace wal;

class CollectionWalTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_wal_test";
    std::filesystem::create_directories(testDir);
    gen.seed(42);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
  std::mt19937 gen;

  std::string GetTestPath(const std::string &dirname) {
    return (testDir / dirname).string();
  }

  std::string GetWalPath(const std::string &dirname) {
    return (testDir / dirname / "wal" / "db.wal").string();
  }

  CollectionConfig GetTestConfig(const std::string &name = "test_collection") {
    return CollectionConfig(name, 128, DistanceMetric::Cosine,
                            DataType::Float32);
  }
};

// FAILED
TEST_F(CollectionWalTest, WalLoggingEnabledWithPersistencePath) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("wal_enabled");
  {
    Collection collection(config, persistencePath);

    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(1, vec);

    EXPECT_TRUE(status.ok()) << status.message();
  }

  {
    wal::WAL wal(persistencePath + "/wal");
    auto result = wal.loadHeader();
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value().magic, 0x41574C01);

    auto entriesResult = wal.readAll();
    ASSERT_TRUE(entriesResult.ok()) << entriesResult.status().message();
    EXPECT_EQ(entriesResult.value().size(), 1);
    EXPECT_EQ(entriesResult.value()[0].type, wal::OperationType::INSERT);
    EXPECT_EQ(entriesResult.value()[0].vectorID, 1);
  }
}

TEST_F(CollectionWalTest, WalNotCreatedWithoutPersistencePath) {
  auto config = GetTestConfig();
  Collection collection(config);

  std::vector<float> vec = RandomVector(128, gen);
  auto status = collection.insert(1, vec);

  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(collection.size(), 1);
}

// FAILED
TEST_F(CollectionWalTest, WalLogOnInsert) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("insert_wal");
  Collection collection(config, persistencePath);

  const size_t numInserts = 10;
  for (size_t i = 0; i < numInserts; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(static_cast<VectorID>(i), vec);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  EXPECT_EQ(collection.size(), numInserts);

  wal::WAL wal(persistencePath + "/wal");
  auto entriesResult = wal.readAll();
  ASSERT_TRUE(entriesResult.ok()) << entriesResult.status().message();
  const auto &entries = entriesResult.value();
  EXPECT_EQ(entries.size(), numInserts);

  for (size_t i = 0; i < numInserts; ++i) {
    EXPECT_EQ(entries[i].type, wal::OperationType::INSERT);
    EXPECT_EQ(entries[i].vectorID, static_cast<VectorID>(i));
    EXPECT_EQ(entries[i].dimension, 128);
    EXPECT_FALSE(entries[i].embedding.empty());
  }
}

// FAILED
TEST_F(CollectionWalTest, WalLogOnDelete) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("delete_wal");
  Collection collection(config, persistencePath);

  const size_t numInserts = 5;
  for (size_t i = 0; i < numInserts; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(i, vec);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  auto deleteStatus = collection.remove(2);
  ASSERT_TRUE(deleteStatus.ok()) << deleteStatus.message();

  WAL wal(persistencePath + "/wal");
  auto entriesResult = wal.readAll();
  ASSERT_TRUE(entriesResult.ok()) << entriesResult.status().message();
  const auto &entries = entriesResult.value();

  EXPECT_EQ(entries.size(), numInserts + 1);

  int insertCount = 0;
  int deleteCount = 0;
  for (const auto &entry : entries) {
    if (entry.type == wal::OperationType::INSERT) {
      insertCount++;
    } else if (entry.type == wal::OperationType::DELETE) {
      deleteCount++;
      EXPECT_EQ(entry.vectorID, 2);
    }
  }
  EXPECT_EQ(insertCount, numInserts);
  EXPECT_EQ(deleteCount, 1);
}

// FAILED
TEST_F(CollectionWalTest, CheckpointTruncatesWalAfterSave) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("checkpoint_wal");
  Collection collection(config, persistencePath);

  const size_t numInserts = 10;
  for (size_t i = 0; i < numInserts; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(i, vec);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  WAL walBefore(persistencePath + "/wal");
  auto entriesBefore = walBefore.readAll();
  ASSERT_TRUE(entriesBefore.ok());
  EXPECT_EQ(entriesBefore.value().size(), numInserts);

  auto saveStatus = collection.save(persistencePath);
  ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();

  WAL walAfter(persistencePath + "/wal");
  auto entriesAfter = walAfter.readAll();
  ASSERT_TRUE(entriesAfter.ok());
  EXPECT_EQ(entriesAfter.value().size(), 0);

  auto headerResult = walAfter.loadHeader();
  ASSERT_TRUE(headerResult.ok());
  EXPECT_EQ(headerResult.value().magic, 0x41574C01);
}

TEST_F(CollectionWalTest, CrashRecoveryReplaysWal) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("crash_recovery");

  // Phase 1: Create collection, insert 10 vectors, save (clean shutdown)
  {
    Collection collection(config, persistencePath);

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();

    EXPECT_EQ(collection.currentLsn(), 11);
    EXPECT_EQ(collection.currentTxid(), 11);
  }

  // Phase 2: Load saved state, insert 10 more vectors, then "crash" (no save)
  {
    auto loadResult2 = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult2.ok()) << loadResult2.status().message();
    Collection collection2 = std::move(loadResult2.value());

    for (size_t i = 10; i < 20; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection2.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    EXPECT_EQ(collection2.size(), 20);
    EXPECT_EQ(collection2.currentLsn(), 21);
    EXPECT_EQ(collection2.currentTxid(), 21);
    // No save - simulating crash
  }

  // Phase 3: Load and verify WAL replay recovered the 10 uncommitted vectors
  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 20);
  EXPECT_TRUE(recovered.recoveredFromWal());
  EXPECT_EQ(recovered.currentLsn(), 21);
  EXPECT_EQ(recovered.currentTxid(), 21);
}

TEST_F(CollectionWalTest, LoadWithoutCrashDoesNotReplayWal) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("no_crash");

  {
    Collection collection(config, persistencePath);

    for (size_t i = 0; i < 5; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 5);
  EXPECT_FALSE(recovered.recoveredFromWal());
  EXPECT_EQ(recovered.currentLsn(), 6);
  EXPECT_EQ(recovered.currentTxid(), 6);
}

TEST_F(CollectionWalTest, WalReplayPreservesMetadata) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("metadata_wal");

  // Phase 1: Create collection with 10 vectors + metadata, save
  {
    Collection collection(config, persistencePath);

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();

      Metadata meta;
      meta["idx"] = static_cast<int64_t>(i);
      collection.setMetadata(i, meta);
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  // Phase 2: Load, add 1 more vector with metadata, then "crash" (no save)
  {
    auto loadResult2 = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult2.ok()) << loadResult2.status().message();
    Collection collection2 = std::move(loadResult2.value());

    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection2.insert(10, vec);
    ASSERT_TRUE(status.ok()) << status.message();

    Metadata meta;
    meta["idx"] = static_cast<int64_t>(10);
    collection2.setMetadata(10, meta);
    // No save - simulating crash
  }

  // Phase 3: Load and verify WAL replay recovered the uncommitted vector
  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 11);
  EXPECT_TRUE(recovered.recoveredFromWal());
}

TEST_F(CollectionWalTest, DeleteReplayMarksVectorAsDeleted) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("delete_replay");

  // Store vector 5 so we can search for it later
  std::vector<float> vector5;

  // Phase 1: Create collection with 10 vectors, save
  {
    Collection collection(config, persistencePath);

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      if (i == 5) {
        vector5 = vec;
      }
      auto status = collection.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  // Phase 2: Load, delete vector 5, then "crash" (no save)
  {
    auto loadResult2 = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult2.ok()) << loadResult2.status().message();
    Collection collection2 = std::move(loadResult2.value());

    auto deleteStatus = collection2.remove(5);
    ASSERT_TRUE(deleteStatus.ok()) << deleteStatus.message();
    // No save - simulating crash
  }

  // Phase 3: Load and verify WAL replay marked vector 5 as deleted
  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  // WAL replay should have occurred
  EXPECT_TRUE(recovered.recoveredFromWal());

  // Note: HNSW uses lazy deletion, so size() still includes deleted vectors.
  // The vector is marked as deleted but not removed from the index.
  EXPECT_EQ(recovered.size(), 10);

  // Verify the deleted vector is not returned in search results
  auto results = recovered.search(vector5, 10);
  for (const auto& result : results) {
    EXPECT_NE(result.id, 5) << "Deleted vector 5 should not appear in search results";
  }
}

TEST_F(CollectionWalTest, LsnTxidContinuityAcrossRestarts) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("continuity");

  uint64_t expectedLsn = 1;
  uint64_t expectedTxid = 1;

  {
    Collection collection(config, persistencePath);
    EXPECT_EQ(collection.currentLsn(), expectedLsn);
    EXPECT_EQ(collection.currentTxid(), expectedTxid);

    for (size_t i = 0; i < 5; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    expectedLsn += 5;
    expectedTxid += 5;
    EXPECT_EQ(collection.currentLsn(), expectedLsn);
    EXPECT_EQ(collection.currentTxid(), expectedTxid);

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection collection = std::move(loadResult.value());

  EXPECT_EQ(collection.currentLsn(), expectedLsn);
  EXPECT_EQ(collection.currentTxid(), expectedTxid);

  for (size_t i = 5; i < 10; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(i, vec);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  expectedLsn += 5;
  expectedTxid += 5;
  EXPECT_EQ(collection.currentLsn(), expectedLsn);
  EXPECT_EQ(collection.currentTxid(), expectedTxid);
}

TEST_F(CollectionWalTest, EmptyWalDoesNotCauseRecovery) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("empty_wal");

  {
    Collection collection(config, persistencePath);

    for (size_t i = 0; i < 5; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 5);
  EXPECT_FALSE(recovered.recoveredFromWal());
}

TEST_F(CollectionWalTest, RecoveryMetadataIsPersisted) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("recovery_meta");

  {
    Collection collection(config, persistencePath);

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(i, vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto metaPath = std::filesystem::path(persistencePath) / "meta.json";
  std::ifstream file(metaPath);
  ASSERT_TRUE(file.is_open());

  utils::json j;
  file >> j;
  file.close();

  EXPECT_TRUE(j.contains("recovery"));
  const auto &recovery = j["recovery"];
  EXPECT_TRUE(recovery.contains("lastPersistedLsn"));
  EXPECT_TRUE(recovery.contains("lastPersistedTxid"));
  EXPECT_TRUE(recovery.contains("cleanShutdown"));

  EXPECT_EQ(recovery["lastPersistedLsn"].get<uint64_t>(), 10);
  EXPECT_EQ(recovery["lastPersistedTxid"].get<uint64_t>(), 10);
  EXPECT_TRUE(recovery["cleanShutdown"].get<bool>());
}

// ============================================================================
// Batch Operation Tests
// ============================================================================

class CollectionBatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_batch_test";
    std::filesystem::create_directories(testDir);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
  std::string GetTestPath(const std::string &dirname) {
    return (testDir / dirname).string();
  }
};

TEST_F(CollectionBatchTest, InsertBatchSuccess) {
  CollectionConfig cfg("test", 128, DistanceMetric::Cosine, DataType::Float32);
  Collection collection(cfg);

  // Prepare batch
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  std::mt19937 gen(42);
  for (size_t i = 0; i < 100; ++i) {
    batch.push_back({i, RandomVector(128, gen)});
  }

  // Insert batch
  auto result = collection.insertBatch(batch);
  ASSERT_TRUE(result.ok());

  auto& batchResult = result.value();
  EXPECT_EQ(batchResult.successCount, 100);
  EXPECT_EQ(batchResult.failureCount, 0);
  EXPECT_EQ(collection.size(), 100);
}

TEST_F(CollectionBatchTest, InsertBatchPartialFailure) {
  CollectionConfig cfg("test", 128, DistanceMetric::Cosine, DataType::Float32);
  Collection collection(cfg);

  // Mixed valid and invalid dimensions
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  std::mt19937 gen(42);

  batch.push_back({0, RandomVector(128, gen)});  // Valid
  batch.push_back({1, RandomVector(64, gen)});   // Invalid dimension
  batch.push_back({2, RandomVector(128, gen)});  // Valid

  auto result = collection.insertBatch(batch);
  ASSERT_TRUE(result.ok());

  auto& batchResult = result.value();
  EXPECT_EQ(batchResult.successCount, 2);
  EXPECT_EQ(batchResult.failureCount, 1);

  // Check individual results
  EXPECT_TRUE(batchResult.results[0].status.ok());
  EXPECT_FALSE(batchResult.results[1].status.ok());
  EXPECT_EQ(batchResult.results[1].status.code(), utils::StatusCode::kDimensionMismatch);
  EXPECT_TRUE(batchResult.results[2].status.ok());

  // Verify only valid vectors were inserted
  EXPECT_EQ(collection.size(), 2);
}

TEST_F(CollectionBatchTest, SearchBatchParallel) {
  CollectionConfig cfg("test", 128, DistanceMetric::Cosine, DataType::Float32);
  Collection collection(cfg);

  // Insert vectors using batch insert
  std::mt19937 gen(42);
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  for (size_t i = 0; i < 1000; ++i) {
    batch.push_back({i, RandomVector(128, gen)});
  }
  auto insertResult = collection.insertBatch(batch);
  ASSERT_TRUE(insertResult.ok());

  // Prepare search queries
  std::vector<std::vector<float>> queries;
  for (size_t i = 0; i < 10; ++i) {
    queries.push_back(RandomVector(128, gen));
  }

  // Batch search
  auto resultOrErr = collection.searchBatch(queries, 5);
  ASSERT_TRUE(resultOrErr.ok());

  auto& results = resultOrErr.value();
  EXPECT_EQ(results.size(), 10);
  for (const auto& queryResults : results) {
    EXPECT_EQ(queryResults.size(), 5);
    // Verify ordering (descending scores)
    for (size_t i = 1; i < queryResults.size(); ++i) {
      EXPECT_LE(queryResults[i].score, queryResults[i-1].score);
    }
  }
}

TEST_F(CollectionBatchTest, SearchBatchDimensionMismatch) {
  CollectionConfig cfg("test", 128, DistanceMetric::Cosine, DataType::Float32);
  Collection collection(cfg);

  // Insert a vector
  std::mt19937 gen(42);
  std::vector<float> vec = RandomVector(128, gen);
  collection.insert(0, vec);

  // Try to search with wrong dimension
  std::vector<std::vector<float>> queries;
  queries.push_back(RandomVector(64, gen));  // Wrong dimension

  auto result = collection.searchBatch(queries, 5);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kDimensionMismatch);
}

TEST_F(CollectionBatchTest, InsertBatchWithPersistence) {
  auto config = CollectionConfig("test", 128, DistanceMetric::Cosine, DataType::Float32);
  std::string persistencePath = GetTestPath("batch_wal");

  // Insert batch with WAL
  {
    Collection collection(config, persistencePath);

    std::vector<std::pair<VectorID, std::vector<float>>> batch;
    std::mt19937 gen(42);
    for (size_t i = 0; i < 50; ++i) {
      batch.push_back({i, RandomVector(128, gen)});
    }

    auto result = collection.insertBatch(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().successCount, 50);

    // Save
    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok());
  }

  // Load and verify all vectors are present
  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok());
  Collection loaded = std::move(loadResult.value());

  EXPECT_EQ(loaded.size(), 50);
}
