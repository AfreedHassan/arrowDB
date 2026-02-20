#include "common.h"
#include <gtest/gtest.h>

TEST_F(CollectionBatchTest, InsertBatchSuccess) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  std::mt19937 gen(42);
  for (size_t i = 0; i < 100; ++i) {
    batch.push_back({std::to_string(i), RandomVector(128, gen)});
  }

  auto result = collection.insertBatch(batch);
  ASSERT_TRUE(result.ok());

  auto& batchResult = result.value();
  EXPECT_EQ(batchResult.successCount, 100);
  EXPECT_EQ(batchResult.failureCount, 0);
  EXPECT_EQ(collection.size(), 100);
}

TEST_F(CollectionBatchTest, InsertBatchPartialFailure) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  std::mt19937 gen(42);

  batch.push_back({"0", RandomVector(128, gen)});
  batch.push_back({"1", RandomVector(64, gen)});
  batch.push_back({"2", RandomVector(128, gen)});

  auto result = collection.insertBatch(batch);
  ASSERT_TRUE(result.ok());

  auto& batchResult = result.value();
  EXPECT_EQ(batchResult.successCount, 2);
  EXPECT_EQ(batchResult.failureCount, 1);

  EXPECT_TRUE(batchResult.results[0].status.ok());
  EXPECT_FALSE(batchResult.results[1].status.ok());
  EXPECT_EQ(batchResult.results[1].status.code(), utils::StatusCode::kDimensionMismatch);
  EXPECT_TRUE(batchResult.results[2].status.ok());

  EXPECT_EQ(collection.size(), 2);
}

TEST_F(CollectionBatchTest, SearchBatchParallel) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  std::mt19937 gen(42);
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  for (size_t i = 0; i < 1000; ++i) {
    batch.push_back({std::to_string(i), RandomVector(128, gen)});
  }
  auto insertResult = collection.insertBatch(batch);
  ASSERT_TRUE(insertResult.ok());

  std::vector<std::vector<float>> queries;
  for (size_t i = 0; i < 10; ++i) {
    queries.push_back(RandomVector(128, gen));
  }

  auto resultOrErr = collection.searchBatch(queries, 5);
  ASSERT_TRUE(resultOrErr.ok());

  auto& results = resultOrErr.value();
  EXPECT_EQ(results.size(), 10);
  for (const auto& queryResults : results) {
    EXPECT_EQ(queryResults.size(), 5);
    for (size_t i = 1; i < queryResults.size(); ++i) {
      EXPECT_LE(queryResults[i].score, queryResults[i-1].score);
    }
  }
}

TEST_F(CollectionBatchTest, SearchBatchDimensionMismatch) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  std::mt19937 gen(42);
  std::vector<float> vec = RandomVector(128, gen);
  collection.insert(uuidv4(0), vec);

  std::vector<std::vector<float>> queries;
  queries.push_back(RandomVector(64, gen));

  auto result = collection.searchBatch(queries, 5);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kDimensionMismatch);
}

TEST_F(CollectionBatchTest, InsertBatchDataIntegrity) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::L2};
  Collection collection(cfg);

  constexpr size_t kCount = 25000;  // well above parallel threshold (1000)
  std::mt19937 gen(42);

  // Store original vectors for verification
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  batch.reserve(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    batch.push_back({std::to_string(i), RandomVector(128, gen)});
  }

  auto result = collection.insertBatch(batch);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, kCount);

  // Verify every vector via get() — exact equality
  for (size_t i = 0; i < kCount; ++i) {
    auto getResult = collection.get(batch[i].first);
    ASSERT_TRUE(getResult.ok()) << "get() failed for id=" << batch[i].first;
    EXPECT_EQ(getResult.value(), batch[i].second)
        << "Vector mismatch at id=" << batch[i].first;
  }
}

TEST_F(CollectionBatchTest, InsertBatchSearchIntegrity) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  constexpr size_t kCount = 2000;
  std::mt19937 gen(42);

  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  batch.reserve(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    batch.push_back({std::to_string(i), RandomVector(128, gen)});
  }

  auto result = collection.insertBatch(batch);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, kCount);

  // Sample 20 vectors and verify search(vec, 1) returns self as top-1
  std::uniform_int_distribution<size_t> dist(0, kCount - 1);
  for (int s = 0; s < 20; ++s) {
    size_t idx = dist(gen);
    auto searchResult = collection.search(batch[idx].second, 1);
    ASSERT_FALSE(searchResult.empty()) << "search returned empty for idx=" << idx;
    EXPECT_EQ(searchResult[0].id, batch[idx].first)
        << "Top-1 mismatch for idx=" << idx;
    // Cosine distance: 0 = identical vectors
    EXPECT_NEAR(searchResult[0].score, 0.0f, 1e-5f)
        << "Self-distance not ~0.0 for idx=" << idx;
  }
}

TEST_F(CollectionBatchTest, InsertBatchMetadataIntegrity) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  constexpr size_t kCount = 2000;
  std::mt19937 gen(42);

  std::vector<Document> docs;
  docs.reserve(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    Document doc;
    doc.id = std::to_string(i);
    doc.embedding = RandomVector(128, gen);
    doc.metadata = {
        {"idx", static_cast<int64_t>(i)},
        {"label", "vec_" + std::to_string(i)}};
    docs.push_back(std::move(doc));
  }

  auto result = collection.insertBatch(std::move(docs));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, kCount);

  // Verify every document's metadata
  for (size_t i = 0; i < kCount; ++i) {
    auto metaResult = collection.getMetadata(std::to_string(i));
    ASSERT_TRUE(metaResult.ok()) << "getMetadata failed for id=" << i;
    auto& meta = metaResult.value();

    auto idxIt = meta.find("idx");
    ASSERT_NE(idxIt, meta.end()) << "Missing 'idx' for id=" << i;
    EXPECT_EQ(std::get<int64_t>(idxIt->second), static_cast<int64_t>(i));

    auto labelIt = meta.find("label");
    ASSERT_NE(labelIt, meta.end()) << "Missing 'label' for id=" << i;
    EXPECT_EQ(std::get<std::string>(labelIt->second), "vec_" + std::to_string(i));
  }
}

TEST_F(CollectionBatchTest, InsertBatchNoOverwrite) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::L2};
  Collection collection(cfg);

  std::mt19937 gen(42);

  // First batch: IDs 0-99
  std::vector<std::pair<VectorID, std::vector<float>>> batch1;
  for (size_t i = 0; i < 100; ++i) {
    batch1.push_back({std::to_string(i), RandomVector(128, gen)});
  }
  auto result1 = collection.insertBatch(batch1);
  ASSERT_TRUE(result1.ok());
  EXPECT_EQ(result1.value().successCount, 100);

  // Second batch: IDs 100-199
  std::vector<std::pair<VectorID, std::vector<float>>> batch2;
  for (size_t i = 100; i < 200; ++i) {
    batch2.push_back({std::to_string(i), RandomVector(128, gen)});
  }
  auto result2 = collection.insertBatch(batch2);
  ASSERT_TRUE(result2.ok());
  EXPECT_EQ(result2.value().successCount, 100);
  EXPECT_EQ(collection.size(), 200);

  // Verify all original vectors are intact
  for (size_t i = 0; i < 100; ++i) {
    auto getResult = collection.get(batch1[i].first);
    ASSERT_TRUE(getResult.ok()) << "get() failed for id=" << batch1[i].first;
    EXPECT_EQ(getResult.value(), batch1[i].second)
        << "First batch vector overwritten at id=" << batch1[i].first;
  }
}

TEST_F(CollectionBatchTest, InsertBatchPartialFailureIntegrity) {
  CollectionConfig cfg{.name = "test", .dimensions = 128, .space = Space::L2};
  Collection collection(cfg);

  std::mt19937 gen(42);

  // Build batch with invalid vectors scattered throughout
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  std::vector<size_t> validIndices;
  std::vector<size_t> invalidIndices;

  for (size_t i = 0; i < 100; ++i) {
    // Every 7th vector has wrong dimensions
    if (i % 7 == 3) {
      batch.push_back({std::to_string(i), RandomVector(64, gen)});  // wrong dim
      invalidIndices.push_back(i);
    } else {
      batch.push_back({std::to_string(i), RandomVector(128, gen)});
      validIndices.push_back(i);
    }
  }

  auto result = collection.insertBatch(batch);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, validIndices.size());
  EXPECT_EQ(result.value().failureCount, invalidIndices.size());

  // Valid vectors are retrievable with exact data
  for (size_t idx : validIndices) {
    auto getResult = collection.get(batch[idx].first);
    ASSERT_TRUE(getResult.ok()) << "get() failed for valid id=" << idx;
    EXPECT_EQ(getResult.value(), batch[idx].second)
        << "Vector mismatch at valid id=" << idx;
  }

  // Invalid IDs return not-found
  for (size_t idx : invalidIndices) {
    auto getResult = collection.get(batch[idx].first);
    EXPECT_FALSE(getResult.ok()) << "get() should fail for invalid id=" << idx;
    EXPECT_EQ(getResult.status().code(), utils::StatusCode::kNotFound);
  }
}

TEST_F(CollectionBatchTest, InsertBatchWithPersistence) {
  CollectionConfig config{.name = "test", .dimensions = 128, .space = Space::Cosine};
  std::string persistencePath = GetTestPath("batch_wal");

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    std::vector<std::pair<VectorID, std::vector<float>>> batch;
    std::mt19937 gen(42);
    for (size_t i = 0; i < 50; ++i) {
      batch.push_back({std::to_string(i), RandomVector(128, gen)});
    }

    auto result = collection.insertBatch(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().successCount, 50);

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok());
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok());
  Collection loaded = std::move(loadResult.value());

  EXPECT_EQ(loaded.size(), 50);
}
