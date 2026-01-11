// Copyright 2025 ArrowDB
#include "arrow/arrow.h"
#include "test_util.h"
#include <filesystem>
#include <gtest/gtest.h>

using namespace arrow;
using arrow::testing::RandomVector;

class ArrowDBTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_db_test";
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

TEST_F(ArrowDBTest, CreateDatabase) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  EXPECT_EQ(db.dataDir(), testDir);
  EXPECT_TRUE(db.listCollections().empty());
}

TEST_F(ArrowDBTest, CreateCollection) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .metric = DistanceMetric::Cosine
  };

  auto result = db.createCollection("test_collection", config);
  ASSERT_TRUE(result.ok()) << result.status().message();

  Collection* collection = result.value();
  EXPECT_EQ(collection->name(), "test_collection");
  EXPECT_EQ(collection->dimension(), 128);
  EXPECT_EQ(collection->metric(), DistanceMetric::Cosine);

  // Verify it's in the list
  auto collections = db.listCollections();
  EXPECT_EQ(collections.size(), 1);
  EXPECT_EQ(collections[0], "test_collection");
}

TEST_F(ArrowDBTest, CreateDuplicateCollectionFails) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .metric = DistanceMetric::Cosine
  };

  auto result1 = db.createCollection("test_collection", config);
  ASSERT_TRUE(result1.ok());

  // Try to create again with same name
  auto result2 = db.createCollection("test_collection", config);
  EXPECT_FALSE(result2.ok());
  EXPECT_EQ(result2.status().code(), utils::StatusCode::kAlreadyExists);
}

TEST_F(ArrowDBTest, GetCollection) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .metric = DistanceMetric::Cosine
  };

  db.createCollection("test_collection", config);

  auto result = db.getCollection("test_collection");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value()->name(), "test_collection");
}

TEST_F(ArrowDBTest, GetNonExistentCollectionFails) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  auto result = db.getCollection("nonexistent");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(ArrowDBTest, DropCollection) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .metric = DistanceMetric::Cosine
  };

  db.createCollection("test_collection", config);
  EXPECT_TRUE(db.hasCollection("test_collection"));

  auto status = db.dropCollection("test_collection");
  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(db.hasCollection("test_collection"));
}

TEST_F(ArrowDBTest, HasCollection) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  EXPECT_FALSE(db.hasCollection("test_collection"));

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .metric = DistanceMetric::Cosine
  };

  db.createCollection("test_collection", config);
  EXPECT_TRUE(db.hasCollection("test_collection"));
}

TEST_F(ArrowDBTest, InsertAndSearchWithQuery) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .metric = DistanceMetric::Cosine
  };

  auto result = db.createCollection("test_collection", config);
  ASSERT_TRUE(result.ok());

  Collection* collection = result.value();

  // Insert vectors with metadata
  std::mt19937 gen(42);
  for (size_t i = 0; i < 100; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    collection->insert(i, vec);

    Metadata meta;
    meta["category"] = std::string("test");
    meta["index"] = static_cast<int64_t>(i);
    collection->setMetadata(i, meta);
  }

  // Use new query() method that returns SearchResult
  std::vector<float> queryVec = RandomVector(128, gen);
  SearchResult searchResult = collection->query(queryVec, 10);

  EXPECT_EQ(searchResult.hits.size(), 10);

  // Check that hits have metadata
  for (const auto& hit : searchResult.hits) {
    EXPECT_TRUE(hit.metadata.contains("category"));
    EXPECT_TRUE(hit.metadata.contains("index"));
    EXPECT_EQ(hit.metadata["category"], "test");
  }
}

TEST_F(ArrowDBTest, MultipleCollections) {
  ClientOptions options{.data_dir = testDir};
  ArrowDB db(options);

  for (int i = 0; i < 3; ++i) {
    CollectionConfig config{
        .name = "collection_" + std::to_string(i),
        .dimensions = static_cast<uint32_t>(64 + i * 32),
        .metric = DistanceMetric::Cosine
    };
    db.createCollection("collection_" + std::to_string(i), config);
  }

  EXPECT_EQ(db.listCollections().size(), 3);

  for (int i = 0; i < 3; ++i) {
    auto result = db.getCollection("collection_" + std::to_string(i));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value()->dimension(), static_cast<uint32_t>(64 + i * 32));
  }
}
