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
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  EXPECT_EQ(db.dataDir(), testDir);
  EXPECT_TRUE(db.listCollections().empty());
}

TEST_F(ArrowDBTest, CreateCollection) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  auto result = db.createCollection("test_collection", config);
  ASSERT_TRUE(result.ok()) << result.status().message();

  Collection* collection = result.value();
  EXPECT_EQ(collection->name(), "test_collection");
  EXPECT_EQ(collection->dimension(), 128);
  EXPECT_EQ(collection->space(), Space::Cosine);

  // Verify it's in the list
  auto collections = db.listCollections();
  EXPECT_EQ(collections.size(), 1);
  EXPECT_EQ(collections[0], "test_collection");
}

TEST_F(ArrowDBTest, CreateDuplicateCollectionFails) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  auto result1 = db.createCollection("test_collection", config);
  ASSERT_TRUE(result1.ok());

  // Try to create again with same name
  auto result2 = db.createCollection("test_collection", config);
  EXPECT_FALSE(result2.ok());
  EXPECT_EQ(result2.status().code(), utils::StatusCode::kAlreadyExists);
}

TEST_F(ArrowDBTest, GetCollection) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  db.createCollection("test_collection", config);

  auto result = db.getCollection("test_collection");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value()->name(), "test_collection");
}

TEST_F(ArrowDBTest, GetNonExistentCollectionFails) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  auto result = db.getCollection("nonexistent");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(ArrowDBTest, DropCollection) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  db.createCollection("test_collection", config);
  EXPECT_TRUE(db.hasCollection("test_collection"));

  auto status = db.dropCollection("test_collection");
  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(db.hasCollection("test_collection"));
}

TEST_F(ArrowDBTest, HasCollection) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  EXPECT_FALSE(db.hasCollection("test_collection"));

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  db.createCollection("test_collection", config);
  EXPECT_TRUE(db.hasCollection("test_collection"));
}

TEST_F(ArrowDBTest, InsertAndSearchWithQuery) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  auto result = db.createCollection("test_collection", config);
  ASSERT_TRUE(result.ok());

  Collection* collection = result.value();

  // Insert vectors with metadata
  std::mt19937 gen(42);
  for (size_t i = 0; i < 100; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    collection->insert(std::to_string(i), vec);

    Metadata meta;
    meta["category"] = std::string("test");
    meta["index"] = static_cast<int64_t>(i);
    collection->setMetadata(std::to_string(i), meta);
  }

  // Use new query() method that returns SearchResult
  std::vector<float> queryVec = RandomVector(128, gen);
  SearchResult searchResult = collection->query(queryVec, 10);

  EXPECT_EQ(searchResult.hits.size(), 10);

  // Check that hits have metadata
  for (const auto& hit : searchResult.hits) {
    EXPECT_TRUE(hit.metadata.contains("category"));
    EXPECT_TRUE(hit.metadata.contains("index"));
    EXPECT_EQ(hit.metadata.at("category").asString(), "test");
  }
}

TEST_F(ArrowDBTest, MultipleCollections) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  for (int i = 0; i < 3; ++i) {
    CollectionConfig config{
        .name = "collection_" + std::to_string(i),
        .dimensions = static_cast<uint32_t>(64 + i * 32),
        .space = Space::Cosine
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

TEST_F(ArrowDBTest, CreateOrGetCollection) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  auto result1 = db.getOrCreateCollection("test_collection", config);
  ASSERT_TRUE(result1.ok()) << result1.status().message();
  EXPECT_EQ(result1.value()->name(), "test_collection");

  auto result2 = db.getOrCreateCollection("test_collection", config);
  ASSERT_TRUE(result2.ok());
  EXPECT_EQ(result1.value(), result2.value());
}

TEST_F(ArrowDBTest, ZeroToSearchConvenienceAPI) {
  // "Zero to search in 5 lines" pattern
  Client db(testDir.string());                                              // 1
  auto* coll = db.getOrCreateCollection("docs", {.dimensions = 4}).value(); // 2
  auto id1 = coll->insert({0.1f, 0.2f, 0.3f, 0.4f});                      // 3 (auto-ID)
  ASSERT_TRUE(id1.ok()) << id1.status().message();
  EXPECT_FALSE(id1.value().empty());
  EXPECT_EQ(id1.value().size(), 36);  // UUID format

  auto status = coll->insert("my-id", {0.5f, 0.6f, 0.7f, 0.8f});          // 4 (explicit ID)
  ASSERT_TRUE(status.ok());

  auto hits = coll->search({0.1f, 0.2f, 0.3f, 0.4f}, 10);                 // 5
  EXPECT_EQ(hits.size(), 2);

  // Verify the auto-generated ID is retrievable
  auto vec = coll->get(id1.value());
  ASSERT_TRUE(vec.ok());
  EXPECT_EQ(vec.value().size(), 4);
}

TEST_F(ArrowDBTest, AutoIdInsertDimensionMismatch) {
  Client db(testDir.string());
  auto* coll = db.getOrCreateCollection("docs", {.dimensions = 4}).value();
  auto result = coll->insert({0.1f, 0.2f});  // wrong dimension
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kDimensionMismatch);
}

TEST_F(ArrowDBTest, PathConstructor) {
  Client db(testDir);  // filesystem::path overload
  EXPECT_EQ(db.dataDir(), testDir);
  EXPECT_TRUE(db.listCollections().empty());
}

TEST_F(ArrowDBTest, StringPathConstructor) {
  Client db(std::filesystem::path(testDir.string()));
  EXPECT_EQ(db.dataDir(), testDir);
  EXPECT_TRUE(db.listCollections().empty());
}

TEST_F(ArrowDBTest, CreateOrGetCollectionLoadsFromDisk) {
  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  {
    ClientOptions options{.dataDir = testDir};
    Client db(options);
    auto result = db.createCollection("test_collection", config);
    ASSERT_TRUE(result.ok());
    db.close();
  }

  {
    ClientOptions options{.dataDir = testDir};
    Client db(options);

    auto result = db.getOrCreateCollection("test_collection", config);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value()->name(), "test_collection");
    EXPECT_EQ(result.value()->dimension(), 128);
  }
}

TEST_F(ArrowDBTest, GetOrCreateFromDisk) {
  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  {
    ClientOptions options{.dataDir = testDir};
    Client db(options);
    auto result = db.createCollection("test_collection", config);
    ASSERT_TRUE(result.ok());

    Collection* collection = result.value();
    std::mt19937 gen(42);
    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      collection->insert(std::to_string(i), vec);
    }
    db.close();
  }

  {
    ClientOptions options{.dataDir = testDir};
    Client db(options);

    auto result = db.getOrCreateCollection("test_collection", config);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value()->size(), 10);
  }
}

TEST_F(ArrowDBTest, GetOrCreateDefaultDims) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  auto result = db.getOrCreateCollection("test_collection");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value()->dimension(), 384);
}

TEST_F(ArrowDBTest, DropPersistent) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  db.createCollection("test_collection", config);
  EXPECT_TRUE(db.hasCollection("test_collection"));

  auto status = db.dropCollection("test_collection");
  ASSERT_TRUE(status.ok());

  EXPECT_FALSE(db.hasCollection("test_collection"));
  EXPECT_FALSE(std::filesystem::exists(testDir / "test_collection"));
}

TEST_F(ArrowDBTest, ListCollectionsEmpty) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  auto names = db.listCollections();
  EXPECT_TRUE(names.empty());
}

TEST_F(ArrowDBTest, CloseClearsCollections) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  db.createCollection("test_collection", config);
  EXPECT_FALSE(db.listCollections().empty());

  auto status = db.close();
  ASSERT_TRUE(status.ok());

  EXPECT_TRUE(db.listCollections().empty());
}

TEST_F(ArrowDBTest, LoadExistingOnStartup) {
  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  {
    ClientOptions options{.dataDir = testDir};
    Client db(options);
    auto r1 = db.createCollection("test_collection", config);
    ASSERT_TRUE(r1.ok());
    std::mt19937 gen(42);
    r1.value()->insert("v1", RandomVector(128, gen));

    auto r2 = db.createCollection("test_collection2", config);
    ASSERT_TRUE(r2.ok());
    r2.value()->insert("v2", RandomVector(128, gen));

    db.close();
  }

  ClientOptions options{.dataDir = testDir};
  Client db(options);

  auto names = db.listCollections();
  EXPECT_EQ(names.size(), 2);
  EXPECT_TRUE(std::find(names.begin(), names.end(), "test_collection") != names.end());
  EXPECT_TRUE(std::find(names.begin(), names.end(), "test_collection2") != names.end());
}

TEST_F(ArrowDBTest, EmptyDataDir) {
  ClientOptions options{.dataDir = ""};
  Client db(options);

  CollectionConfig config{
      .name = "test_collection",
      .dimensions = 128,
      .space = Space::Cosine
  };

  auto result = db.createCollection("test_collection", config);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value()->size(), 0);
}

TEST_F(ArrowDBTest, ClientMoveAssignment) {
  ClientOptions options{.dataDir = testDir};
  Client db1(options);

  CollectionConfig config{
      .name = "move_assign",
      .dimensions = 128,
      .space = Space::Cosine
  };
  db1.createCollection("move_assign", config);

  Client db2(options);
  db2 = std::move(db1);
  EXPECT_TRUE(db2.hasCollection("move_assign"));
}
