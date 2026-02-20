#include "common.h"
#include <gtest/gtest.h>

TEST_F(CollectionTest, CreateCollection) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};

  Collection collection(cfg);

  EXPECT_EQ(collection.name(), "test_collection");
  EXPECT_EQ(collection.dimension(), 128);
  EXPECT_EQ(collection.space(), Space::Cosine);
  EXPECT_EQ(collection.size(), 0);
}

TEST_F(CollectionTest, InsertVectors) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(std::to_string(i), vec);
  }

  EXPECT_EQ(collection.size(), num_vectors);
}

TEST_F(CollectionTest, SearchFunctionality) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(std::to_string(i), vec);
  }

  std::vector<float> query = RandomVector(dim, gen);
  const size_t k = 10;
  std::vector<IndexSearchResult> results = collection.search(query, k);

  EXPECT_EQ(results.size(), k);

  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i].score, results[i - 1].score)
        << "Results should be sorted in descending order";
  }
}

TEST_F(CollectionTest, SearchWithDifferentEf) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(std::to_string(i), vec);
  }

  std::vector<float> query = RandomVector(dim, gen);
  const size_t k = 10;

  for (size_t ef : {10, 50, 100}) {
    std::vector<IndexSearchResult> results = collection.search(query, k, ef);
    EXPECT_EQ(results.size(), k) << "ef=" << ef;
  }
}

TEST_F(CollectionTest, UpdateNonexistent) {
  CollectionConfig cfg{.name = "update_nonexist", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  auto s = col.update("missing", RandomVector(4, gen));
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionTest, RemoveNonexistent) {
  CollectionConfig cfg{.name = "remove_nonexist", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  auto s = col.remove("missing");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionTest, GetExisting) {
  CollectionConfig cfg{.name = "get_existing", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
  auto s = col.insert("v1", vec);
  ASSERT_TRUE(s.ok());

  auto result = col.get("v1");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().size(), 4);
}

TEST_F(CollectionTest, GetNonexistent) {
  CollectionConfig cfg{.name = "get_nonexist", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  auto result = col.get("missing");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionTest, UpdateSuccess) {
  CollectionConfig cfg{.name = "update_success", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
  auto s = col.insert("v1", vec1);
  ASSERT_TRUE(s.ok());

  std::vector<float> vec2 = {5.0f, 6.0f, 7.0f, 8.0f};
  auto s2 = col.update("v1", vec2);
  EXPECT_TRUE(s2.ok());

  auto result = col.get("v1");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), vec2);
}

TEST_F(CollectionTest, UpsertExisting) {
  CollectionConfig cfg{.name = "upsert_existing", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
  col.insert("v1", vec1);

  std::vector<float> vec2 = {5.0f, 6.0f, 7.0f, 8.0f};
  auto s = col.upsert("v1", vec2);
  EXPECT_TRUE(s.ok());

  auto result = col.get("v1");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), vec2);
  EXPECT_EQ(col.size(), 1);
}

TEST_F(CollectionTest, UpsertNew) {
  CollectionConfig cfg{.name = "upsert_new", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
  auto s = col.upsert("v1", vec);
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(col.size(), 1);

  auto result = col.get("v1");
  ASSERT_TRUE(result.ok());
}

TEST_F(CollectionTest, RemoveSuccess) {
  CollectionConfig cfg{.name = "remove_success", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  auto s = col.insert("v1", RandomVector(4, gen));
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(col.size(), 1);

  auto s2 = col.remove("v1");
  EXPECT_TRUE(s2.ok());

  // After remove, get should return not found
  auto result = col.get("v1");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionTest, SearchEmptyCollection) {
  CollectionConfig cfg{.name = "search_empty", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
  auto results = col.search(query, 5);
  EXPECT_TRUE(results.empty());
}

TEST_F(CollectionTest, InsertAutoGeneratedID) {
  CollectionConfig cfg{.name = "auto_id", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
  auto result = col.insert(vec);
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result.value().empty());
  EXPECT_EQ(result.value().size(), 36);  // UUID format
  EXPECT_EQ(col.size(), 1);
}

TEST_F(CollectionTest, StatsBasic) {
  CollectionConfig cfg{.name = "stats_test", .dimensions = 128, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 10; ++i) {
    col.insert(std::to_string(i), RandomVector(128, gen));
  }

  auto s = col.stats();
  EXPECT_EQ(s.vectorCount, 10);
  EXPECT_EQ(s.dimensions, 128);
}

TEST_F(CollectionTest, CloseInMemory) {
  CollectionConfig cfg{.name = "close_test", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  auto s = col.close();
  EXPECT_TRUE(s.ok());
}

TEST_F(CollectionTest, SearchPerformance) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};

  Collection collection(cfg);

  std::mt19937 gen(42);
  const size_t num_vectors = 1000;
  const size_t dim = collection.dimension();

  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec = RandomVector(dim, gen);
    collection.insert(std::to_string(i), vec);
  }

  std::vector<float> query = RandomVector(dim, gen);
  const size_t k = 10;

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<IndexSearchResult> results = collection.search(query, k, 100);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  EXPECT_EQ(results.size(), k);
  EXPECT_LT(duration.count(), 1000)
      << "Search took " << duration.count() << " microseconds";
}
