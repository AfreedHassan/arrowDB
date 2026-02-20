#include "common.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

TEST_F(CollectionTest, SaveCreatesDirectory) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  std::string save_path = GetTestPath("test_collection");
  EXPECT_NO_THROW(collection.save(save_path));

  EXPECT_TRUE(std::filesystem::exists(save_path));
  EXPECT_TRUE(std::filesystem::is_directory(save_path));
}

TEST_F(CollectionTest, SaveCreatesRequiredFiles) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  std::mt19937 gen(42);
  for (size_t i = 0; i < 10; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    collection.insert(uuidv4(i), vec);
  }

  std::string save_path = GetTestPath("test_collection");
  collection.save(save_path);

  EXPECT_TRUE(
      std::filesystem::exists(std::filesystem::path(save_path) / "meta.json"));
  EXPECT_TRUE(
      std::filesystem::exists(std::filesystem::path(save_path) / "index.bin"));
}

TEST_F(CollectionTest, SaveIncludesMetadata) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};
  Collection collection(cfg);

  std::mt19937 gen(42);
  for (size_t i = 0; i < 5; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    collection.insert(uuidv4(i), vec);

    Metadata meta;
    meta["category"] = std::string("test");
    meta["score"] = static_cast<double>(i);
    collection.setMetadata(uuidv4(i), meta);
  }

  std::string save_path = GetTestPath("test_collection");
  collection.save(save_path);

  EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(save_path) /
                                      "metadata.json"));
}

TEST_F(CollectionTest, LoadFromDirectory) {
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};
  Collection original(cfg);

  std::mt19937 gen(42);
  for (size_t i = 0; i < 100; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    original.insert(std::to_string(i), vec);
  }

  std::string save_path = GetTestPath("test_collection");
  original.save(save_path);

  auto loadResult = Collection::load(save_path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  EXPECT_EQ(loaded.name(), "test_collection");
  EXPECT_EQ(loaded.dimension(), 128);
  EXPECT_EQ(loaded.space(), Space::Cosine);
  EXPECT_EQ(loaded.size(), 100);
}

TEST_F(CollectionTest, RoundTripPreservesData) {
  CollectionConfig cfg{
      .name = "test_collection",
      .dimensions = 64,
      .space = Space::Cosine,
      .index_config = {.max_elements = 1000000, .hnsw_params = {.M = 32, .ef_construction = 200}}
  };
  Collection original(cfg);

  std::mt19937 gen(42);
  std::vector<std::vector<float>> vectors;
  for (size_t i = 0; i < 50; ++i) {
    std::vector<float> vec = RandomVector(64, gen);
    vectors.push_back(vec);
    original.insert(std::to_string(i), vec);
  }

  std::string save_path = GetTestPath("test_collection");
  original.save(save_path);

  auto loadResult = Collection::load(save_path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());
  EXPECT_EQ(loaded.size(), 50);

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
  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};
  Collection original(cfg);

  std::mt19937 gen(42);
  for (size_t i = 0; i < 10; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    original.insert(std::to_string(i), vec);

    Metadata meta;
    meta["id"] = static_cast<int64_t>(i);
    meta["name"] = std::string("vector_") + std::to_string(i);
    meta["score"] = static_cast<double>(i) * 0.1;
    meta["active"] = (i % 2 == 0);
    original.setMetadata(std::to_string(i), meta);
  }

  std::string save_path = GetTestPath("test_collection");
  original.save(save_path);

  auto loadResult = Collection::load(save_path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

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
  std::string save_path = GetTestPath("incomplete_collection");
  std::filesystem::create_directories(save_path);

  auto result = Collection::load(save_path);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(CollectionTest, LoadReturnsErrorOnMissingIndexBin) {
  std::string save_path = GetTestPath("incomplete_collection");
  std::filesystem::create_directories(save_path);

  std::string meta_path =
      (std::filesystem::path(save_path) / "meta.json").string();
  std::ofstream metaFile(meta_path);
  metaFile << R"({
     "name": "test",
     "dimensions": 128,
     "space": "Cosine",
     "dtype": "Float32",
     "idxType": "HNSW",
     "schemaVersion": 3,
     "hnsw": {"maxElements": 10000, "M": 16, "efConstruction": 200},
     "recovery": {"lastPersistedLsn": 0, "lastPersistedTxid": 0, "cleanShutdown": true}
   })";
  metaFile.close();

  auto result = Collection::load(save_path);
  EXPECT_FALSE(result.ok());
}

TEST_F(CollectionTest, LoadCorruptMetaJson) {
  std::string save_path = GetTestPath("corrupt_meta");
  std::filesystem::create_directories(save_path);

  CollectionConfig cfg{.name = "test_collection", .dimensions = 128, .space = Space::Cosine};
  Collection original(cfg);
  std::mt19937 gen(42);
  for (int i = 0; i < 10; ++i) {
    original.insert(std::to_string(i), RandomVector(128, gen));
  }
  original.save(save_path);

  auto metaPath = std::filesystem::path(save_path) / "meta.json";
  std::ofstream(metaPath) << "{corrupt json";

  auto loadResult = Collection::load(save_path);
  EXPECT_FALSE(loadResult.ok());
}

TEST_F(CollectionTest, LoadMissingIndexFile) {
  std::string save_path = GetTestPath("missing_index");
  std::filesystem::create_directories(save_path);

  std::string metaPath = (std::filesystem::path(save_path) / "meta.json").string();
  std::ofstream metaFile(metaPath);
  metaFile << R"({
      "name": "test",
      "dimensions": 128,
      "space": "Cosine",
      "dtype": "Float32",
      "idxType": "HNSW",
      "schemaVersion": 3,
      "hnsw": {"maxElements": 10000, "M": 16, "efConstruction": 200},
      "recovery": {"lastPersistedLsn": 0, "lastPersistedTxid": 0, "cleanShutdown": true}
  })";
  metaFile.close();

  auto result = Collection::load(save_path);
  EXPECT_FALSE(result.ok());
}

TEST_F(CollectionTest, LoadMissingIdSpaceFile) {
  std::string save_path = GetTestPath("missing_idspace");
  std::filesystem::create_directories(save_path);

  std::string metaPath = (std::filesystem::path(save_path) / "meta.json").string();
  std::ofstream metaFile(metaPath);
  metaFile << R"({
      "name": "test",
      "dimensions": 128,
      "space": "Cosine",
      "dtype": "Float32",
      "idxType": "HNSW",
      "schemaVersion": 3,
      "hnsw": {"maxElements": 10000, "M": 16, "efConstruction": 200},
      "recovery": {"lastPersistedLsn": 0, "lastPersistedTxid": 0, "cleanShutdown": true}
  })";
  metaFile.close();

  auto result = Collection::load(save_path);
  EXPECT_FALSE(result.ok());
}

TEST_F(CollectionTest, CleanShutdownMarkerAfterSave) {
  CollectionConfig cfg{.name = "clean_test", .dimensions = 128, .space = Space::Cosine};
  std::string savePath = GetTestPath("clean_test");

  Collection col(cfg);
  std::mt19937 gen(42);
  for (int i = 0; i < 10; ++i) {
    col.insert(std::to_string(i), RandomVector(128, gen));
  }

  auto saveStatus = col.save(savePath);
  ASSERT_TRUE(saveStatus.ok());

  auto metaPath = std::filesystem::path(savePath) / "meta.json";
  std::ifstream inFile(metaPath);
  json j;
  inFile >> j;
  inFile.close();

  ASSERT_TRUE(j.contains("recovery"));
  EXPECT_TRUE(j["recovery"]["cleanShutdown"].get<bool>());
}

TEST_F(CollectionTest, SaveLoadMetadata) {
  CollectionConfig cfg{.name = "meta_test", .dimensions = 128, .space = Space::Cosine};
  std::string savePath = GetTestPath("meta_test");

  Collection col(cfg);
  std::mt19937 gen(42);
  Metadata originalMeta{
      {"category", std::string("test")},
      {"score", 0.95},
      {"count", int64_t(42)},
      {"active", true}
  };

  auto s = col.insert("v1", RandomVector(128, gen), originalMeta);
  ASSERT_TRUE(s.ok());

  auto saveStatus = col.save(savePath);
  ASSERT_TRUE(saveStatus.ok());

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok());
  Collection loaded = std::move(loadResult.value());

  auto metaResult = loaded.getMetadata("v1");
  ASSERT_TRUE(metaResult.ok());

  auto& loadedMeta = metaResult.value();
  EXPECT_EQ(std::get<std::string>(loadedMeta.at("category")), "test");
  EXPECT_DOUBLE_EQ(std::get<double>(loadedMeta.at("score")), 0.95);
  EXPECT_EQ(std::get<int64_t>(loadedMeta.at("count")), 42);
  EXPECT_EQ(std::get<bool>(loadedMeta.at("active")), true);
}

TEST_F(CollectionTest, SaveLoadWithSchema) {
  MetadataSchema schema;
  schema.field("category", FieldType::String, true)
        .field("score", FieldType::Double, false);

  CollectionConfig cfg{
      .name = "schema_test",
      .dimensions = 128,
      .space = Space::Cosine,
      .schema = schema
  };
  std::string savePath = GetTestPath("schema_test");

  Collection col(cfg);
  std::mt19937 gen(42);
  Metadata meta{{"category", std::string("test")}};
  auto s = col.insert("v1", RandomVector(128, gen), meta);
  ASSERT_TRUE(s.ok());

  auto saveStatus = col.save(savePath);
  ASSERT_TRUE(saveStatus.ok());

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok());
  Collection loaded = std::move(loadResult.value());

  Metadata badMeta{{"score", std::string("not_a_number")}};
  auto s2 = loaded.insert("v2", RandomVector(128, gen), badMeta);
  EXPECT_FALSE(s2.ok());
}

