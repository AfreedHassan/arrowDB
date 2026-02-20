#include "common.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
using nlohmann::json;

TEST_F(CollectionTest, DefaultConfigValues) {
  IndexConfig idx;
  EXPECT_EQ(idx.index_type, IndexType::HNSW);
  EXPECT_EQ(idx.max_elements, 1000000u);
  EXPECT_EQ(idx.quantization, Quantization::None);
  EXPECT_EQ(idx.hnsw_params.M, 16u);
  EXPECT_EQ(idx.hnsw_params.ef_construction, 200u);
  EXPECT_EQ(idx.hnsw_params.ef_search, 200u);
}

TEST_F(CollectionTest, QuantizationEnumValues) {
  EXPECT_EQ(static_cast<uint8_t>(Quantization::None), 0);
  EXPECT_EQ(static_cast<uint8_t>(Quantization::INT8), 1);
  EXPECT_EQ(static_cast<uint8_t>(IndexType::HNSW), 0);
}

TEST_F(CollectionTest, CreateWithQuantizationINT8) {
  CollectionConfig cfg{
    .name = "quant_test",
    .dimensions = 32,
    .space = Space::Cosine,
    .index_config = {.quantization = Quantization::INT8, .hnsw_params = {.M = 16}}
  };
  Collection col(cfg);
  EXPECT_EQ(col.name(), "quant_test");
  EXPECT_EQ(col.dimension(), 32);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    auto s = col.insert(std::to_string(i), RandomVector(32, gen));
    ASSERT_TRUE(s.ok());
  }
  EXPECT_EQ(col.size(), 50);

  auto results = col.search(RandomVector(32, gen), 5);
  EXPECT_EQ(results.size(), 5);
}

TEST_F(CollectionTest, CreateWithCustomHNSWParams) {
  CollectionConfig cfg{
    .name = "custom_hnsw",
    .dimensions = 64,
    .space = Space::L2,
    .index_config = {
      .max_elements = 5000,
      .hnsw_params = {.M = 32, .ef_construction = 400, .ef_search = 300}
    }
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 100; ++i) {
    auto s = col.insert(std::to_string(i), RandomVector(64, gen));
    ASSERT_TRUE(s.ok());
  }
  EXPECT_EQ(col.size(), 100);
}

TEST_F(CollectionTest, OptimizeNoOpWhenQuantizationDisabled) {
  CollectionConfig cfg{
    .name = "opt_noop",
    .dimensions = 32,
    .space = Space::Cosine,
    .index_config = {.quantization = Quantization::None}
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 20; ++i) {
    col.insert(std::to_string(i), RandomVector(32, gen));
  }

  auto status = col.optimize();
  EXPECT_TRUE(status.ok());
}

TEST_F(CollectionTest, OptimizeWithQuantization) {
  CollectionConfig cfg{
    .name = "opt_sq",
    .dimensions = 32,
    .space = Space::L2,
    .index_config = {.quantization = Quantization::INT8, .hnsw_params = {.M = 8}}
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<std::vector<float>> vecs;
  for (int i = 0; i < 100; ++i) {
    auto v = RandomVector(32, gen);
    vecs.push_back(v);
    auto s = col.insert(std::to_string(i), v);
    ASSERT_TRUE(s.ok());
  }

  auto status = col.optimize();
  EXPECT_TRUE(status.ok());

  auto status2 = col.optimize();
  EXPECT_TRUE(status2.ok());

  auto results = col.search(vecs[0], 5);
  EXPECT_EQ(results.size(), 5);
  EXPECT_EQ(results[0].id, "0");
}

TEST_F(CollectionTest, OptimizeEmptyCollectionIsNoOp) {
  CollectionConfig cfg{
    .name = "opt_empty",
    .dimensions = 32,
    .space = Space::L2,
    .index_config = {.quantization = Quantization::INT8}
  };
  Collection col(cfg);

  auto status = col.optimize();
  EXPECT_TRUE(status.ok());
}

TEST_F(CollectionTest, ConfigRoundTripWithQuantization) {
  CollectionConfig cfg{
    .name = "roundtrip_quant",
    .dimensions = 32,
    .space = Space::L2,
    .index_config = {.quantization = Quantization::INT8, .hnsw_params = {.M = 16}}
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<std::vector<float>> vecs;
  for (int i = 0; i < 50; ++i) {
    auto v = RandomVector(32, gen);
    vecs.push_back(v);
    col.insert(std::to_string(i), v);
  }

  std::string savePath = GetTestPath("roundtrip_quant");
  auto saveStatus = col.save(savePath);
  ASSERT_TRUE(saveStatus.ok());

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  EXPECT_EQ(loaded.name(), "roundtrip_quant");
  EXPECT_EQ(loaded.dimension(), 32);
  EXPECT_EQ(loaded.size(), 50);

  auto results = loaded.search(vecs[0], 5);
  EXPECT_EQ(results.size(), 5);
}

TEST_F(CollectionTest, AutoOptimizeOnLoad) {
  CollectionConfig cfg{
    .name = "auto_opt",
    .dimensions = 32,
    .space = Space::L2,
    .index_config = {.quantization = Quantization::INT8, .hnsw_params = {.M = 8}}
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<std::vector<float>> vecs;
  for (int i = 0; i < 50; ++i) {
    auto v = RandomVector(32, gen);
    vecs.push_back(v);
    col.insert(std::to_string(i), v);
  }

  std::string savePath = GetTestPath("auto_opt");
  auto saveStatus = col.save(savePath);
  ASSERT_TRUE(saveStatus.ok());

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  EXPECT_EQ(loaded.size(), 50);

  auto results = loaded.search(vecs[0], 5);
  EXPECT_EQ(results.size(), 5);
  EXPECT_EQ(results[0].id, "0");
}

TEST_F(CollectionTest, OptimizedIndexPersistsSQMode) {
  CollectionConfig cfg{
    .name = "persist_sq_mode",
    .dimensions = 32,
    .space = Space::L2,
    .index_config = {.quantization = Quantization::INT8, .hnsw_params = {.M = 8}}
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<std::vector<float>> vecs;
  for (int i = 0; i < 50; ++i) {
    auto v = RandomVector(32, gen);
    vecs.push_back(v);
    col.insert(std::to_string(i), v);
  }

  auto optStatus = col.optimize();
  ASSERT_TRUE(optStatus.ok());

  std::string savePath = GetTestPath("persist_sq_mode");
  auto saveStatus = col.save(savePath);
  ASSERT_TRUE(saveStatus.ok());

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  EXPECT_EQ(loaded.size(), 50);

  auto results = loaded.search(vecs[0], 5);
  EXPECT_EQ(results.size(), 5);
  EXPECT_EQ(results[0].id, "0");
}

TEST_F(CollectionTest, BackwardCompatOldQuantizeBoolInMetaJson) {
  std::string savePath = GetTestPath("backward_compat");
  std::filesystem::create_directories(savePath);

  CollectionConfig cfg{
    .name = "backward_compat",
    .dimensions = 32,
    .space = Space::L2,
    .index_config = {.quantization = Quantization::INT8, .hnsw_params = {.M = 8}}
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 20; ++i) {
    col.insert(std::to_string(i), RandomVector(32, gen));
  }

  col.save(savePath);

  auto metaPath = std::filesystem::path(savePath) / "meta.json";
  std::ifstream inFile(metaPath);
  json j;
  inFile >> j;
  inFile.close();

  ASSERT_TRUE(j.contains("hnsw"));
  EXPECT_TRUE(j["hnsw"].contains("quantize"));
  EXPECT_TRUE(j["hnsw"]["quantize"].get<bool>());

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());
  EXPECT_EQ(loaded.size(), 20);
}

TEST_F(CollectionTest, NonQuantizedCollectionSkipsOptimizeOnLoad) {
  CollectionConfig cfg{
    .name = "no_quant",
    .dimensions = 32,
    .space = Space::L2,
    .index_config = {.quantization = Quantization::None}
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 20; ++i) {
    col.insert(std::to_string(i), RandomVector(32, gen));
  }

  std::string savePath = GetTestPath("no_quant");
  col.save(savePath);

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());
  EXPECT_EQ(loaded.size(), 20);
}
