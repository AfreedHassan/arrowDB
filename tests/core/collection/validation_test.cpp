#include "common.h"
#include <limits>
#include <gtest/gtest.h>

TEST_F(CollectionTest, InsertNaNVector) {
  CollectionConfig cfg{.name = "nan_test", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec = {1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(), 4.0f};
  auto s = col.insert("v1", vec);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, InsertInfVector) {
  CollectionConfig cfg{.name = "inf_test", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec = {1.0f, std::numeric_limits<float>::infinity(), 3.0f, 4.0f};
  auto s = col.insert("v1", vec);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, InsertNegInfVector) {
  CollectionConfig cfg{.name = "neginf_test", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec = {1.0f, 2.0f, -std::numeric_limits<float>::infinity(), 4.0f};
  auto s = col.insert("v1", vec);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, UpdateNaNVector) {
  CollectionConfig cfg{.name = "update_nan", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  auto goodVec = RandomVector(4, gen);
  auto s = col.insert("v1", goodVec);
  ASSERT_TRUE(s.ok());

  std::vector<float> nanVec = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
  auto s2 = col.update("v1", nanVec);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, BatchInsertWithNaN) {
  CollectionConfig cfg{.name = "batch_nan", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  batch.push_back({"good1", RandomVector(4, gen)});
  batch.push_back({"bad", {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f}});
  batch.push_back({"good2", RandomVector(4, gen)});

  auto result = col.insertBatch(batch);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, 2);
  EXPECT_EQ(result.value().failureCount, 1);
  EXPECT_TRUE(result.value().results[0].status.ok());
  EXPECT_FALSE(result.value().results[1].status.ok());
  EXPECT_EQ(result.value().results[1].status.code(), utils::StatusCode::kInvalidArgument);
  EXPECT_TRUE(result.value().results[2].status.ok());
}

TEST_F(CollectionTest, TooManyMetadataKeys) {
  CollectionConfig cfg{.name = "meta_keys", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  Metadata meta;
  for (int i = 0; i < 257; ++i) {
    meta["key_" + std::to_string(i)] = int64_t(i);
  }
  auto s = col.insert("v1", RandomVector(4, gen), std::move(meta));
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, MetadataValueTooLarge) {
  CollectionConfig cfg{.name = "meta_large", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  std::string largeValue(65 * 1024, 'x');
  Metadata meta{{"big", largeValue}};
  auto s = col.insert("v1", RandomVector(4, gen), std::move(meta));
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, MetadataKeyTooLong) {
  CollectionConfig cfg{.name = "meta_longkey", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  std::string longKey(257, 'k');
  Metadata meta{{longKey, std::string("value")}};
  auto s = col.insert("v1", RandomVector(4, gen), std::move(meta));
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, SetMetadataTooManyKeys) {
    CollectionConfig cfg{.name = "setmeta_keys", .dimensions = 4, .space = Space::Cosine};
    Collection col(cfg);

    std::mt19937 gen(42);
    auto s = col.insert("v1", RandomVector(4, gen));
    ASSERT_TRUE(s.ok());

    Metadata meta;
    for (int i = 0; i < 257; ++i) {
        meta["key_" + std::to_string(i)] = int64_t(i);
    }
    auto s2 = col.setMetadata("v1", meta);
    EXPECT_FALSE(s2.ok());
    EXPECT_EQ(s2.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, InsertIDTooLong) {
    CollectionConfig cfg{.name = "id_test", .dimensions = 4, .space = Space::Cosine};
    Collection col(cfg);

    std::mt19937 gen(42);
    std::string longID(128, 'x');
    auto s = col.insert(longID, RandomVector(4, gen));
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, RemoveIDTooLong) {
    CollectionConfig cfg{.name = "id_test", .dimensions = 4, .space = Space::Cosine};
    Collection col(cfg);

    std::string longID(128, 'x');
    auto s = col.remove(longID);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, InsertDimensionMismatch) {
    CollectionConfig cfg{.name = "dim_test", .dimensions = 4, .space = Space::Cosine};
    Collection col(cfg);

    std::mt19937 gen(42);
    std::vector<float> wrongDimVec = RandomVector(128, gen);
    auto s = col.insert("v1", wrongDimVec);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), utils::StatusCode::kDimensionMismatch);
}
