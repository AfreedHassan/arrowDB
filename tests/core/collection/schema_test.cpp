#include "common.h"
#include <gtest/gtest.h>

TEST_F(CollectionTest, EmptySchemaAcceptsAnything) {
  CollectionConfig cfg{.name = "schema_empty", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  Metadata meta{{"foo", std::string("bar")}, {"num", int64_t(42)}};
  auto s = col.insert("v1", RandomVector(4, gen), meta);
  EXPECT_TRUE(s.ok());
}

TEST_F(CollectionTest, RequiredFieldPresent) {
  MetadataSchema schema;
  schema.field("category", FieldType::String, true);

  CollectionConfig cfg{
    .name = "schema_req", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  Metadata meta{{"category", std::string("image")}};
  auto s = col.insert("v1", RandomVector(4, gen), meta);
  EXPECT_TRUE(s.ok()) << s.message();
}

TEST_F(CollectionTest, RequiredFieldMissing) {
  MetadataSchema schema;
  schema.field("category", FieldType::String, true);

  CollectionConfig cfg{
    .name = "schema_miss", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  auto s = col.insert("v1", RandomVector(4, gen), {});
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, WrongFieldType) {
  MetadataSchema schema;
  schema.field("count", FieldType::Int64, false);

  CollectionConfig cfg{
    .name = "schema_type", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  Metadata meta{{"count", std::string("not_a_number")}};
  auto s = col.insert("v1", RandomVector(4, gen), meta);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, OptionalFieldMissing) {
  MetadataSchema schema;
  schema.field("label", FieldType::String, false);

  CollectionConfig cfg{
    .name = "schema_opt", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  auto s = col.insert("v1", RandomVector(4, gen), {});
  EXPECT_TRUE(s.ok());
}

TEST_F(CollectionTest, ExtraFieldsAllowed) {
  MetadataSchema schema;
  schema.field("category", FieldType::String, true);

  CollectionConfig cfg{
    .name = "schema_extra", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  Metadata meta{{"category", std::string("img")}, {"undeclared", int64_t(99)}};
  auto s = col.insert("v1", RandomVector(4, gen), meta);
  EXPECT_TRUE(s.ok());
}

TEST_F(CollectionTest, SchemaValidationOnUpdate) {
  MetadataSchema schema;
  schema.field("score", FieldType::Double, true);

  CollectionConfig cfg{
    .name = "schema_update", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  auto vec = RandomVector(4, gen);
  Metadata good{{"score", double(0.5)}};
  auto s = col.insert("v1", vec, good);
  ASSERT_TRUE(s.ok());

  auto s2 = col.update("v1", vec, {});
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, SchemaValidationOnSetMetadata) {
  MetadataSchema schema;
  schema.field("tag", FieldType::String, true);

  CollectionConfig cfg{
    .name = "schema_setmeta", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  Metadata good{{"tag", std::string("a")}};
  auto s = col.insert("v1", RandomVector(4, gen), good);
  ASSERT_TRUE(s.ok());

  Metadata bad{{"tag", int64_t(123)}};
  auto s2 = col.setMetadata("v1", bad);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(CollectionTest, SchemaPersistsAcrossReload) {
  MetadataSchema schema;
  schema.field("label", FieldType::String, true)
        .field("score", FieldType::Double, false);

  CollectionConfig cfg{
    .name = "schema_persist", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };

  std::string savePath = GetTestPath("schema_persist");
  {
    Collection col(cfg);
    std::mt19937 gen(42);
    Metadata meta{{"label", std::string("cat")}};
    auto s = col.insert("v1", RandomVector(4, gen), meta);
    ASSERT_TRUE(s.ok());
    auto saveStatus = col.save(savePath);
    ASSERT_TRUE(saveStatus.ok());
  }

  auto loadResult = Collection::load(savePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  std::mt19937 gen2(99);
  Metadata meta{{"label", std::string("dog")}};
  auto s = loaded.insert("v2", RandomVector(4, gen2), meta);
  EXPECT_TRUE(s.ok());

  auto s2 = loaded.insert("v3", RandomVector(4, gen2), {});
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.code(), utils::StatusCode::kInvalidArgument);
}
