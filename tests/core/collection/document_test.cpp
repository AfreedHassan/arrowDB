#include "common.h"
#include <gtest/gtest.h>

TEST_F(CollectionTest, InsertDocument) {
  CollectionConfig cfg{.name = "doc_test", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  Document doc{
    .id = "doc1",
    .embedding = RandomVector(4, gen),
    .metadata = {{"key", std::string("value")}}
  };

  auto result = col.insert(doc);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), "doc1");
  EXPECT_EQ(col.size(), 1);

  auto meta = col.getMetadata("doc1");
  ASSERT_TRUE(meta.ok());
  EXPECT_EQ(std::get<std::string>(meta.value().at("key")), "value");
}

TEST_F(CollectionTest, InsertDocumentAutoID) {
  CollectionConfig cfg{.name = "doc_auto", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  Document doc{
    .id = "",
    .embedding = RandomVector(4, gen),
    .metadata = {}
  };

  auto result = col.insert(doc);
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result.value().empty());
  EXPECT_EQ(col.size(), 1);
}

TEST_F(CollectionTest, InsertBatchDocuments) {
  CollectionConfig cfg{.name = "doc_batch", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<Document> docs;
  for (int i = 0; i < 10; ++i) {
    docs.push_back({
      .id = "d" + std::to_string(i),
      .embedding = RandomVector(4, gen),
      .metadata = {{"idx", int64_t(i)}}
    });
  }

  auto result = col.insertBatch(std::move(docs));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, 10);
  EXPECT_EQ(result.value().failureCount, 0);
  EXPECT_EQ(col.size(), 10);

  auto meta = col.getMetadata("d5");
  ASSERT_TRUE(meta.ok());
  EXPECT_EQ(std::get<int64_t>(meta.value().at("idx")), 5);
}

TEST_F(CollectionTest, InsertBatchDocumentsSchemaValidation) {
  MetadataSchema schema;
  schema.field("tag", FieldType::String, true);

  CollectionConfig cfg{
    .name = "doc_batch_schema", .dimensions = 4, .space = Space::Cosine, .schema = schema
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<Document> docs;
  docs.push_back({
    .id = "d0",
    .embedding = RandomVector(4, gen),
    .metadata = {{"tag", std::string("ok")}}
  });
  docs.push_back({
    .id = "d1",
    .embedding = RandomVector(4, gen),
    .metadata = {}
  });
  docs.push_back({
    .id = "d2",
    .embedding = RandomVector(4, gen),
    .metadata = {{"tag", std::string("also ok")}}
  });

  auto result = col.insertBatch(std::move(docs));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, 2);
  EXPECT_EQ(result.value().failureCount, 1);
  EXPECT_FALSE(result.value().results[1].status.ok());
}

TEST_F(CollectionTest, OldBatchAPIStillWorks) {
  CollectionConfig cfg{.name = "old_batch", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  for (int i = 0; i < 5; ++i) {
    batch.push_back({std::to_string(i), RandomVector(4, gen)});
  }

  auto result = col.insertBatch(batch);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, 5);
  EXPECT_EQ(col.size(), 5);
}
