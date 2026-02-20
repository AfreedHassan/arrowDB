#include "common.h"
#include "arrow/filter.h"
#include <gtest/gtest.h>

TEST_F(CollectionTest, SearchWithFilter) {
  CollectionConfig cfg{.name = "filter_search", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    std::string category = (i % 2 == 0) ? "even" : "odd";
    Metadata meta{{"category", category}, {"index", int64_t(i)}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto queryVec = RandomVector(4, gen);
  auto results = col.search(queryVec, 10, MetadataFilter::Eq("category", std::string("even")));
  EXPECT_GT(results.size(), 0u);
  for (const auto& r : results) {
    auto meta = col.getMetadata(r.id);
    ASSERT_TRUE(meta.ok());
    auto it = meta.value().find("category");
    ASSERT_NE(it, meta.value().end());
    EXPECT_EQ(std::get<std::string>(it->second), "even");
  }
}

TEST_F(CollectionTest, QueryWithFilter) {
  CollectionConfig cfg{.name = "filter_query", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    Metadata meta{{"score", double(i) / 50.0}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto queryVec = RandomVector(4, gen);
  auto result = col.query(queryVec, 10, MetadataFilter::Gte("score", 0.5));
  EXPECT_GT(result.hits.size(), 0u);
  for (const auto& hit : result.hits) {
    auto it = hit.metadata.find("score");
    ASSERT_NE(it, hit.metadata.end());
    EXPECT_GE(std::get<double>(it->second), 0.5);
  }
}

TEST_F(CollectionTest, QueryWithMetadataFilter) {
  CollectionConfig cfg{.name = "mf_query", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 30; ++i) {
    Metadata meta{{"val", int64_t(i)}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto queryVec = RandomVector(4, gen);
  MetadataFilter mf = [](const Metadata& m) {
    auto it = m.find("val");
    return it != m.end() && std::get<int64_t>(it->second) >= 20;
  };
  auto result = col.query(queryVec, 10, mf);
  EXPECT_GT(result.hits.size(), 0u);
  for (const auto& hit : result.hits) {
    auto it = hit.metadata.find("val");
    ASSERT_NE(it, hit.metadata.end());
    EXPECT_GE(std::get<int64_t>(it->second), 20);
  }
}

TEST_F(CollectionTest, FilterImplicitConversion) {
  CollectionConfig cfg{.name = "implicit", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 20; ++i) {
    Metadata meta{{"x", int64_t(i)}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto queryVec = RandomVector(4, gen);
  MetadataFilter mf = MetadataFilter::Gte("x", int64_t(10));
  auto results = col.search(queryVec, 5, mf);
  for (const auto& r : results) {
    auto meta = col.getMetadata(r.id);
    ASSERT_TRUE(meta.ok());
    EXPECT_GE(std::get<int64_t>(meta.value().at("x")), 10);
  }
}
