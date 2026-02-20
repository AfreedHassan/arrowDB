#include "common.h"
#include <gtest/gtest.h>

TEST_F(CollectionTest, PreparedFilterZeroSelectivity) {
  CollectionConfig cfg{.name = "pf_zero", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    Metadata meta{{"val", int64_t(i)}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto pf = col.prepareFilter([](const Metadata&) { return false; });
  auto results = col.search(RandomVector(4, gen), 10, pf);
  EXPECT_TRUE(results.empty());
}

TEST_F(CollectionTest, PreparedFilterFullSelectivity) {
  CollectionConfig cfg{.name = "pf_full", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    col.insert(std::to_string(i), RandomVector(4, gen));
  }

  auto query = RandomVector(4, gen);
  auto pf = col.prepareFilter([](const Metadata&) { return true; });
  auto filtered = col.search(query, 10, pf);
  auto unfiltered = col.search(query, 10);

  ASSERT_EQ(filtered.size(), unfiltered.size());
  for (size_t i = 0; i < filtered.size(); ++i) {
    EXPECT_EQ(filtered[i].id, unfiltered[i].id);
    EXPECT_FLOAT_EQ(filtered[i].score, unfiltered[i].score);
  }
}

TEST_F(CollectionTest, PreparedFilterWithDeletedVectors) {
  CollectionConfig cfg{.name = "pf_del", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 30; ++i) {
    Metadata meta{{"group", std::string(i % 2 == 0 ? "even" : "odd")}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  for (int i = 0; i < 10; i += 2) {
    auto s = col.remove(std::to_string(i));
    ASSERT_TRUE(s.ok());
  }

  auto pf = col.prepareFilter(MetadataFilter::Eq("group", std::string("even")));
  auto results = col.search(RandomVector(4, gen), 20, pf);

  for (const auto& r : results) {
    int id = std::stoi(r.id);
    EXPECT_TRUE(id >= 10 || id % 2 != 0) << "Deleted vector " << id << " appeared in results";
    auto meta = col.getMetadata(r.id);
    ASSERT_TRUE(meta.ok());
    EXPECT_EQ(std::get<std::string>(meta.value().at("group")), "even");
  }
}

TEST_F(CollectionTest, PreparedFilterNoMetadata) {
  CollectionConfig cfg{.name = "pf_nometa", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(55);
  for (int i = 0; i < 20; ++i) {
    col.insert(std::to_string(i), RandomVector(4, gen));
  }
  for (int i = 20; i < 25; ++i) {
    Metadata meta{{"tag", std::string("special")}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto pf = col.prepareFilter(
      [](const Metadata& m) { return m.find("tag") == m.end(); });
  auto results = col.search(RandomVector(4, gen), 10, pf);

  EXPECT_EQ(results.size(), 10u);
  for (const auto& r : results) {
    int id = std::stoi(r.id);
    EXPECT_LT(id, 20) << "Vector with metadata should not pass filter";
  }
}

TEST_F(CollectionTest, PreparedFilterQueryMethod) {
  CollectionConfig cfg{.name = "pf_query", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    Metadata meta{{"score", double(i) / 50.0}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto pf = col.prepareFilter(MetadataFilter::Gte("score", 0.5));
  auto result = col.query(RandomVector(4, gen), 10, pf);
  EXPECT_GT(result.hits.size(), 0u);
  for (const auto& hit : result.hits) {
    auto it = hit.metadata.find("score");
    ASSERT_NE(it, hit.metadata.end());
    EXPECT_GE(std::get<double>(it->second), 0.5);
  }
}

TEST_F(CollectionTest, PreparedFilterMatchesCallbackFilter) {
  CollectionConfig cfg{.name = "pf_compare", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 100; ++i) {
    Metadata meta{{"category", std::string(i % 2 == 0 ? "even" : "odd")}};
    col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
  }

  auto query = RandomVector(4, gen);
  MetadataFilter filter = MetadataFilter::Eq("category", std::string("even"));

  auto callbackResults = col.search(query, 10, filter);
  auto pf = col.prepareFilter(filter);
  auto bitmapResults = col.search(query, 10, pf);

  ASSERT_EQ(callbackResults.size(), bitmapResults.size());
  for (size_t i = 0; i < callbackResults.size(); ++i) {
    EXPECT_EQ(callbackResults[i].id, bitmapResults[i].id);
    EXPECT_FLOAT_EQ(callbackResults[i].score, bitmapResults[i].score);
  }
}
