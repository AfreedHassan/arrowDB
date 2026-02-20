#include "common.h"
#include <gtest/gtest.h>

TEST_F(CollectionTest, PreFilterReturnsK) {
  CollectionConfig cfg{.name = "prefilter", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 100; ++i) {
    Metadata meta;
    if (i % 2 == 0) meta["even"] = true;
    auto s = col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
    ASSERT_TRUE(s.ok());
  }

  auto results = col.search(
      RandomVector(4, gen), 10,
      [](const Metadata& m) {
        auto it = m.find("even");
        return it != m.end() && std::get<bool>(it->second);
      });

  EXPECT_EQ(results.size(), 10u);
}

TEST_F(CollectionTest, PreFilterHighSelectivity) {
  CollectionConfig cfg{.name = "prefilter_high", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(123);
  for (int i = 0; i < 200; ++i) {
    Metadata meta;
    if (i < 5) meta["rare"] = true;
    auto s = col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
    ASSERT_TRUE(s.ok());
  }

  auto results = col.search(
      RandomVector(4, gen), 10,
      [](const Metadata& m) {
        auto it = m.find("rare");
        return it != m.end() && std::get<bool>(it->second);
      });

  EXPECT_EQ(results.size(), 5u);
}

TEST_F(CollectionTest, PreFilterNoMatches) {
  CollectionConfig cfg{.name = "prefilter_none", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(99);
  for (int i = 0; i < 50; ++i) {
    auto s = col.insert(std::to_string(i), RandomVector(4, gen));
    ASSERT_TRUE(s.ok());
  }

  auto results = col.search(
      RandomVector(4, gen), 10,
      [](const Metadata&) { return false; });

  EXPECT_TRUE(results.empty());
}

TEST_F(CollectionTest, PreFilterMatchAll) {
  CollectionConfig cfg{.name = "prefilter_all", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(77);
  for (int i = 0; i < 50; ++i) {
    auto s = col.insert(std::to_string(i), RandomVector(4, gen));
    ASSERT_TRUE(s.ok());
  }

  auto query = RandomVector(4, gen);
  auto filtered = col.search(query, 10, [](const Metadata&) { return true; });
  auto unfiltered = col.search(query, 10);

  ASSERT_EQ(filtered.size(), unfiltered.size());
  for (size_t i = 0; i < filtered.size(); ++i) {
    EXPECT_EQ(filtered[i].id, unfiltered[i].id);
    EXPECT_FLOAT_EQ(filtered[i].score, unfiltered[i].score);
  }
}

TEST_F(CollectionTest, PreFilterMissingMetadata) {
  CollectionConfig cfg{.name = "prefilter_missing", .dimensions = 4, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(55);
  for (int i = 0; i < 20; ++i) {
    auto s = col.insert(std::to_string(i), RandomVector(4, gen));
    ASSERT_TRUE(s.ok());
  }
  for (int i = 20; i < 25; ++i) {
    Metadata meta{{"tag", std::string("special")}};
    auto s = col.insert(std::to_string(i), RandomVector(4, gen), std::move(meta));
    ASSERT_TRUE(s.ok());
  }

  auto results = col.search(
      RandomVector(4, gen), 10,
      [](const Metadata& m) { return m.find("tag") == m.end(); });

  EXPECT_EQ(results.size(), 10u);
}
