#include "common.h"
#include <gtest/gtest.h>

TEST_F(CollectionConcurrencyTest, ConcurrentInsertAndSearch) {
  CollectionConfig cfg{.name = "concurrent", .dimensions = 128, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 seedGen(42);
  for (int i = 0; i < 100; ++i) {
    auto vec = RandomVector(128, seedGen);
    auto s = col.insert("pre_" + std::to_string(i), vec);
    ASSERT_TRUE(s.ok());
  }

  constexpr int kInserters = 4;
  constexpr int kSearchers = 4;
  constexpr int kInsertsPerThread = 200;
  constexpr int kSearchesPerThread = 200;

  std::latch startLatch(kInserters + kSearchers);
  std::atomic<int> insertSuccesses{0};
  std::atomic<int> searchSuccesses{0};

  auto insertFn = [&](int threadId) {
    std::mt19937 gen(1000 + threadId);
    startLatch.arrive_and_wait();
    for (int i = 0; i < kInsertsPerThread; ++i) {
      std::string id = "t" + std::to_string(threadId) + "_" + std::to_string(i);
      auto vec = RandomVector(128, gen);
      auto s = col.insert(id, vec);
      if (s.ok()) insertSuccesses.fetch_add(1, std::memory_order_relaxed);
    }
  };

  auto searchFn = [&](int threadId) {
    std::mt19937 gen(2000 + threadId);
    startLatch.arrive_and_wait();
    for (int i = 0; i < kSearchesPerThread; ++i) {
      auto query = RandomVector(128, gen);
      auto results = col.search(query, 5, 50);
      if (!results.empty()) searchSuccesses.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < kInserters; ++i)
    threads.emplace_back(insertFn, i);
  for (int i = 0; i < kSearchers; ++i)
    threads.emplace_back(searchFn, i);

  for (auto& t : threads) t.join();

  EXPECT_EQ(insertSuccesses.load(), kInserters * kInsertsPerThread);
  EXPECT_GT(searchSuccesses.load(), 0);
  EXPECT_EQ(col.size(), 100 + kInserters * kInsertsPerThread);
}

TEST_F(CollectionConcurrencyTest, ConcurrentSearchBatchAndInsert) {
  CollectionConfig cfg{.name = "batch_concurrent", .dimensions = 128, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 seedGen(42);
  for (int i = 0; i < 200; ++i) {
    col.insert("pre_" + std::to_string(i), RandomVector(128, seedGen));
  }

  std::latch startLatch(2);
  std::atomic<bool> batchOk{false};

  auto insertFn = [&]() {
    std::mt19937 gen(3000);
    startLatch.arrive_and_wait();
    for (int i = 0; i < 200; ++i) {
      col.insert("ins_" + std::to_string(i), RandomVector(128, gen));
    }
  };

  auto searchBatchFn = [&]() {
    std::mt19937 gen(4000);
    startLatch.arrive_and_wait();
    for (int round = 0; round < 10; ++round) {
      std::vector<std::vector<float>> queries;
      for (int q = 0; q < 5; ++q)
        queries.push_back(RandomVector(128, gen));
      auto result = col.searchBatch(queries, 5, 50);
      if (result.ok()) batchOk.store(true, std::memory_order_relaxed);
    }
  };

  std::thread t1(insertFn);
  std::thread t2(searchBatchFn);
  t1.join();
  t2.join();

  EXPECT_TRUE(batchOk.load());
  EXPECT_EQ(col.size(), 400);
}

TEST_F(CollectionConcurrencyTest, ConcurrentSetMetadataAndSearch) {
  CollectionConfig cfg{.name = "meta_concurrent", .dimensions = 128, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 seedGen(42);
  for (int i = 0; i < 200; ++i) {
    Metadata meta{{"category", std::string(i % 2 == 0 ? "even" : "odd")}};
    col.insert(std::to_string(i), RandomVector(128, seedGen), std::move(meta));
  }

  std::latch startLatch(2);

  auto metaFn = [&]() {
    startLatch.arrive_and_wait();
    for (int i = 0; i < 200; ++i) {
      Metadata meta{{"category", std::string("updated_" + std::to_string(i))}};
      col.setMetadata(std::to_string(i), meta);
    }
  };

  auto searchFn = [&]() {
    std::mt19937 gen(5000);
    startLatch.arrive_and_wait();
    for (int i = 0; i < 100; ++i) {
      auto query = RandomVector(128, gen);
      auto results = col.search(query, 5,
        MetadataFilter::Where<std::string>("category", [](const std::string& s) {
          return !s.empty();
        }));
      (void)results;
    }
  };

  std::thread t1(metaFn);
  std::thread t2(searchFn);
  t1.join();
  t2.join();

  EXPECT_EQ(col.size(), 200);
}
