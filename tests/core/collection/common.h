#ifndef TESTS_CORE_COLLECTION_COMMON_H_
#define TESTS_CORE_COLLECTION_COMMON_H_

#include "arrow/collection.h"
#include "arrow/options.h"
#include "arrow/utils/uuid.h"
#include "test_util.h"
#include <filesystem>
#include <latch>
#include <thread>
#include <gtest/gtest.h>

using namespace arrow;
using namespace arrow::uuid;
using arrow::testing::RandomVector;

class CollectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_collection_test";
    std::filesystem::create_directories(testDir);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
  std::string GetTestPath(const std::string &dirname) {
    return (testDir / dirname).string();
  }
};

class CollectionWalTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_wal_test";
    std::filesystem::create_directories(testDir);
    gen.seed(42);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
  std::mt19937 gen;

  std::string GetTestPath(const std::string &dirname) {
    return (testDir / dirname).string();
  }

  std::string GetWalPath(const std::string &dirname) {
    return (testDir / dirname / "wal" / "db.wal").string();
  }

  CollectionConfig GetTestConfig(const std::string &name = "test_collection") {
    return CollectionConfig{.name = name, .dimensions = 128, .space = Space::Cosine};
  }
};

class CollectionBatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_batch_test";
    std::filesystem::create_directories(testDir);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
  std::string GetTestPath(const std::string &dirname) {
    return (testDir / dirname).string();
  }
};

class CollectionConcurrencyTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_concurrency_test";
    std::filesystem::create_directories(testDir);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
};

#endif  // TESTS_CORE_COLLECTION_COMMON_H_
