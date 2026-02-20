#include "common.h"
#include <thread>
#include <chrono>
#include <gtest/gtest.h>

using namespace arrow;
using arrow::testing::RandomVector;

// ── Compaction thread fires after threshold ops ─────────────

TEST_F(CollectionTest, CompactionThreadFires) {
  std::string path = GetTestPath("compaction_test");
  std::mt19937 gen(42);

  {
    CollectionConfig cfg{.name = "compaction", .dimensions = 4, .space = Space::Cosine};
    auto cr = Collection::create(cfg, path);
    ASSERT_TRUE(cr.ok()) << cr.status().message();
    Collection col = std::move(cr.value());

    // Insert enough vectors to trigger compaction (kCompactionOpsThreshold = 5000)
    for (int i = 0; i < 5100; ++i) {
      auto s = col.insert(std::to_string(i), RandomVector(4, gen));
      ASSERT_TRUE(s.ok()) << "Insert " << i << " failed: " << s.message();
    }

    // Give compaction thread time to fire
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // After compaction, data should still be searchable
    auto results = col.search(RandomVector(4, gen), 10);
    EXPECT_EQ(results.size(), 10u);

    // Check that WAL was truncated by examining file size
    auto walPath = std::filesystem::path(path) / "wal" / "db.wal";
    if (std::filesystem::exists(walPath)) {
      auto walSize = std::filesystem::file_size(walPath);
      // After truncation, WAL should be small (just header or empty)
      EXPECT_LT(walSize, 50000u) << "WAL should be truncated after compaction";
    }

    // Verify collection is still fully functional
    EXPECT_EQ(col.size(), 5100u);
  }  // col destructor releases file lock

  // Reload to verify persistence from compaction save
  auto loadResult = Collection::load(path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection loaded = std::move(loadResult.value());

  EXPECT_EQ(loaded.size(), 5100u);
  auto results2 = loaded.search(RandomVector(4, gen), 10);
  EXPECT_EQ(results2.size(), 10u);
}
