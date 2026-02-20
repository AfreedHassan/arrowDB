#include <gtest/gtest.h>
#include "index/hnsw_index.h"
#include "test_util.h"
#include <filesystem>

using namespace arrow;
using arrow::testing::RandomVector;

class HNSWCoverageTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = std::filesystem::temp_directory_path() / "arrow_hnsw_coverage";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    std::filesystem::path testDir;
    std::string GetTestPath(const std::string& filename) {
        return (testDir / filename).string();
    }
};

// ── updatePoint (duplicate label insert triggers update) ────

TEST_F(HNSWCoverageTest, InsertDuplicateLabelTriggersUpdate) {
    HNSWIndex index(4, Space::Cosine);

    // Insert with label 1
    EXPECT_TRUE(index.insert(1, {1.0f, 0.0f, 0.0f, 0.0f}));

    // Insert again with same label (triggers updatePoint internally)
    EXPECT_TRUE(index.insert(1, {0.0f, 1.0f, 0.0f, 0.0f}));

    // Should still have size 1
    EXPECT_EQ(index.size(), 1u);

    // The vector data should be updated
    const float* data = index.getVectorData(1);
    ASSERT_NE(data, nullptr);
    EXPECT_NEAR(data[0], 0.0f, 1e-5f);
    EXPECT_NEAR(data[1], 1.0f, 1e-5f);
}

TEST_F(HNSWCoverageTest, MultipleUpdatesViaInsert) {
    HNSWIndex index(4, Space::Cosine);

    // Insert and update several times
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(4, 0.0f);
        vec[i % 4] = 1.0f;
        EXPECT_TRUE(index.insert(1, vec));
    }

    EXPECT_EQ(index.size(), 1u);
}

// ── markDelete ──────────────────────────────────────────────

TEST_F(HNSWCoverageTest, MarkDelete) {
    HNSWIndex index(4, Space::Cosine);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 20; ++i) {
        index.insert(i, RandomVector(4, gen));
    }
    EXPECT_EQ(index.size(), 20u);

    // Mark some as deleted
    auto s1 = index.markDelete(5);
    EXPECT_TRUE(s1.ok());

    auto s2 = index.markDelete(10);
    EXPECT_TRUE(s2.ok());

    EXPECT_EQ(index.deletedCount(), 2u);

    // Search should not return deleted elements
    auto results = index.search(RandomVector(4, gen), 20);
    for (const auto& r : results) {
        EXPECT_NE(r.id, 5u);
        EXPECT_NE(r.id, 10u);
    }
}

TEST_F(HNSWCoverageTest, MarkDeleteNonexistent) {
    HNSWIndex index(4, Space::Cosine);

    index.insert(0, {1.0f, 2.0f, 3.0f, 4.0f});

    auto s = index.markDelete(999);
    EXPECT_FALSE(s.ok());
}

// ── getVectorData ───────────────────────────────────────────

TEST_F(HNSWCoverageTest, GetVectorData) {
    HNSWIndex index(4, Space::Cosine);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    index.insert(0, vec);

    const float* data = index.getVectorData(0);
    ASSERT_NE(data, nullptr);
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST_F(HNSWCoverageTest, GetVectorDataInvalid) {
    HNSWIndex index(4, Space::Cosine);

    const float* data = index.getVectorData(999);
    EXPECT_EQ(data, nullptr);
}

// ── reserve/deletedCount ────────────────────────────────────

TEST_F(HNSWCoverageTest, Reserve) {
    HNSWIndex index(4, Space::Cosine);

    index.reserve(10000);
    EXPECT_GE(index.capacity(), 10000u);
}

TEST_F(HNSWCoverageTest, DeletedCountInitially) {
    HNSWIndex index(4, Space::Cosine);

    EXPECT_EQ(index.deletedCount(), 0u);
}

// ── Save and load ───────────────────────────────────────────

TEST_F(HNSWCoverageTest, SaveAndLoad) {
    std::string path = GetTestPath("index.bin");

    {
        HNSWIndex index(4, Space::Cosine);
        std::mt19937 gen(42);
        for (InternalID i = 0; i < 100; ++i) {
            index.insert(i, RandomVector(4, gen));
        }

        auto s = index.saveIndex(path);
        EXPECT_TRUE(s.ok());
    }

    {
        HNSWIndex index(4, Space::Cosine);
        auto s = index.loadIndex(path);
        EXPECT_TRUE(s.ok());
        EXPECT_EQ(index.size(), 100u);
    }
}

// ── Search with filter (IDFilter) ───────────────────────────

TEST_F(HNSWCoverageTest, SearchWithFilter) {
    HNSWIndex index(4, Space::Cosine);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 50; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    // Filter: only allow even IDs
    HNSWIndex::IDFilter filter = [](InternalID id) { return id % 2 == 0; };
    auto results = index.search(RandomVector(4, gen), 10, filter);

    for (const auto& r : results) {
        EXPECT_EQ(r.id % 2, 0u);
    }
}

TEST_F(HNSWCoverageTest, SearchWithFilterDimensionMismatch) {
    HNSWIndex index(4, Space::Cosine);

    index.insert(0, {1.0f, 2.0f, 3.0f, 4.0f});

    HNSWIndex::IDFilter filter = [](InternalID) { return true; };
    auto results = index.search({1.0f, 2.0f}, 5, filter);
    EXPECT_TRUE(results.empty());
}

// ── Non-4-aligned dimensions ────────────────────────────────

TEST_F(HNSWCoverageTest, NonAlignedDimension13Cosine) {
    HNSWIndex index(13, Space::Cosine);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 50; ++i) {
        index.insert(i, RandomVector(13, gen));
    }

    auto results = index.search(RandomVector(13, gen), 5);
    EXPECT_EQ(results.size(), 5u);
}

TEST_F(HNSWCoverageTest, NonAlignedDimension17L2) {
    HNSWIndex index(17, Space::L2);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 50; ++i) {
        index.insert(i, RandomVector(17, gen));
    }

    auto results = index.search(RandomVector(17, gen), 5);
    EXPECT_EQ(results.size(), 5u);
}

TEST_F(HNSWCoverageTest, NonAlignedDimension11IP) {
    HNSWIndex index(11, Space::InnerProduct);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 50; ++i) {
        index.insert(i, RandomVector(11, gen));
    }

    auto results = index.search(RandomVector(11, gen), 5);
    EXPECT_EQ(results.size(), 5u);
}

// ── Quantized search (SQ8 after optimize) ───────────────────

TEST_F(HNSWCoverageTest, QuantizedSearch) {
    HNSWConfig cfg;
    cfg.quantize = true;
    HNSWIndex index(4, Space::Cosine, cfg);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 100; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    // Enable global SQ and BFS reorder
    index.computeGlobalSQ();
    index.reorderBFS();

    EXPECT_TRUE(index.isGlobalSQ());

    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_EQ(results.size(), 10u);
}

// ── Move semantics ──────────────────────────────────────────

TEST_F(HNSWCoverageTest, MoveConstructor) {
    HNSWIndex index1(4, Space::Cosine);
    index1.insert(0, {1.0f, 2.0f, 3.0f, 4.0f});

    HNSWIndex index2(std::move(index1));
    EXPECT_EQ(index2.size(), 1u);
}

TEST_F(HNSWCoverageTest, MoveAssignment) {
    HNSWIndex index1(4, Space::Cosine);
    index1.insert(0, {1.0f, 2.0f, 3.0f, 4.0f});

    HNSWIndex index2(4, Space::L2);
    index2 = std::move(index1);
    EXPECT_EQ(index2.size(), 1u);
}

// ── BFS reorder with deletions ──────────────────────────────

TEST_F(HNSWCoverageTest, ReorderBFSWithDeletions) {
    HNSWIndex index(4, Space::Cosine);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 50; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    // Delete some elements
    index.markDelete(5);
    index.markDelete(15);
    index.markDelete(25);

    // BFS reorder should handle deleted elements
    EXPECT_NO_THROW(index.reorderBFS());

    // Search should still work
    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_FALSE(results.empty());
}

// ── Quantized save/load ─────────────────────────────────────

// ── Unmark delete via re-insert ─────────────────────────────

TEST_F(HNSWCoverageTest, UnmarkDeleteViaReinsert) {
    HNSWIndex index(4, Space::Cosine);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 20; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    ASSERT_TRUE(index.markDelete(5).ok());
    EXPECT_EQ(index.deletedCount(), 1u);

    // Re-insert same label — triggers unmarkDeletedInternal + updatePoint
    index.insert(5, RandomVector(4, gen));
    EXPECT_EQ(index.deletedCount(), 0u);

    auto results = index.search(RandomVector(4, gen), 20);
    bool found = false;
    for (const auto& r : results) {
        if (r.id == 5) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

// ── UpdatePoint single element (early return) ───────────────

TEST_F(HNSWCoverageTest, UpdatePointSingleElement) {
    HNSWIndex index(4, Space::Cosine);

    index.insert(0, {1.0f, 0.0f, 0.0f, 0.0f});
    index.insert(0, {0.0f, 1.0f, 0.0f, 0.0f});

    EXPECT_EQ(index.size(), 1u);
    const float* data = index.getVectorData(0);
    ASSERT_NE(data, nullptr);
    EXPECT_NEAR(data[1], 1.0f, 1e-5f);
}

// ── UpdatePoint with SQ re-quantization ─────────────────────

TEST_F(HNSWCoverageTest, UpdatePointWithSQ) {
    HNSWConfig cfg;
    cfg.quantize = true;
    HNSWIndex index(4, Space::Cosine, cfg);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 50; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    index.computeGlobalSQ();
    ASSERT_TRUE(index.isGlobalSQ());

    index.insert(10, RandomVector(4, gen));
    EXPECT_EQ(index.size(), 50u);

    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_EQ(results.size(), 10u);
}

// ── BFS reorder with SQ ─────────────────────────────────────

TEST_F(HNSWCoverageTest, ReorderBFSWithSQ) {
    HNSWConfig cfg;
    cfg.quantize = true;
    HNSWIndex index(4, Space::Cosine, cfg);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 200; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    index.computeGlobalSQ();
    ASSERT_TRUE(index.isGlobalSQ());

    EXPECT_NO_THROW(index.reorderBFS());

    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_EQ(results.size(), 10u);
}

// ── BFS reorder with large index (upper layers) ─────────────

TEST_F(HNSWCoverageTest, ReorderBFSLargeIndex) {
    HNSWConfig cfg;
    cfg.M = 8;
    HNSWIndex index(4, Space::Cosine, cfg);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 500; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    EXPECT_NO_THROW(index.reorderBFS());

    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_EQ(results.size(), 10u);
}

// ── Deleted entry point then insert ─────────────────────────

TEST_F(HNSWCoverageTest, DeletedEntryPointThenInsert) {
    HNSWIndex index(4, Space::Cosine);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 100; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    ASSERT_TRUE(index.markDelete(0).ok());

    index.insert(100, RandomVector(4, gen));

    EXPECT_EQ(index.size(), 101u);
    EXPECT_EQ(index.deletedCount(), 1u);

    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_FALSE(results.empty());
}

// ── Deleted entry point then update (re-insert existing) ────

TEST_F(HNSWCoverageTest, DeletedEntryPointThenUpdate) {
    HNSWIndex index(4, Space::Cosine);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 100; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    ASSERT_TRUE(index.markDelete(0).ok());

    index.insert(50, RandomVector(4, gen));

    EXPECT_EQ(index.size(), 100u);
    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_FALSE(results.empty());
}

// ── nth_element heuristic path ──────────────────────────────

TEST_F(HNSWCoverageTest, NthElementHeuristicPath) {
    HNSWConfig cfg;
    cfg.M = 4;
    cfg.efConstruction = 200;
    HNSWIndex index(4, Space::Cosine, cfg);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 500; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_EQ(results.size(), 10u);
}

// ── UpdatePoint multi-level (greedy descent in repair) ──────

TEST_F(HNSWCoverageTest, UpdatePointMultiLevel) {
    HNSWConfig cfg;
    cfg.M = 4;
    HNSWIndex index(4, Space::Cosine, cfg);

    std::mt19937 gen(42);
    for (InternalID i = 0; i < 2000; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    for (InternalID i = 0; i < 10; ++i) {
        index.insert(i, RandomVector(4, gen));
    }

    EXPECT_EQ(index.size(), 2000u);
    auto results = index.search(RandomVector(4, gen), 10);
    EXPECT_EQ(results.size(), 10u);
}

// ── BFS reorder empty and single element ────────────────────

TEST_F(HNSWCoverageTest, ReorderBFSEmptyAndSingle) {
    {
        HNSWIndex index(4, Space::Cosine);
        EXPECT_NO_THROW(index.reorderBFS());
    }
    {
        HNSWIndex index(4, Space::Cosine);
        index.insert(0, {1.0f, 2.0f, 3.0f, 4.0f});
        EXPECT_NO_THROW(index.reorderBFS());
    }
}

// ── Quantized save/load ─────────────────────────────────────

TEST_F(HNSWCoverageTest, QuantizedSaveLoad) {
    std::string path = GetTestPath("quantized.bin");

    {
        HNSWConfig cfg;
        cfg.quantize = true;
        HNSWIndex index(4, Space::Cosine, cfg);
        std::mt19937 gen(42);
        for (InternalID i = 0; i < 100; ++i) {
            index.insert(i, RandomVector(4, gen));
        }
        index.computeGlobalSQ();
        index.reorderBFS();

        auto s = index.saveIndex(path);
        EXPECT_TRUE(s.ok());
    }

    {
        HNSWConfig cfg;
        cfg.quantize = true;
        HNSWIndex index(4, Space::Cosine, cfg);
        auto s = index.loadIndex(path);
        EXPECT_TRUE(s.ok());
        EXPECT_EQ(index.size(), 100u);
        EXPECT_TRUE(index.isGlobalSQ());
    }
}
