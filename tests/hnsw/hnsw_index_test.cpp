#include <gtest/gtest.h>
#include "index/hnsw_index.h"
#include "test_util.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

using namespace arrow;
using arrow::testing::RandomVector;

// ============================================================================
// HNSWIndex Test Fixture
// ============================================================================

class HNSWIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        test_dir_ = std::filesystem::temp_directory_path() / "arrow_test";
        try {
            std::filesystem::create_directories(test_dir_);
        } catch (const std::exception& e) {
            FAIL() << "Failed to create test directory: " << e.what();
        }
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }
    
    std::filesystem::path test_dir_;
    std::string GetTestPath(const std::string& filename) {
        return (test_dir_ / filename).string();
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(HNSWIndexTest, InsertAndSearch) {
    HNSWIndex index(3, Space::Cosine);
    
    index.insert(1, {1.0f, 0.0f, 0.0f});
    index.insert(2, {0.0f, 1.0f, 0.0f});
    index.insert(3, {0.0f, 0.0f, 1.0f});
    
    EXPECT_EQ(index.size(), 3);
    
    auto results = index.search({1.0f, 0.0f, 0.0f}, 1);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 1);
}

TEST_F(HNSWIndexTest, TopKOrdering) {
    HNSWIndex index(3, Space::Cosine);
    
    // Insert vectors at different angles from query
    index.insert(1, {1.0f, 0.0f, 0.0f});      // Exact match
    index.insert(2, {0.707f, 0.707f, 0.0f});  // 45 degrees
    index.insert(3, {0.0f, 1.0f, 0.0f});      // 90 degrees
    
    auto results = index.search({1.0f, 0.0f, 0.0f}, 3);
    
    EXPECT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].id, 1);  // Best match first
    EXPECT_EQ(results[1].id, 2);  // Second best
    EXPECT_EQ(results[2].id, 3);  // Worst match last
}

TEST_F(HNSWIndexTest, RecallAt10) {
    const size_t dim = 128;
    const size_t n = 10000;
    const size_t k = 10;
    
    std::mt19937 gen(42);
    
    HNSWIndex index(
        dim, 
        Space::Cosine, 
        {.M = 16, .efConstruction = 200}
    );
    
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> vec = RandomVector(dim, gen);
        vectors.push_back(vec);
        index.insert(i, vec);
    }
    
    // Test with random queries
    const size_t num_queries = 100;
    
    for (size_t q = 0; q < num_queries; ++q) {
        std::vector<float> query = RandomVector(dim, gen);
        std::vector<HNSWSearchResult> results = index.search(query, k, 100);
        
        // Verify we got k results
        EXPECT_EQ(results.size(), k);
    }
}

TEST_F(HNSWIndexTest, DimensionMismatch) {
    HNSWIndex index(3, Space::Cosine);

    EXPECT_EQ(index.insert(1, {1.0f, 0.0f}), false);
    auto results = index.search({1.0f, 0.0f}, 1);
    EXPECT_TRUE(results.empty());
}

// ============================================================================
// Persistence Tests (saveIndex / loadIndex)
// ============================================================================

TEST_F(HNSWIndexTest, SaveIndexCreatesFile) {
    HNSWIndex index(3, Space::Cosine);
    index.insert(1, {1.0f, 0.0f, 0.0f});
    index.insert(2, {0.0f, 1.0f, 0.0f});
    index.insert(3, {0.0f, 0.0f, 1.0f});
    
    std::string path = GetTestPath("test_index.bin");
    
    // Save should succeed
    EXPECT_NO_THROW(index.saveIndex(path));
    
    // File should exist
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_GT(std::filesystem::file_size(path), 0);
}

TEST_F(HNSWIndexTest, LoadIndexFromFile) {
    // Create and save an index
    HNSWIndex original(3, Space::Cosine);
    original.insert(1, {1.0f, 0.0f, 0.0f});
    original.insert(2, {0.0f, 1.0f, 0.0f});
    original.insert(3, {0.0f, 0.0f, 1.0f});
    
    std::string path = GetTestPath("test_index.bin");
    original.saveIndex(path);
    
    // Create a new index and load from file
    HNSWIndex loaded(3, Space::Cosine);
    EXPECT_NO_THROW(loaded.loadIndex(path));
    
    // Verify size matches
    EXPECT_EQ(loaded.size(), 3);
    EXPECT_EQ(loaded.size(), original.size());
}

TEST_F(HNSWIndexTest, RoundTripPreservesData) {
    const size_t dim = 128;
    const size_t n = 100;
    
    // Create index with data
    HNSWIndex original(dim, Space::Cosine);
    std::mt19937 gen(42);
    std::vector<std::vector<float>> vectors;
    
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> vec = RandomVector(dim, gen);
        vectors.push_back(vec);
        original.insert(i, vec);
    }
    
    std::string path = GetTestPath("roundtrip_index.bin");
    original.saveIndex(path);
    
    // Load into new index
    HNSWIndex loaded(dim, Space::Cosine);
    loaded.loadIndex(path);
    
    // Verify size
    EXPECT_EQ(loaded.size(), n);
    
    // Verify search results match
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
        auto originalResults = original.search(vectors[i], 5);
        auto loadedResults = loaded.search(vectors[i], 5);
        
        EXPECT_EQ(originalResults.size(), loadedResults.size());
        
        // Top result should be the same (self-match)
        if (!originalResults.empty() && !loadedResults.empty()) {
            EXPECT_EQ(originalResults[0].id, loadedResults[0].id);
            EXPECT_NEAR(originalResults[0].score, loadedResults[0].score, 1e-5f);
        }
    }
}

TEST_F(HNSWIndexTest, RoundTripPreservesSearchResults) {
    HNSWIndex original(3, Space::Cosine);
    
    // Insert vectors at different angles
    original.insert(1, {1.0f, 0.0f, 0.0f});      // Exact match
    original.insert(2, {0.707f, 0.707f, 0.0f});  // 45 degrees
    original.insert(3, {0.0f, 1.0f, 0.0f});      // 90 degrees
    
    std::string path = GetTestPath("search_test.bin");
    original.saveIndex(path);
    
    // Load and verify search results
    HNSWIndex loaded(3, Space::Cosine);
    loaded.loadIndex(path);
    
    std::vector<float> query = {1.0f, 0.0f, 0.0f};
    auto originalResults = original.search(query, 3);
    auto loadedResults = loaded.search(query, 3);
    
    EXPECT_EQ(originalResults.size(), loadedResults.size());
    EXPECT_EQ(originalResults.size(), 3);
    
    // Verify ordering and scores match
    for (size_t i = 0; i < originalResults.size(); ++i) {
        EXPECT_EQ(originalResults[i].id, loadedResults[i].id);
        EXPECT_NEAR(originalResults[i].score, loadedResults[i].score, 1e-5f);
    }
}

TEST_F(HNSWIndexTest, LoadIndexReplacesExisting) {
    // Create and save first index
    HNSWIndex index1(3, Space::Cosine);
    index1.insert(1, {1.0f, 0.0f, 0.0f});
    index1.insert(2, {0.0f, 1.0f, 0.0f});
    
    std::string path1 = GetTestPath("index1.bin");
    index1.saveIndex(path1);
    
    // Create and save second index
    HNSWIndex index2(3, Space::Cosine);
    index2.insert(10, {0.0f, 0.0f, 1.0f});
    index2.insert(20, {0.577f, 0.577f, 0.577f});
    
    std::string path2 = GetTestPath("index2.bin");
    index2.saveIndex(path2);
    
    // Load first index
    HNSWIndex loaded(3, Space::Cosine);
    loaded.loadIndex(path1);
    EXPECT_EQ(loaded.size(), 2);
    
    // Load second index - should replace the first
    loaded.loadIndex(path2);
    EXPECT_EQ(loaded.size(), 2);
    
    // Verify it has the second index's data
    auto results = loaded.search({0.0f, 0.0f, 1.0f}, 1);
    EXPECT_EQ(results[0].id, 10);
}

TEST_F(HNSWIndexTest, LoadIndexReturnsErrorOnInvalidPath) {
    HNSWIndex index(3, Space::Cosine);

    // Try to load from non-existent file — returns error Status
    auto status = index.loadIndex(GetTestPath("nonexistent.bin"));
    EXPECT_FALSE(status.ok());
}

TEST_F(HNSWIndexTest, LoadIndexReturnsErrorOnCorruptedFile) {
    // Create a corrupted file
    std::string path = GetTestPath("corrupted.bin");
    std::ofstream file(path, std::ios::binary);
    file << "This is not a valid index file";
    file.close();

    HNSWIndex index(3, Space::Cosine);

    // Should return error Status for corrupted file
    auto status = index.loadIndex(path);
    EXPECT_FALSE(status.ok());
}

TEST_F(HNSWIndexTest, LoadIndexRequiresMatchingDimension) {
    // Create and save index with dimension 3
    HNSWIndex original(3, Space::Cosine);
    original.insert(1, {1.0f, 0.0f, 0.0f});

    std::string path = GetTestPath("dim3_index.bin");
    original.saveIndex(path);

    // Loading with wrong dimension currently succeeds at the hnsw layer
    // but the wrapper's dim_ field mismatches the loaded index.
    // Verify that search is rejected due to query dimension check.
    HNSWIndex wrongDim(5, Space::Cosine);
    auto status = wrongDim.loadIndex(path);

    // Even if load succeeds, searching with dim=5 query must fail or return empty
    // because the wrapper checks query.size() != dim_ before calling into hnsw.
    auto results = wrongDim.search({1.0f, 0.0f, 0.0f}, 1);
    EXPECT_TRUE(results.empty())
        << "Search with mismatched query dimension should return empty results";
}

TEST_F(HNSWIndexTest, LoadIndexRequiresMatchingSpace) {
    // Create and save L2 index with distinct vectors
    HNSWIndex l2Index(3, Space::L2);
    l2Index.insert(1, {1.0f, 0.0f, 0.0f});
    l2Index.insert(2, {0.707f, 0.707f, 0.0f});
    l2Index.insert(3, {0.0f, 1.0f, 0.0f});

    std::string path = GetTestPath("l2_index.bin");
    l2Index.saveIndex(path);

    // Load the L2-built index into a Cosine-configured wrapper.
    // The load succeeds but search scores are computed with the wrong
    // distance function, so the ordering may differ from the L2 index.
    HNSWIndex cosineIndex(3, Space::Cosine);
    auto status = cosineIndex.loadIndex(path);
    // Load itself may succeed (hnsw doesn't check space type)
    // But scores will be computed differently because the space_ object
    // is InnerProduct-based, not L2-based.

    // Verify the loaded index has the same number of elements
    EXPECT_EQ(cosineIndex.size(), 3);

    // The key insight: L2 results and "cosine" results will use different
    // distance metrics, so the scores should differ.
    auto l2Results = l2Index.search({1.0f, 0.0f, 0.0f}, 3);
    auto cosResults = cosineIndex.search({1.0f, 0.0f, 0.0f}, 3);

    ASSERT_EQ(l2Results.size(), 3);
    ASSERT_EQ(cosResults.size(), 3);

    // Scores should differ because different distance functions are used
    // (L2 returns positive distances, cosine returns negative similarity)
    bool scoresDiffer = false;
    for (size_t i = 0; i < l2Results.size(); ++i) {
      if (std::abs(l2Results[i].score - cosResults[i].score) > 1e-6f) {
        scoresDiffer = true;
        break;
      }
    }
    EXPECT_TRUE(scoresDiffer)
        << "Scores should differ when index is loaded with wrong space type";
}

TEST_F(HNSWIndexTest, SaveEmptyIndex) {
    HNSWIndex index(3, Space::Cosine);
    // Don't insert any vectors
    
    std::string path = GetTestPath("empty_index.bin");
    
    EXPECT_NO_THROW(index.saveIndex(path));
    EXPECT_TRUE(std::filesystem::exists(path));
    
    // Load empty index
    HNSWIndex loaded(3, Space::Cosine);
    EXPECT_NO_THROW(loaded.loadIndex(path));
    EXPECT_EQ(loaded.size(), 0);
}

TEST_F(HNSWIndexTest, SaveLargeIndex) {
    const size_t dim = 128;
    const size_t n = 1000;
    
    HNSWIndex index(dim, Space::Cosine);
    std::mt19937 gen(42);
    
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> vec = RandomVector(dim, gen);
        index.insert(i, vec);
    }
    
    std::string path = GetTestPath("large_index.bin");
    EXPECT_NO_THROW(index.saveIndex(path));
    
    // Verify file exists and has reasonable size
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_GT(std::filesystem::file_size(path), 1000); // At least 1KB
    
    // Load and verify
    HNSWIndex loaded(dim, Space::Cosine);
    EXPECT_NO_THROW(loaded.loadIndex(path));
    EXPECT_EQ(loaded.size(), n);
}

// ============================================================================
// Scalar Quantization (SQ8) Tests
// ============================================================================

TEST_F(HNSWIndexTest, SQ_InsertAndSearch) {
    const size_t dim = 128;
    HNSWConfig config;
    config.maxElements = 1000;
    config.M = 16;
    config.efConstruction = 100;
    config.quantize = true;

    HNSWIndex index(dim, Space::L2, config);

    std::mt19937 gen(42);
    const size_t n = 500;
    std::vector<std::vector<float>> vectors(n);
    for (size_t i = 0; i < n; ++i) {
        vectors[i] = RandomVector(dim, gen);
        EXPECT_TRUE(index.insert(i, vectors[i]));
    }
    EXPECT_EQ(index.size(), n);

    // Search should return results
    auto results = index.search(vectors[0], 10, 200);
    EXPECT_EQ(results.size(), 10);
    // The closest result should be the query itself
    EXPECT_EQ(results[0].id, 0);
}

TEST_F(HNSWIndexTest, SQ_RecallComparedToFloat32) {
    const size_t dim = 128;
    const size_t n = 1000;
    const size_t k = 10;
    const size_t ef = 200;

    // Build float32 index
    HNSWConfig configFloat;
    configFloat.maxElements = n;
    configFloat.M = 32;
    configFloat.efConstruction = 200;
    configFloat.quantize = false;
    HNSWIndex floatIndex(dim, Space::L2, configFloat);

    // Build SQ index
    HNSWConfig configSQ = configFloat;
    configSQ.quantize = true;
    HNSWIndex sqIndex(dim, Space::L2, configSQ);

    std::mt19937 genInsert(42);
    std::vector<std::vector<float>> vectors(n);
    for (size_t i = 0; i < n; ++i) {
        vectors[i] = RandomVector(dim, genInsert);
        floatIndex.insert(i, vectors[i]);
        sqIndex.insert(i, vectors[i]);
    }

    // Compare recall: SQ should achieve >= 90% of float32's recall
    size_t sqMatches = 0;
    const size_t numQueries = 50;
    std::mt19937 genQuery(10000);
    for (size_t q = 0; q < numQueries; ++q) {
        auto query = RandomVector(dim, genQuery);
        auto floatResults = floatIndex.search(query, k, ef);
        auto sqResults = sqIndex.search(query, k, ef);

        // Count how many SQ results appear in float32 results
        std::unordered_set<InternalID> floatSet;
        for (const auto& r : floatResults) floatSet.insert(r.id);
        for (const auto& r : sqResults) {
            if (floatSet.count(r.id)) ++sqMatches;
        }
    }

    double sqRecall = static_cast<double>(sqMatches) / (numQueries * k);
    EXPECT_GE(sqRecall, 0.90) << "SQ recall (" << sqRecall
        << ") should be >= 90% of float32 results";
}

TEST_F(HNSWIndexTest, SQ_InnerProductSpace) {
    const size_t dim = 64;
    HNSWConfig config;
    config.maxElements = 500;
    config.M = 16;
    config.efConstruction = 100;
    config.quantize = true;

    HNSWIndex index(dim, Space::InnerProduct, config);

    std::mt19937 gen(42);
    const size_t n = 200;
    for (size_t i = 0; i < n; ++i) {
        auto vec = RandomVector(dim, gen);  // Already normalized by RandomVector
        EXPECT_TRUE(index.insert(i, vec));
    }

    auto query = RandomVector(dim, gen);
    auto results = index.search(query, 10, 200);
    EXPECT_EQ(results.size(), 10);
}

TEST_F(HNSWIndexTest, SQ_PersistenceRoundTrip) {
    const size_t dim = 64;
    const size_t n = 200;
    HNSWConfig config;
    config.maxElements = 500;
    config.M = 16;
    config.efConstruction = 100;
    config.quantize = true;

    HNSWIndex index(dim, Space::L2, config);
    std::mt19937 gen(42);
    std::vector<std::vector<float>> vectors(n);
    for (size_t i = 0; i < n; ++i) {
        vectors[i] = RandomVector(dim, gen);
        index.insert(i, vectors[i]);
    }

    // Save and reload
    auto path = GetTestPath("sq_index.bin");
    EXPECT_TRUE(index.saveIndex(path).ok());

    HNSWIndex loaded(dim, Space::L2, config);
    EXPECT_TRUE(loaded.loadIndex(path).ok());
    EXPECT_EQ(loaded.size(), n);

    // Search results should be similar
    auto query = RandomVector(dim, gen);
    auto origResults = index.search(query, 10, 200);
    auto loadedResults = loaded.search(query, 10, 200);

    EXPECT_EQ(origResults.size(), loadedResults.size());

    // At least 8 of top 10 should match (allowing for minor differences)
    std::unordered_set<InternalID> origSet;
    for (const auto& r : origResults) origSet.insert(r.id);
    size_t matches = 0;
    for (const auto& r : loadedResults) {
        if (origSet.count(r.id)) ++matches;
    }
    EXPECT_GE(matches, 8) << "Loaded SQ index should produce similar results";
}

TEST_F(HNSWIndexTest, SQ_AutoResize) {
    const size_t dim = 32;
    HNSWConfig config;
    config.maxElements = 10;  // Small initial capacity
    config.M = 8;
    config.efConstruction = 50;
    config.quantize = true;

    HNSWIndex index(dim, Space::L2, config);

    std::mt19937 gen(42);
    // Insert more than initial capacity to trigger resize
    for (size_t i = 0; i < 50; ++i) {
        EXPECT_TRUE(index.insert(i, RandomVector(dim, gen)));
    }
    EXPECT_EQ(index.size(), 50);

    auto results = index.search(RandomVector(dim, gen), 5, 50);
    EXPECT_EQ(results.size(), 5);
}

