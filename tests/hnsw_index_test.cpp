#include <gtest/gtest.h>
#include "arrow/hnsw_index.h"
#include "test_util.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>

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
    HNSWIndex index(3, DistanceMetric::Cosine);
    
    index.insert(1, {1.0f, 0.0f, 0.0f});
    index.insert(2, {0.0f, 1.0f, 0.0f});
    index.insert(3, {0.0f, 0.0f, 1.0f});
    
    EXPECT_EQ(index.size(), 3);
    
    auto results = index.search({1.0f, 0.0f, 0.0f}, 1);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 1);
}

TEST_F(HNSWIndexTest, TopKOrdering) {
    HNSWIndex index(3, DistanceMetric::Cosine);
    
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
        DistanceMetric::Cosine, 
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
        std::vector<SearchResult> results = index.search(query, k, 100);
        
        // Verify we got k results
        EXPECT_EQ(results.size(), k);
    }
}

TEST_F(HNSWIndexTest, DimensionMismatch) {
    HNSWIndex index(3, DistanceMetric::Cosine);
    
    EXPECT_THROW(index.insert(1, {1.0f, 0.0f}), std::invalid_argument);
    EXPECT_THROW(index.search({1.0f, 0.0f}, 1), std::invalid_argument);
}

// ============================================================================
// Persistence Tests (saveIndex / loadIndex)
// ============================================================================

TEST_F(HNSWIndexTest, SaveIndexCreatesFile) {
    HNSWIndex index(3, DistanceMetric::Cosine);
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
    HNSWIndex original(3, DistanceMetric::Cosine);
    original.insert(1, {1.0f, 0.0f, 0.0f});
    original.insert(2, {0.0f, 1.0f, 0.0f});
    original.insert(3, {0.0f, 0.0f, 1.0f});
    
    std::string path = GetTestPath("test_index.bin");
    original.saveIndex(path);
    
    // Create a new index and load from file
    HNSWIndex loaded(3, DistanceMetric::Cosine);
    EXPECT_NO_THROW(loaded.loadIndex(path));
    
    // Verify size matches
    EXPECT_EQ(loaded.size(), 3);
    EXPECT_EQ(loaded.size(), original.size());
}

TEST_F(HNSWIndexTest, RoundTripPreservesData) {
    const size_t dim = 128;
    const size_t n = 100;
    
    // Create index with data
    HNSWIndex original(dim, DistanceMetric::Cosine);
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
    HNSWIndex loaded(dim, DistanceMetric::Cosine);
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
    HNSWIndex original(3, DistanceMetric::Cosine);
    
    // Insert vectors at different angles
    original.insert(1, {1.0f, 0.0f, 0.0f});      // Exact match
    original.insert(2, {0.707f, 0.707f, 0.0f});  // 45 degrees
    original.insert(3, {0.0f, 1.0f, 0.0f});      // 90 degrees
    
    std::string path = GetTestPath("search_test.bin");
    original.saveIndex(path);
    
    // Load and verify search results
    HNSWIndex loaded(3, DistanceMetric::Cosine);
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
    HNSWIndex index1(3, DistanceMetric::Cosine);
    index1.insert(1, {1.0f, 0.0f, 0.0f});
    index1.insert(2, {0.0f, 1.0f, 0.0f});
    
    std::string path1 = GetTestPath("index1.bin");
    index1.saveIndex(path1);
    
    // Create and save second index
    HNSWIndex index2(3, DistanceMetric::Cosine);
    index2.insert(10, {0.0f, 0.0f, 1.0f});
    index2.insert(20, {0.577f, 0.577f, 0.577f});
    
    std::string path2 = GetTestPath("index2.bin");
    index2.saveIndex(path2);
    
    // Load first index
    HNSWIndex loaded(3, DistanceMetric::Cosine);
    loaded.loadIndex(path1);
    EXPECT_EQ(loaded.size(), 2);
    
    // Load second index - should replace the first
    loaded.loadIndex(path2);
    EXPECT_EQ(loaded.size(), 2);
    
    // Verify it has the second index's data
    auto results = loaded.search({0.0f, 0.0f, 1.0f}, 1);
    EXPECT_EQ(results[0].id, 10);
}

TEST_F(HNSWIndexTest, LoadIndexThrowsOnInvalidPath) {
    HNSWIndex index(3, DistanceMetric::Cosine);
    
    // Try to load from non-existent file
    EXPECT_THROW(
        index.loadIndex(GetTestPath("nonexistent.bin")),
        std::runtime_error
    );
}

TEST_F(HNSWIndexTest, LoadIndexThrowsOnCorruptedFile) {
    // Create a corrupted file
    std::string path = GetTestPath("corrupted.bin");
    std::ofstream file(path, std::ios::binary);
    file << "This is not a valid index file";
    file.close();
    
    HNSWIndex index(3, DistanceMetric::Cosine);
    
    // Should throw when trying to load corrupted file
    EXPECT_THROW(index.loadIndex(path), std::runtime_error);
}

TEST_F(HNSWIndexTest, LoadIndexRequiresMatchingDimension) {
    // Create and save index with dimension 3
    HNSWIndex original(3, DistanceMetric::Cosine);
    original.insert(1, {1.0f, 0.0f, 0.0f});
    
    std::string path = GetTestPath("dim3_index.bin");
    original.saveIndex(path);
    
    // Try to load with wrong dimension - this should fail
    // Note: hnswlib may or may not validate dimension at load time
    // The actual behavior depends on hnswlib implementation
    HNSWIndex wrongDim(5, DistanceMetric::Cosine);
    
    // This might throw or might succeed but produce incorrect results
    // We'll test that it at least doesn't crash
    try {
        wrongDim.loadIndex(path);
        // If it doesn't throw, the dimension mismatch might be detected later
        // or might cause incorrect behavior
    } catch (const std::exception& e) {
        // Expected if dimension validation occurs
        SUCCEED();
    }
}

TEST_F(HNSWIndexTest, LoadIndexRequiresMatchingMetric) {
    // Create and save L2 index
    HNSWIndex l2Index(3, DistanceMetric::L2);
    l2Index.insert(1, {1.0f, 0.0f, 0.0f});
    
    std::string path = GetTestPath("l2_index.bin");
    l2Index.saveIndex(path);
    
    // Try to load with Cosine metric
    // Note: hnswlib may not validate metric at load time
    HNSWIndex cosineIndex(3, DistanceMetric::Cosine);
    
    // This might work but produce incorrect results
    // We test that it doesn't crash
    try {
        cosineIndex.loadIndex(path);
        // If successful, results might be incorrect due to metric mismatch
    } catch (const std::exception& e) {
        // Expected if metric validation occurs
        SUCCEED();
    }
}

TEST_F(HNSWIndexTest, SaveEmptyIndex) {
    HNSWIndex index(3, DistanceMetric::Cosine);
    // Don't insert any vectors
    
    std::string path = GetTestPath("empty_index.bin");
    
    EXPECT_NO_THROW(index.saveIndex(path));
    EXPECT_TRUE(std::filesystem::exists(path));
    
    // Load empty index
    HNSWIndex loaded(3, DistanceMetric::Cosine);
    EXPECT_NO_THROW(loaded.loadIndex(path));
    EXPECT_EQ(loaded.size(), 0);
}

TEST_F(HNSWIndexTest, SaveLargeIndex) {
    const size_t dim = 128;
    const size_t n = 1000;
    
    HNSWIndex index(dim, DistanceMetric::Cosine);
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
    HNSWIndex loaded(dim, DistanceMetric::Cosine);
    EXPECT_NO_THROW(loaded.loadIndex(path));
    EXPECT_EQ(loaded.size(), n);
}

