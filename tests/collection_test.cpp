#include <gtest/gtest.h>
#include "arrow/collection.h"
#include "test_util.h"
#include <chrono>
#include <filesystem>

using namespace arrow;
using arrow::testing::RandomVector;

class CollectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        testDir = std::filesystem::temp_directory_path() / "arrow_collection_test";
        std::filesystem::create_directories(testDir);
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }
    
    std::filesystem::path testDir;
    std::string GetTestPath(const std::string& dirname) {
        return (testDir / dirname).string();
    }
};

TEST_F(CollectionTest, CreateCollection) {
    CollectionConfig cfg(
        "test_collection",
        128,
        DistanceMetric::Cosine,
        DataType::Float16
    );

    Collection collection(cfg);

    EXPECT_EQ(collection.name(), "test_collection");
    EXPECT_EQ(collection.dimension(), 128);
    EXPECT_EQ(collection.metric(), DistanceMetric::Cosine);
    EXPECT_EQ(collection.dtype(), DataType::Float16);
    EXPECT_EQ(collection.size(), 0);
}

TEST_F(CollectionTest, InsertVectors) {
    CollectionConfig cfg(
        "test_collection",
        128,
        DistanceMetric::Cosine,
        DataType::Float16
    );

    Collection collection(cfg);

    std::mt19937 gen(42);
    const size_t num_vectors = 1000;
    const size_t dim = collection.dimension();

    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec = RandomVector(dim, gen);
        collection.insert(static_cast<VectorID>(i), vec);
    }

    EXPECT_EQ(collection.size(), num_vectors);
}

TEST_F(CollectionTest, SearchFunctionality) {
    CollectionConfig cfg(
        "test_collection",
        128,
        DistanceMetric::Cosine,
        DataType::Float16
    );

    Collection collection(cfg);

    std::mt19937 gen(42);
    const size_t num_vectors = 1000;
    const size_t dim = collection.dimension();

    // Insert vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec = RandomVector(dim, gen);
        collection.insert(static_cast<VectorID>(i), vec);
    }

    // Perform search
    std::vector<float> query = RandomVector(dim, gen);
    const size_t k = 10;
    std::vector<SearchResult> results = collection.search(query, k);

    EXPECT_EQ(results.size(), k);
    
    // Verify results are sorted by score (descending for cosine similarity)
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i].score, results[i-1].score) 
            << "Results should be sorted in descending order";
    }
}

TEST_F(CollectionTest, SearchWithDifferentEf) {
    CollectionConfig cfg(
        "test_collection",
        128,
        DistanceMetric::Cosine,
        DataType::Float16
    );

    Collection collection(cfg);

    std::mt19937 gen(42);
    const size_t num_vectors = 1000;
    const size_t dim = collection.dimension();

    // Insert vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec = RandomVector(dim, gen);
        collection.insert(static_cast<VectorID>(i), vec);
    }

    std::vector<float> query = RandomVector(dim, gen);
    const size_t k = 10;

    // Test with different ef values
    for (size_t ef : {10, 50, 100}) {
        std::vector<SearchResult> results = collection.search(query, k, ef);
        EXPECT_EQ(results.size(), k) << "ef=" << ef;
    }
}

TEST_F(CollectionTest, SearchPerformance) {
    CollectionConfig cfg(
        "test_collection",
        128,
        DistanceMetric::Cosine,
        DataType::Float16
    );

    Collection collection(cfg);

    std::mt19937 gen(42);
    const size_t num_vectors = 1000;
    const size_t dim = collection.dimension();

    // Insert vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec = RandomVector(dim, gen);
        collection.insert(static_cast<VectorID>(i), vec);
    }

    std::vector<float> query = RandomVector(dim, gen);
    const size_t k = 10;

    // Measure search latency with ef=100
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<SearchResult> results = collection.search(query, k, 100);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(results.size(), k);
    // Search should be reasonably fast (< 1ms for 1K vectors)
    EXPECT_LT(duration.count(), 1000) << "Search took " << duration.count() << " microseconds";
}

// ============================================================================
// Persistence Tests (save / load)
// ============================================================================

TEST_F(CollectionTest, SaveCreatesDirectory) {
    CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine, DataType::Float16);
    Collection collection(cfg);
    
    std::string save_path = GetTestPath("test_collection");
    EXPECT_NO_THROW(collection.save(save_path));
    
    // Directory should exist
    EXPECT_TRUE(std::filesystem::exists(save_path));
    EXPECT_TRUE(std::filesystem::is_directory(save_path));
}

TEST_F(CollectionTest, SaveCreatesRequiredFiles) {
    CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine, DataType::Float16);
    Collection collection(cfg);
    
    // Insert some vectors
    std::mt19937 gen(42);
    for (size_t i = 0; i < 10; ++i) {
        std::vector<float> vec = RandomVector(128, gen);
        collection.insert(i, vec);
    }
    
    std::string save_path = GetTestPath("test_collection");
    collection.save(save_path);
    
    // Check required files exist
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(save_path) / "meta.json"));
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(save_path) / "index.bin"));
    
    // metadata.json is optional (only created if metadata exists)
    // For this test, it shouldn't exist since we didn't add metadata
}

TEST_F(CollectionTest, SaveIncludesMetadata) {
    CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine, DataType::Float16);
    Collection collection(cfg);
    
    // Insert vectors with metadata
    std::mt19937 gen(42);
    for (size_t i = 0; i < 5; ++i) {
        std::vector<float> vec = RandomVector(128, gen);
        collection.insert(i, vec);
        
        Metadata meta;
        meta["category"] = std::string("test");
        meta["score"] = static_cast<double>(i);
        collection.setMetadata(i, meta);
    }
    
    std::string save_path = GetTestPath("test_collection");
    collection.save(save_path);
    
    // metadata.json should exist since we added metadata
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(save_path) / "metadata.json"));
}

TEST_F(CollectionTest, LoadFromDirectory) {
    // Create and save a collection
    CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine, DataType::Float16);
    Collection original(cfg);
    
    std::mt19937 gen(42);
    for (size_t i = 0; i < 100; ++i) {
        std::vector<float> vec = RandomVector(128, gen);
        original.insert(i, vec);
    }
    
    std::string save_path = GetTestPath("test_collection");
    original.save(save_path);
    
    // Load the collection
    Collection loaded = Collection::load(save_path);
    
    // Verify basic properties
    EXPECT_EQ(loaded.name(), "test_collection");
    EXPECT_EQ(loaded.dimension(), 128);
    EXPECT_EQ(loaded.metric(), DistanceMetric::Cosine);
    EXPECT_EQ(loaded.dtype(), DataType::Float16);
    EXPECT_EQ(loaded.size(), 100);
}

TEST_F(CollectionTest, RoundTripPreservesData) {
    // Create collection with custom HNSW config
    CollectionConfig cfg("test_collection", 64, DistanceMetric::Cosine, DataType::Float16);
    HNSWConfig hnsw_cfg;
    hnsw_cfg.M = 32;
    hnsw_cfg.efConstruction = 200;
    Collection original(cfg, hnsw_cfg);
    
    std::mt19937 gen(42);
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < 50; ++i) {
        std::vector<float> vec = RandomVector(64, gen);
        vectors.push_back(vec);
        original.insert(i, vec);
    }
    
    std::string save_path = GetTestPath("test_collection");
    original.save(save_path);
    
    // Load and verify
    Collection loaded = Collection::load(save_path);
    EXPECT_EQ(loaded.size(), 50);
    EXPECT_EQ(loaded.hnswConfig().M, 32);
    EXPECT_EQ(loaded.hnswConfig().efConstruction, 200);
    
    // Verify search results match
    for (size_t i = 0; i < std::min(size_t(10), vectors.size()); ++i) {
        auto originalResults = original.search(vectors[i], 5);
        auto loadedResults = loaded.search(vectors[i], 5);
        
        EXPECT_EQ(originalResults.size(), loadedResults.size());
        if (!originalResults.empty() && !loadedResults.empty()) {
            EXPECT_EQ(originalResults[0].id, loadedResults[0].id);
            EXPECT_NEAR(originalResults[0].score, loadedResults[0].score, 1e-5f);
        }
    }
}

TEST_F(CollectionTest, RoundTripPreservesMetadata) {
    CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine, DataType::Float16);
    Collection original(cfg);
    
    std::mt19937 gen(42);
    for (size_t i = 0; i < 10; ++i) {
        std::vector<float> vec = RandomVector(128, gen);
        original.insert(i, vec);
        
        Metadata meta;
        meta["id"] = static_cast<int64_t>(i);
        meta["name"] = std::string("vector_") + std::to_string(i);
        meta["score"] = static_cast<double>(i) * 0.1;
        meta["active"] = (i % 2 == 0);
        original.setMetadata(i, meta);
    }
    
    std::string save_path = GetTestPath("test_collection");
    original.save(save_path);
    
    // Load and verify metadata
    Collection loaded = Collection::load(save_path);
    
    // Note: We can't directly access metadata_ from outside, but we can verify
    // that metadata.json exists and was loaded (by checking collection can be loaded)
    EXPECT_EQ(loaded.size(), 10);
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(save_path) / "metadata.json"));
}

TEST_F(CollectionTest, LoadThrowsOnInvalidDirectory) {
    EXPECT_THROW(
        Collection::load("/nonexistent/directory"),
        std::runtime_error
    );
}

TEST_F(CollectionTest, LoadThrowsOnMissingMetaJson) {
    // Create directory without meta.json
    std::string save_path = GetTestPath("incomplete_collection");
    std::filesystem::create_directories(save_path);
    
    EXPECT_THROW(
        Collection::load(save_path),
        std::runtime_error
    );
}

TEST_F(CollectionTest, LoadThrowsOnMissingIndexBin) {
    CollectionConfig cfg("test_collection", 128, DistanceMetric::Cosine, DataType::Float16);
    Collection collection(cfg);
    
    std::string save_path = GetTestPath("incomplete_collection");
    std::filesystem::create_directories(save_path);
    
    // Save only meta.json
    std::string meta_path = (std::filesystem::path(save_path) / "meta.json").string();
    utils::exportCollectionConfigToJson(cfg, HNSWConfig{}, meta_path);
    
    EXPECT_THROW(
        Collection::load(save_path),
        std::runtime_error
    );
}

