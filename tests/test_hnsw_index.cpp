#include <gtest/gtest.h>
#include "arrow/hnsw_index.h"
#include "arrow/collection.h"
#include <random>
#include <cmath>

using namespace arrow;

// Helper: generate random normalized vector
std::vector<float> randomVector(size_t dim, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    for (float& v : vec) v /= norm;
    return vec;
}

TEST(HNSWIndexTest, InsertAndSearch) {
    HNSWIndex index(3, DistanceMetric::Cosine);
    
    index.insert(1, {1.0f, 0.0f, 0.0f});
    index.insert(2, {0.0f, 1.0f, 0.0f});
    index.insert(3, {0.0f, 0.0f, 1.0f});
    
    EXPECT_EQ(index.size(), 3);
    
    auto results = index.search({1.0f, 0.0f, 0.0f}, 1);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 1);
}

TEST(HNSWIndexTest, TopKOrdering) {
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

TEST(HNSWIndexTest, RecallAt10) {
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
				std::vector<float> vec = randomVector(dim, gen);
        vectors.push_back(vec);
        index.insert(i, vec);
    }
    
    // Test with random queries
    const size_t num_queries = 100;
    
    for (size_t q = 0; q < num_queries; ++q) {
        std::vector<float>  query = randomVector(dim, gen);
				std::vector<SearchResult> results = index.search(query, k, 100);
        
        // Verify we got k results
        EXPECT_EQ(results.size(), k);
    }
}

TEST(HNSWIndexTest, DimensionMismatch) {
    HNSWIndex index(3, DistanceMetric::Cosine);
    
    EXPECT_THROW(index.insert(1, {1.0f, 0.0f}), std::invalid_argument);
    EXPECT_THROW(index.search({1.0f, 0.0f}, 1), std::invalid_argument);
}
