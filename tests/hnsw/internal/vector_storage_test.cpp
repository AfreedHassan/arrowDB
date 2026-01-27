/**
 * @file vector_storage_test.cpp
 * @brief Test that vectors are stored and retrieved correctly from HNSW
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

#include "internal/hnsw_index.h"
#include "index/hnsw/hnsw.cpp"
#include "index/hnsw/space_ip.h"

using namespace arrow;

// Generate random normalized vector
std::vector<float> generate_normalized_vector(size_t dim, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] /= norm;
    }
    return vec;
}

// ============================================================================
// Direct HNSW storage test - check memory layout
// ============================================================================

TEST(VectorStorageTest, VectorsStoredCorrectly) {
    const size_t dim = 384;
    const size_t num_vectors = 100;
    
    hnsw::InnerProductSpace space(dim);
    hnsw::HierarchicalNSW<float> index(&space, num_vectors, 16, 200, 42);
    
    std::mt19937 gen(42);
    std::vector<std::vector<float>> original_vectors;
    
    // Insert vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        auto vec = generate_normalized_vector(dim, gen);
        original_vectors.push_back(vec);
        index.addPoint(vec.data(), static_cast<hnsw::label_t>(i + 1));
    }
    
    ASSERT_EQ(index.size(), num_vectors);
    
    // Verify each stored vector matches the original
    for (size_t i = 0; i < num_vectors; ++i) {
        // Search for the exact vector - should return itself as top result
        auto results = index.searchKnn(original_vectors[i].data(), 1);
        
        ASSERT_FALSE(results.empty()) << "Search should return at least one result";
        
        auto [dist, label] = results.top();
        
        // The top result should be itself (label = i + 1)
        EXPECT_EQ(label, static_cast<hnsw::label_t>(i + 1)) 
            << "Vector " << i << " should find itself as closest neighbor";
        
        // Distance to itself should be ~0
        EXPECT_NEAR(dist, 0.0f, 0.01f) 
            << "Distance to self should be ~0, got " << dist;
    }
}

TEST(VectorStorageTest, DistancesMatchExpected) {
    const size_t dim = 384;
    
    hnsw::InnerProductSpace space(dim);
    hnsw::HierarchicalNSW<float> index(&space, 10, 16, 200, 42);
    auto distFunc = space.getDistFunc();
    
    std::mt19937 gen(123);
    
    auto vec1 = generate_normalized_vector(dim, gen);
    auto vec2 = generate_normalized_vector(dim, gen);
    auto vec3 = generate_normalized_vector(dim, gen);
    
    // Compute expected distances BEFORE insertion
    float expected_dist_12 = distFunc(vec1.data(), vec2.data(), dim);
    float expected_dist_13 = distFunc(vec1.data(), vec3.data(), dim);
    float expected_dist_23 = distFunc(vec2.data(), vec3.data(), dim);
    
    std::cout << "Expected distances (before insertion):" << std::endl;
    std::cout << "  vec1 ↔ vec2: " << expected_dist_12 << std::endl;
    std::cout << "  vec1 ↔ vec3: " << expected_dist_13 << std::endl;
    std::cout << "  vec2 ↔ vec3: " << expected_dist_23 << std::endl;
    
    // Insert vectors
    index.addPoint(vec1.data(), 1);
    index.addPoint(vec2.data(), 2);
    index.addPoint(vec3.data(), 3);
    
    // Search from vec1
    auto results = index.searchKnn(vec1.data(), 3);
    
    std::cout << "\nSearch results from vec1:" << std::endl;
    while (!results.empty()) {
        auto [dist, label] = results.top();
        results.pop();
        std::cout << "  Label " << label << ": distance = " << dist << std::endl;
        
        // Verify distances match expected
        if (label == 1) {
            EXPECT_NEAR(dist, 0.0f, 0.01f) << "Self distance should be 0";
        } else if (label == 2) {
            EXPECT_NEAR(dist, expected_dist_12, 0.01f) << "Distance to vec2 mismatch";
        } else if (label == 3) {
            EXPECT_NEAR(dist, expected_dist_13, 0.01f) << "Distance to vec3 mismatch";
        }
    }
}

// ============================================================================
// Test via HNSWIndex wrapper (what Collection uses)
// ============================================================================

TEST(VectorStorageTest, HNSWIndexStoresCorrectly) {
    const size_t dim = 384;
    const size_t num_vectors = 50;
    
    HNSWIndex index(dim, Space::Cosine, {.M = 16, .efConstruction = 200});
    
    std::mt19937 gen(999);
    std::vector<std::vector<float>> vectors;
    
    for (size_t i = 0; i < num_vectors; ++i) {
        auto vec = generate_normalized_vector(dim, gen);
        vectors.push_back(vec);
        index.insert(i + 1, vec);
    }
    
    ASSERT_EQ(index.size(), num_vectors);
    
    // Verify self-search works correctly
    size_t correct = 0;
    for (size_t i = 0; i < num_vectors; ++i) {
        auto results = index.search(vectors[i], 1, 100);
        if (!results.empty() && results[0].id == static_cast<InternalID>(i + 1)) {
            correct++;
        }
    }
    
    std::cout << "Self-search accuracy: " << correct << "/" << num_vectors << std::endl;
    EXPECT_EQ(correct, num_vectors) << "All vectors should find themselves as closest";
}

// ============================================================================
// Specific regression test with known values
// ============================================================================

TEST(VectorStorageTest, KnownVectorDistances) {
    const size_t dim = 3;  // Simple case for verification
    
    hnsw::InnerProductSpace space(dim);
    hnsw::HierarchicalNSW<float> index(&space, 10, 4, 10, 42);
    auto distFunc = space.getDistFunc();
    
    // Normalized unit vectors along axes
    std::vector<float> vec_x = {1.0f, 0.0f, 0.0f};
    std::vector<float> vec_y = {0.0f, 1.0f, 0.0f};
    std::vector<float> vec_xy = {0.7071f, 0.7071f, 0.0f};  // 45 degrees
    
    // Expected distances (1 - dot_product):
    // x ↔ y: 1 - 0 = 1.0
    // x ↔ xy: 1 - 0.7071 = 0.2929
    float dist_x_y = distFunc(vec_x.data(), vec_y.data(), dim);
    float dist_x_xy = distFunc(vec_x.data(), vec_xy.data(), dim);
    
    std::cout << "Distance x ↔ y: " << dist_x_y << " (expected 1.0)" << std::endl;
    std::cout << "Distance x ↔ xy: " << dist_x_xy << " (expected 0.2929)" << std::endl;
    
    EXPECT_NEAR(dist_x_y, 1.0f, 0.01f);
    EXPECT_NEAR(dist_x_xy, 0.2929f, 0.01f);
    
    // Insert and search
    index.addPoint(vec_x.data(), 1);
    index.addPoint(vec_y.data(), 2);
    index.addPoint(vec_xy.data(), 3);
    
    // Query with vec_x - should rank: 1 (self), 3 (xy), 2 (y)
    auto results = index.searchKnn(vec_x.data(), 3);
    
    std::vector<std::pair<float, hnsw::label_t>> ordered;
    while (!results.empty()) {
        ordered.push_back(results.top());
        results.pop();
    }
    std::reverse(ordered.begin(), ordered.end());
    
    std::cout << "\nRanking from vec_x:" << std::endl;
    for (const auto& [dist, label] : ordered) {
        std::cout << "  Label " << label << ": distance = " << dist << std::endl;
    }
    
    ASSERT_EQ(ordered.size(), 3);
    EXPECT_EQ(ordered[0].second, 1) << "Self should be first";
    EXPECT_EQ(ordered[1].second, 3) << "vec_xy should be second (closer than vec_y)";
    EXPECT_EQ(ordered[2].second, 2) << "vec_y should be last";
}
