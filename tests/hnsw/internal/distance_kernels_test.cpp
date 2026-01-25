/**
 * @file distance_kernels_test.cpp
 * @brief Tests for HNSW distance kernels (IP, L2) and batch distance functions.
 * 
 * These tests verify:
 * 1. Single-pair distance functions (distFunc)
 * 2. Batch distance functions (batchDistFunc)
 * 3. Consistency between single and batch versions
 * 4. Correctness of cosine/IP calculations with known vectors
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

#include "index/hnsw/lib.h"
#include "index/hnsw/space_ip.h"
#include "index/hnsw/space_l2.h"
#include "index/hnsw/backend_registry.h"

using namespace hnsw;

// ============================================================================
// Test Utilities
// ============================================================================

/// Compute dot product manually
float manual_dot_product(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/// Compute L2 distance manually
float manual_l2_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/// Compute IP distance (1 - dot_product) manually
float manual_ip_distance(const float* a, const float* b, size_t dim) {
    return 1.0f - manual_dot_product(a, b, dim);
}

/// L2 normalize a vector in-place
void normalize_l2(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-12f) {
        for (float& v : vec) v /= norm;
    }
}

/// Generate random normalized vector
std::vector<float> random_normalized_vector(size_t dim, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
    }
    normalize_l2(vec);
    return vec;
}

// ============================================================================
// Inner Product Space Tests
// ============================================================================

class IPDistanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_.seed(42);
    }
    std::mt19937 gen_;
};

TEST_F(IPDistanceTest, IdenticalVectorsHaveZeroDistance) {
    const size_t dim = 384;  // MiniLM dimension
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    auto vec = random_normalized_vector(dim, gen_);
    
    float dist = distFunc(vec.data(), vec.data(), dim);
    
    // For normalized identical vectors: dot = 1, distance = 1 - 1 = 0
    EXPECT_NEAR(dist, 0.0f, 1e-5f) << "Identical normalized vectors should have distance 0";
}

TEST_F(IPDistanceTest, OrthogonalVectorsHaveDistanceOne) {
    const size_t dim = 3;
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};
    
    float dist = distFunc(a.data(), b.data(), dim);
    
    // Orthogonal vectors: dot = 0, distance = 1 - 0 = 1
    EXPECT_NEAR(dist, 1.0f, 1e-5f) << "Orthogonal vectors should have distance 1";
}

TEST_F(IPDistanceTest, OppositeVectorsHaveDistanceTwo) {
    const size_t dim = 3;
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f, 0.0f};
    
    float dist = distFunc(a.data(), b.data(), dim);
    
    // Opposite vectors: dot = -1, distance = 1 - (-1) = 2
    EXPECT_NEAR(dist, 2.0f, 1e-5f) << "Opposite vectors should have distance 2";
}

TEST_F(IPDistanceTest, MatchesManualCalculation) {
    const size_t dim = 384;
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    auto a = random_normalized_vector(dim, gen_);
    auto b = random_normalized_vector(dim, gen_);
    
    float simd_dist = distFunc(a.data(), b.data(), dim);
    float manual_dist = manual_ip_distance(a.data(), b.data(), dim);
    
    EXPECT_NEAR(simd_dist, manual_dist, 1e-4f) 
        << "SIMD IP distance should match manual calculation";
}

TEST_F(IPDistanceTest, SIMDMatchesScalarForAligned16) {
    const size_t dim = 384;  // Aligned to 16
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    for (int trial = 0; trial < 100; ++trial) {
        auto a = random_normalized_vector(dim, gen_);
        auto b = random_normalized_vector(dim, gen_);
        
        float simd_dist = distFunc(a.data(), b.data(), dim);
        float scalar_dist = manual_ip_distance(a.data(), b.data(), dim);
        
        EXPECT_NEAR(simd_dist, scalar_dist, 1e-4f) 
            << "Trial " << trial << ": SIMD should match scalar";
    }
}

TEST_F(IPDistanceTest, SIMDMatchesScalarForAligned4) {
    const size_t dim = 100;  // Aligned to 4 but not 16
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    for (int trial = 0; trial < 100; ++trial) {
        auto a = random_normalized_vector(dim, gen_);
        auto b = random_normalized_vector(dim, gen_);
        
        float simd_dist = distFunc(a.data(), b.data(), dim);
        float scalar_dist = manual_ip_distance(a.data(), b.data(), dim);
        
        EXPECT_NEAR(simd_dist, scalar_dist, 1e-4f) 
            << "Trial " << trial << ": SIMD (aligned4) should match scalar";
    }
}

TEST_F(IPDistanceTest, SIMDMatchesScalarForUnaligned) {
    const size_t dim = 127;  // Not aligned
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    for (int trial = 0; trial < 100; ++trial) {
        auto a = random_normalized_vector(dim, gen_);
        auto b = random_normalized_vector(dim, gen_);
        
        float simd_dist = distFunc(a.data(), b.data(), dim);
        float scalar_dist = manual_ip_distance(a.data(), b.data(), dim);
        
        EXPECT_NEAR(simd_dist, scalar_dist, 1e-4f) 
            << "Trial " << trial << ": SIMD (unaligned) should match scalar";
    }
}

// ============================================================================
// Batch Distance Tests
// ============================================================================

class BatchDistanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_.seed(42);
    }
    std::mt19937 gen_;
};

TEST_F(BatchDistanceTest, IPBatchMatchesSingleDistFunc) {
    const size_t dim = 384;
    const size_t num_targets = 128;
    
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    auto batchDistFunc = space.getBatchDistFunc();
    
    auto query = random_normalized_vector(dim, gen_);
    
    std::vector<std::vector<float>> targets(num_targets);
    std::vector<const float*> target_ptrs(num_targets);
    for (size_t i = 0; i < num_targets; ++i) {
        targets[i] = random_normalized_vector(dim, gen_);
        target_ptrs[i] = targets[i].data();
    }
    
    std::vector<float> batch_distances(num_targets);
    batchDistFunc(query.data(), target_ptrs.data(), num_targets, dim, batch_distances.data());
    
    for (size_t i = 0; i < num_targets; ++i) {
        float single_dist = distFunc(query.data(), targets[i].data(), dim);
        EXPECT_NEAR(batch_distances[i], single_dist, 1e-5f)
            << "Batch distance[" << i << "] should match single distFunc";
    }
}

TEST_F(BatchDistanceTest, L2BatchMatchesSingleDistFunc) {
    const size_t dim = 384;
    const size_t num_targets = 128;
    
    L2Space space(dim);
    auto distFunc = space.getDistFunc();
    auto batchDistFunc = space.getBatchDistFunc();
    
    auto query = random_normalized_vector(dim, gen_);
    
    std::vector<std::vector<float>> targets(num_targets);
    std::vector<const float*> target_ptrs(num_targets);
    for (size_t i = 0; i < num_targets; ++i) {
        targets[i] = random_normalized_vector(dim, gen_);
        target_ptrs[i] = targets[i].data();
    }
    
    std::vector<float> batch_distances(num_targets);
    batchDistFunc(query.data(), target_ptrs.data(), num_targets, dim, batch_distances.data());
    
    for (size_t i = 0; i < num_targets; ++i) {
        float single_dist = distFunc(query.data(), targets[i].data(), dim);
        EXPECT_NEAR(batch_distances[i], single_dist, 1e-5f)
            << "Batch L2 distance[" << i << "] should match single distFunc";
    }
}

TEST_F(BatchDistanceTest, IPBatchWithSmallCount) {
    const size_t dim = 384;
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    auto batchDistFunc = space.getBatchDistFunc();
    
    auto query = random_normalized_vector(dim, gen_);
    
    // Test with count < 4 (below vectorization threshold)
    for (size_t count = 1; count <= 3; ++count) {
        std::vector<std::vector<float>> targets(count);
        std::vector<const float*> target_ptrs(count);
        for (size_t i = 0; i < count; ++i) {
            targets[i] = random_normalized_vector(dim, gen_);
            target_ptrs[i] = targets[i].data();
        }
        
        std::vector<float> batch_distances(count);
        batchDistFunc(query.data(), target_ptrs.data(), count, dim, batch_distances.data());
        
        for (size_t i = 0; i < count; ++i) {
            float single_dist = distFunc(query.data(), targets[i].data(), dim);
            EXPECT_NEAR(batch_distances[i], single_dist, 1e-5f)
                << "Count=" << count << ", Batch distance[" << i << "] should match";
        }
    }
}

// ============================================================================
// Specific Regression Tests
// ============================================================================

TEST_F(IPDistanceTest, RegressionCosineDistanceRange) {
    // For normalized vectors with cosine distance:
    // - Distance = 1 - cos(theta), where theta is angle between vectors
    // - Distance should be in range [0, 2]
    
    const size_t dim = 384;
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    for (int trial = 0; trial < 1000; ++trial) {
        auto a = random_normalized_vector(dim, gen_);
        auto b = random_normalized_vector(dim, gen_);
        
        float dist = distFunc(a.data(), b.data(), dim);
        
        EXPECT_GE(dist, -0.01f) << "Distance should be >= 0 (with tolerance)";
        EXPECT_LE(dist, 2.01f) << "Distance should be <= 2 (with tolerance)";
    }
}

TEST_F(IPDistanceTest, SymmetricDistance) {
    const size_t dim = 384;
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    auto a = random_normalized_vector(dim, gen_);
    auto b = random_normalized_vector(dim, gen_);
    
    float dist_ab = distFunc(a.data(), b.data(), dim);
    float dist_ba = distFunc(b.data(), a.data(), dim);
    
    EXPECT_NEAR(dist_ab, dist_ba, 1e-6f) << "Distance should be symmetric";
}

// ============================================================================
// Known Embedding Test (Simulated)
// ============================================================================

TEST_F(IPDistanceTest, SimilarTextsShouldHaveLowerDistance) {
    // Simulate what we'd expect from similar text embeddings
    // Two very similar 384-dim embeddings should have distance close to 0
    
    const size_t dim = 384;
    InnerProductSpace space(dim);
    auto distFunc = space.getDistFunc();
    
    // Create "similar" vectors (same direction with small perturbation)
    auto base = random_normalized_vector(dim, gen_);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    std::vector<float> similar(dim);
    for (size_t i = 0; i < dim; ++i) {
        similar[i] = base[i] + noise(gen_);
    }
    normalize_l2(similar);
    
    // Create "dissimilar" vector (random direction)
    auto dissimilar = random_normalized_vector(dim, gen_);
    
    float dist_similar = distFunc(base.data(), similar.data(), dim);
    float dist_dissimilar = distFunc(base.data(), dissimilar.data(), dim);
    
    EXPECT_LT(dist_similar, 0.5f) 
        << "Similar vectors should have low distance";
    EXPECT_LT(dist_similar, dist_dissimilar) 
        << "Similar vectors should have lower distance than random vectors";
}

// ============================================================================
// L2 Space Tests
// ============================================================================

class L2DistanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_.seed(42);
    }
    std::mt19937 gen_;
};

TEST_F(L2DistanceTest, IdenticalVectorsHaveZeroDistance) {
    const size_t dim = 384;
    L2Space space(dim);
    auto distFunc = space.getDistFunc();
    
    auto vec = random_normalized_vector(dim, gen_);
    
    float dist = distFunc(vec.data(), vec.data(), dim);
    
    EXPECT_NEAR(dist, 0.0f, 1e-5f) << "Identical vectors should have L2 distance 0";
}

TEST_F(L2DistanceTest, MatchesManualCalculation) {
    const size_t dim = 384;
    L2Space space(dim);
    auto distFunc = space.getDistFunc();
    
    auto a = random_normalized_vector(dim, gen_);
    auto b = random_normalized_vector(dim, gen_);
    
    float simd_dist = distFunc(a.data(), b.data(), dim);
    float manual_dist = manual_l2_distance(a.data(), b.data(), dim);
    
    EXPECT_NEAR(simd_dist, manual_dist, 1e-4f) 
        << "SIMD L2 distance should match manual calculation";
}

TEST_F(L2DistanceTest, UnitVectorOrthogonalL2) {
    // For two orthogonal unit vectors: ||a - b||^2 = ||a||^2 + ||b||^2 = 2
    const size_t dim = 3;
    L2Space space(dim);
    auto distFunc = space.getDistFunc();
    
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};
    
    float dist = distFunc(a.data(), b.data(), dim);
    
    EXPECT_NEAR(dist, 2.0f, 1e-5f) 
        << "Orthogonal unit vectors should have L2 distance 2";
}

// ============================================================================
// Performance Sanity Check
// ============================================================================

TEST_F(BatchDistanceTest, BatchIsReasonablyFast) {
    const size_t dim = 384;
    const size_t num_targets = 1000;
    
    InnerProductSpace space(dim);
    auto batchDistFunc = space.getBatchDistFunc();
    
    auto query = random_normalized_vector(dim, gen_);
    
    std::vector<std::vector<float>> targets(num_targets);
    std::vector<const float*> target_ptrs(num_targets);
    for (size_t i = 0; i < num_targets; ++i) {
        targets[i] = random_normalized_vector(dim, gen_);
        target_ptrs[i] = targets[i].data();
    }
    
    std::vector<float> distances(num_targets);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        batchDistFunc(query.data(), target_ptrs.data(), num_targets, dim, distances.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double us_per_batch = elapsed.count() / 100.0;
    
    // Should be able to compute 1000 distances in < 1ms on modern hardware
    EXPECT_LT(us_per_batch, 1000.0) 
        << "Batch distance computation should be fast (< 1ms for 1000 distances)";
    
    std::cout << "[INFO] Batch IP distance: " << us_per_batch << " µs for " 
              << num_targets << " distances (" << dim << "D)" << std::endl;
}
