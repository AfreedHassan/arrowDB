#include <gtest/gtest.h>
#include "arrow/collection.h"
#include "arrow/hnsw_index.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace arrow;

namespace arrow {
namespace testing {

/**
 * Load SIFT dataset vectors from binary format (.fvecs files).
 * 
 * SIFT datasets are available from: http://corpus-texmex.irisa.fr/
 * Format: Each vector is stored as:
 *   - 4 bytes: dimension (int)
 *   - dimension * 4 bytes: vector components (float32)
 * 
 * @param filepath Path to .fvecs file
 * @param max_vectors Maximum number of vectors to load (0 = load all)
 * @return Vector of vectors, each inner vector is normalized float32
 */
std::vector<std::vector<float>> LoadSIFTVectors(
    const std::string& filepath,
    size_t max_vectors = 0
) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SIFT file: " + filepath);
    }

    std::vector<std::vector<float>> vectors;
    
    // Read first vector to get dimension
    int32_t dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (file.gcount() != sizeof(int32_t)) {
        throw std::runtime_error("Failed to read dimension from SIFT file");
    }
    
    file.seekg(0); // Reset to beginning
    
    size_t count = 0;
    while (file.good() && (max_vectors == 0 || count < max_vectors)) {
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (file.gcount() != sizeof(int32_t)) {
            break; // End of file
        }
        
        if (d != dim) {
            throw std::runtime_error("Inconsistent dimension in SIFT file");
        }
        
        // Read vector components (float32)
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (file.gcount() != static_cast<std::streamsize>(dim * sizeof(float))) {
            break; // Incomplete vector
        }

        // NOTE: SIFT vectors not normalized as ground truth uses L2 distance on raw vectors
        vectors.push_back(std::move(vec));
        count++;
    }
    
    return vectors;
}

/**
 * Load SIFT ground truth from binary format (.ivecs files).
 * 
 * Format: Each query's ground truth is stored as:
 *   - 4 bytes: k (int, number of nearest neighbors)
 *   - k * 4 bytes: vector IDs (int)
 * 
 * @param filepath Path to .ivecs file
 * @param num_queries Number of queries
 * @param k Number of nearest neighbors per query
 * @return Vector of query results, each containing k vector IDs
 */
std::vector<std::vector<VectorID>> LoadSIFTGroundTruth(
    const std::string& filepath,
    size_t num_queries,
    size_t k
) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ground truth file: " + filepath);
    }

    std::vector<std::vector<VectorID>> ground_truth;
    ground_truth.reserve(num_queries);
    
    for (size_t q = 0; q < num_queries; ++q) {
        int32_t k_read;
        file.read(reinterpret_cast<char*>(&k_read), sizeof(int32_t));
        if (file.gcount() != sizeof(int32_t)) {
            throw std::runtime_error("Failed to read k from ground truth file");
        }
        
        size_t k_actual = std::min(static_cast<size_t>(k_read), k);
        std::vector<int32_t> ids(k_read);
        file.read(reinterpret_cast<char*>(ids.data()), k_read * sizeof(int32_t));
        if (file.gcount() != static_cast<std::streamsize>(k_read * sizeof(int32_t))) {
            throw std::runtime_error("Failed to read ground truth IDs");
        }
        
        std::vector<VectorID> query_gt;
        query_gt.reserve(k_actual);
        for (size_t i = 0; i < k_actual; ++i) {
            query_gt.push_back(static_cast<VectorID>(ids[i]));
        }
        ground_truth.push_back(std::move(query_gt));
    }
    
    return ground_truth;
}

/**
 * Calculate recall@k for SIFT dataset.
 * 
 * @param ground_truth Ground truth results for queries
 * @param results Approximate search results
 * @param k Number of neighbors to consider
 * @return Average recall@k across all queries
 */
double CalculateSIFTRecall(
    const std::vector<std::vector<VectorID>>& groundTruthFull,
    const std::vector<std::vector<arrow::SearchResult>>& results,
    size_t k
) {
    if (groundTruthFull.size() != results.size()) {
        throw std::runtime_error("Ground truth and results size mismatch");
    }
    
    if (groundTruthFull.empty()) {
        return 0.0;
    }
    
    double totalRecall = 0.0;
    for (size_t q = 0; q < groundTruthFull.size(); ++q) {
        const auto& gt = groundTruthFull[q];
        const auto& res = results[q];
        
        std::unordered_set<VectorID> gtSet;
        size_t kActual = std::min(k, gt.size());
        for (size_t i = 0; i < kActual; ++i) {
            gtSet.insert(gt[i]);
        }
        size_t found = 0;
        size_t resK = std::min(k, res.size());
        for (size_t i = 0; i < resK; ++i) {
            if (gtSet.count(res[i].id) > 0) {
                found++;
            }
        }
        
        totalRecall += static_cast<double>(found) / kActual;
    }
    
    return totalRecall / groundTruthFull.size();
}

} // namespace testing
} // namespace arrow

using arrow::testing::LoadSIFTVectors;
using arrow::testing::LoadSIFTGroundTruth;
using arrow::testing::CalculateSIFTRecall;

// SIFT dataset paths (update these to point to your SIFT dataset)
const std::string SIFT_BASE_PATH = "../data/sift/";
const std::string SIFT1M_VECTORS = SIFT_BASE_PATH + "sift_base.fvecs";  // Note: .fvecs for float, .bvecs for byte
const std::string SIFT1M_QUERIES = SIFT_BASE_PATH + "sift_query.fvecs";
const std::string SIFT1M_GROUND_TRUTH = SIFT_BASE_PATH + "sift_groundtruth.ivecs";

// Helper to check if file exists
bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

// Test SIFT1M dataset (if available)
TEST(SIFTTest, DISABLED_SIFT1M_Recall) {
    // Skip if dataset not available
    if (!fileExists(SIFT1M_VECTORS)) {
        GTEST_SKIP() << "SIFT1M dataset not found at " << SIFT1M_VECTORS;
    }
    
    std::cout << "Loading SIFT1M vectors..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Load all 1M vectors (0 = load all)
    auto vectors = LoadSIFTVectors(SIFT1M_VECTORS, 0);
    
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Loaded " << vectors.size() << " vectors in " << load_time << "ms" << std::endl;
    
    // Create collection with optimized config for 1M vectors
    // NOTE: SIFT ground truth uses L2 distance, so we must use L2 here too
    CollectionConfig collectionCfg("sift1m", 128, DistanceMetric::L2, DataType::Float16);
    HNSWConfig hnswCfg;
    hnswCfg.M = 64;
    hnswCfg.efConstruction = 200;
    hnswCfg.maxElements = vectors.size();
    
    Collection collection(collectionCfg, hnswCfg);
    
    // Insert vectors
    std::cout << "Building HNSW index..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < vectors.size(); ++i) {
        collection.insert(i, vectors[i]);
    }
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Built index in " << build_time << "ms" << std::endl;
    
    // Load queries and ground truth
    if (fileExists(SIFT1M_QUERIES) && fileExists(SIFT1M_GROUND_TRUTH)) {
        auto queries = LoadSIFTVectors(SIFT1M_QUERIES, 1000); // Load 1000 queries
        auto groundTruthFull = LoadSIFTGroundTruth(SIFT1M_GROUND_TRUTH, queries.size(), 100);
        
        // Run searches
        std::cout << "Running searches..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<SearchResult>> results;
        results.reserve(queries.size());
        
        // For L2 distance with 1M vectors, use EF=400 for >90% recall@100
        for (const auto& query : queries) {
            results.push_back(collection.search(query, 100, 400));
        }
        
        auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Completed " << queries.size() << " searches in " << search_time << "ms" << std::endl;
        std::cout << "Average search time: " << (search_time / queries.size()) << "ms" << std::endl;
        
        // Calculate recall
        double recall = CalculateSIFTRecall(groundTruthFull, results, 100);
        std::cout << "Recall@100: " << (recall * 100.0) << "%" << std::endl;
        
        // Expect >90% recall for 1M vectors with M=64, EF=400
        EXPECT_GT(recall, 0.90);
    }
}

// Benchmark SIFT dataset loading and indexing
TEST(SIFTTest, DISABLED_SIFT_Performance) {
    if (!fileExists(SIFT1M_VECTORS)) {
        GTEST_SKIP() << "SIFT dataset not found";
    }
    
    // Test with different dataset sizes
    std::vector<size_t> sizes = {10000, 100000, 1000000};
    
    for (size_t size : sizes) {
        std::cout << "\n=== Testing with " << size << " vectors ===" << std::endl;
        
        auto vectors = LoadSIFTVectors(SIFT1M_VECTORS, size);
        
        // SIFT uses L2 distance (ground truth is computed with L2)
        CollectionConfig cfg("sift_bench", 128, DistanceMetric::L2, DataType::Float16);
        HNSWConfig hnsw_cfg;
        hnsw_cfg.M = 64;
        hnsw_cfg.maxElements = vectors.size();
        
        Collection collection(cfg, hnsw_cfg);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < vectors.size(); ++i) {
            collection.insert(i, vectors[i]);
        }
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "Build time: " << build_time << "ms" << std::endl;
        std::cout << "Throughput: " << (vectors.size() * 1000.0 / build_time) << " vectors/sec" << std::endl;
        
        // Test search performance
        if (!vectors.empty()) {
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 100; ++i) {
                collection.search(vectors[i % vectors.size()], 10, 200);
            }
            auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Search time (100 queries): " << search_time << "ms" << std::endl;
            std::cout << "Average search time: " << (search_time / 100.0) << "ms" << std::endl;
        }
    }
}
