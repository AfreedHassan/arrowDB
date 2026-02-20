/**
 * @file embedding_debug_test.cpp  
 * @brief Debug test to compare embeddings of specific texts
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#include "embedder/embedder.h"

// Compute cosine similarity directly (for normalized vectors = dot product)
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

// Compute L2 norm
float l2_norm(const std::vector<float>& v) {
    float sum = 0.0f;
    for (float x : v) sum += x * x;
    return std::sqrt(sum);
}

TEST(DISABLED_EmbeddingDebug, SciQQueryVsDocuments) {
    Embedder embedder(ARROW_EMBEDDING_MODEL_DIR "/all-MiniLM-L6-v2.onnx",
                      "sentence-transformers/all-MiniLM-L6-v2");
    ASSERT_TRUE(embedder.ok()) << "Failed to init embedder";

    // The exact query from your test
    const char* query = "What kind of a reaction occurs when a substance reacts quickly with oxygen?";
    
    // Your result #1 (ranked highest - the "wrong" one)
    const char* doc_hydrocarbon = "Combustion reactions involve the reaction of a hydrocarbon with oxygen gas to produce water and carbon dioxide.";
    
    // Expected result (Chroma's #1)
    const char* doc_expected = "A combustion reaction occurs when a substance reacts quickly with oxygen (O 2 ). You can see an example of a combustion reaction in Figure below . Combustion is commonly called burning. The substance that burns is usually referred to as fuel. The products of a combustion reaction include carbon dioxide (CO 2 ) and water (H 2 O). The reaction typically gives off heat and light as well. The general equation for a combustion reaction can be represented by:.";

    auto emb_query = embedder.embed(query);
    auto emb_hydrocarbon = embedder.embed(doc_hydrocarbon);
    auto emb_expected = embedder.embed(doc_expected);

    ASSERT_EQ(emb_query.size(), 384);
    ASSERT_EQ(emb_hydrocarbon.size(), 384);
    ASSERT_EQ(emb_expected.size(), 384);

    // Check if embeddings are normalized
    float norm_query = l2_norm(emb_query);
    float norm_hydrocarbon = l2_norm(emb_hydrocarbon);
    float norm_expected = l2_norm(emb_expected);

    std::cout << "\n=== EMBEDDING ANALYSIS ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "L2 Norms (should be ~1.0 if normalized):" << std::endl;
    std::cout << "  Query:       " << norm_query << std::endl;
    std::cout << "  Hydrocarbon: " << norm_hydrocarbon << std::endl;
    std::cout << "  Expected:    " << norm_expected << std::endl;

    EXPECT_NEAR(norm_query, 1.0f, 0.01f) << "Query embedding should be normalized";
    EXPECT_NEAR(norm_hydrocarbon, 1.0f, 0.01f) << "Hydrocarbon embedding should be normalized";
    EXPECT_NEAR(norm_expected, 1.0f, 0.01f) << "Expected embedding should be normalized";

    // Compute similarities
    float sim_hydrocarbon = cosine_similarity(emb_query, emb_hydrocarbon);
    float sim_expected = cosine_similarity(emb_query, emb_expected);

    // Compute distances (1 - similarity, what HNSW stores)
    float dist_hydrocarbon = 1.0f - sim_hydrocarbon;
    float dist_expected = 1.0f - sim_expected;

    std::cout << "\nCosine Similarities (higher = more similar):" << std::endl;
    std::cout << "  Query ↔ Hydrocarbon: " << sim_hydrocarbon << std::endl;
    std::cout << "  Query ↔ Expected:    " << sim_expected << std::endl;

    std::cout << "\nIP Distances (lower = more similar, what HNSW uses):" << std::endl;
    std::cout << "  Query ↔ Hydrocarbon: " << dist_hydrocarbon << std::endl;
    std::cout << "  Query ↔ Expected:    " << dist_expected << std::endl;

    std::cout << "\nScores (negated distance, what you see in results):" << std::endl;
    std::cout << "  Query ↔ Hydrocarbon: " << -dist_hydrocarbon << std::endl;
    std::cout << "  Query ↔ Expected:    " << -dist_expected << std::endl;

    // Print first few embedding dimensions for comparison
    std::cout << "\nFirst 5 dimensions of each embedding:" << std::endl;
    std::cout << "  Query:       [";
    for (int i = 0; i < 5; ++i) std::cout << emb_query[i] << (i < 4 ? ", " : "");
    std::cout << "]" << std::endl;
    
    std::cout << "  Hydrocarbon: [";
    for (int i = 0; i < 5; ++i) std::cout << emb_hydrocarbon[i] << (i < 4 ? ", " : "");
    std::cout << "]" << std::endl;
    
    std::cout << "  Expected:    [";
    for (int i = 0; i < 5; ++i) std::cout << emb_expected[i] << (i < 4 ? ", " : "");
    std::cout << "]" << std::endl;

    std::cout << "\n=== VERDICT ===" << std::endl;
    if (sim_expected > sim_hydrocarbon) {
        std::cout << "✅ Expected doc IS more similar (as Chroma reports)" << std::endl;
        std::cout << "   → Bug is likely in HNSW search/ranking" << std::endl;
    } else {
        std::cout << "❌ Hydrocarbon doc is more similar (our result is correct!)" << std::endl;
        std::cout << "   → Difference is in embedding model/tokenization" << std::endl;
    }

    // The key assertion: which one SHOULD be more similar?
    // We expect the expected doc to be more similar since it contains verbatim query text
    // But let's see what our embedder actually produces
}
