#include <gtest/gtest.h>
#include "arrow/collection.h"
#include "embedder/embedder.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>

using namespace arrow;
namespace fs = std::filesystem;

class DatasetTest : public ::testing::Test {
protected:
    std::string test_dir;

    void SetUp() override {
        // Create temporary test directory
        test_dir = "/tmp/arrow_dataset_test_" + std::to_string(getpid());
        fs::create_directories(test_dir);
    }

    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    /// Create a test text file with sample chunks
    std::string createTestTextFile(
        const std::vector<std::string>& chunks,
        const std::string& filename = "test_chunks.txt"
    ) {
        std::string path = test_dir + "/" + filename;
        std::ofstream file(path);
        for (const auto& chunk : chunks) {
            file << chunk << "\n";
        }
        file.close();
        return path;
    }

    /// Create a test embeddings file with random float32 vectors
    std::string createTestEmbeddingsFile(
        size_t num_embeddings,
        size_t embedding_dim = 384,
        const std::string& filename = "test_embeddings.bin"
    ) {
        std::string path = test_dir + "/" + filename;
        std::ofstream file(path, std::ios::binary);

        // Write embeddings as little-endian float32
        for (size_t i = 0; i < num_embeddings; ++i) {
            for (size_t j = 0; j < embedding_dim; ++j) {
                // Create simple deterministic embeddings for testing
                float value = static_cast<float>(i * embedding_dim + j) / 1000.0f;
                file.write(reinterpret_cast<const char*>(&value), sizeof(float));
            }
        }
        file.close();
        return path;
    }
};

/// Test that dataset loading function exists and is callable
TEST_F(DatasetTest, LoadOpenWebTextBasic) {
    Embedder embedder(ARROW_EMBEDDING_MODEL_DIR "/all-MiniLM-L6-v2.onnx");
    if (!embedder.ok()) {
        GTEST_SKIP() << "Embedder failed to initialize";
    }

    // Create small test dataset
    std::vector<std::string> chunks = {
        "This is a test chunk about machine learning and artificial intelligence",
        "Another sample text with sufficient length for testing purposes here",
        "One more example chunk that meets the minimum character requirement"
    };

    std::string text_path = createTestTextFile(chunks);
    std::string emb_path = createTestEmbeddingsFile(chunks.size(), 384);

    // Load dataset
    auto result = embedder.loadOpenWebText(
        text_path,
        emb_path,
        chunks.size(),  // numChunks = 3
        40,             // minLength
        200             // maxLength
    );

    // Verify result
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->chunks.size(), 3);
    EXPECT_EQ(result->embeddings.size(), 3);
    EXPECT_EQ(result->embeddings[0].size(), 384);

    // Verify chunk content
    EXPECT_EQ(result->chunks[0], chunks[0]);
    EXPECT_EQ(result->chunks[1], chunks[1]);
    EXPECT_EQ(result->chunks[2], chunks[2]);
}

/// Test that dataset loads only chunks within length bounds
TEST_F(DatasetTest, LengthFiltering) {
    Embedder embedder(ARROW_EMBEDDING_MODEL_DIR "/all-MiniLM-L6-v2.onnx");
    if (!embedder.ok()) {
        GTEST_SKIP() << "Embedder failed to initialize";
    }

    // Mix of short, valid, and long chunks
    std::vector<std::string> chunks = {
        "Too short",  // Should be filtered (< 40 chars)
        "This is a text chunk that is long enough to meet the minimum length requirement",  // Valid
        "Another valid chunk with sufficient length to pass the minimum character count test",  // Valid
        "A" // Way too short
    };

    std::string text_path = createTestTextFile(chunks);
    std::string emb_path = createTestEmbeddingsFile(chunks.size(), 384);

    // Load with min length = 40
    auto result = embedder.loadOpenWebText(
        text_path,
        emb_path,
        chunks.size(),
        40,      // minLength
        500      // maxLength
    );

    ASSERT_TRUE(result.has_value());
    // Should load only the valid chunks (2 chunks that meet the length requirement)
    EXPECT_EQ(result->chunks.size(), 2);
    EXPECT_EQ(result->embeddings.size(), 2);
}

/// Test error handling for missing files
TEST_F(DatasetTest, MissingFiles) {
    Embedder embedder(ARROW_EMBEDDING_MODEL_DIR "/all-MiniLM-L6-v2.onnx");
    if (!embedder.ok()) {
        GTEST_SKIP() << "Embedder failed to initialize";
    }

    // Try to load from non-existent files
    auto result = embedder.loadOpenWebText(
        "/tmp/nonexistent_text.txt",
        "/tmp/nonexistent_embeddings.bin",
        100,
        40,
        200
    );

    // Should fail gracefully
    EXPECT_FALSE(result.has_value());
}

/// Test embedding dimension validation
TEST_F(DatasetTest, EmbeddingDimension) {
    Embedder embedder(ARROW_EMBEDDING_MODEL_DIR "/all-MiniLM-L6-v2.onnx");
    if (!embedder.ok()) {
        GTEST_SKIP() << "Embedder failed to initialize";
    }

    std::vector<std::string> chunks = {
        "This is a test chunk about machine learning and artificial intelligence",
        "Another sample text with sufficient length for testing purposes here"
    };

    std::string text_path = createTestTextFile(chunks);
    // Create embeddings with CORRECT dimension (384)
    std::string emb_path = createTestEmbeddingsFile(chunks.size(), 384);

    auto result = embedder.loadOpenWebText(
        text_path,
        emb_path,
        chunks.size(),
        40,
        200
    );

    ASSERT_TRUE(result.has_value());
    // Verify all embeddings have correct dimension
    for (const auto& emb : result->embeddings) {
        EXPECT_EQ(emb.size(), 384);
    }
}

/// Test loading larger dataset
TEST_F(DatasetTest, DISABLED_LargeDatasetLoad) {
    Embedder embedder(ARROW_EMBEDDING_MODEL_DIR "/all-MiniLM-L6-v2.onnx");
    if (!embedder.ok()) {
        GTEST_SKIP() << "Embedder failed to initialize";
    }

    // Create larger test dataset (1000 chunks)
    std::vector<std::string> chunks;
    for (int i = 0; i < 1000; ++i) {
        chunks.push_back(
            "This is test chunk number " + std::to_string(i) +
            " with sufficient length to pass the minimum character requirement for loading"
        );
    }

    std::string text_path = createTestTextFile(chunks);
    std::string emb_path = createTestEmbeddingsFile(chunks.size(), 384);

    auto result = embedder.loadOpenWebText(
        text_path,
        emb_path,
        chunks.size(),
        40,
        200
    );

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->chunks.size(), 1000);
    EXPECT_EQ(result->embeddings.size(), 1000);

    // Verify all embeddings have correct dimension
    for (const auto& emb : result->embeddings) {
        EXPECT_EQ(emb.size(), 384);
    }
}
