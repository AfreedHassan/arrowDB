#ifndef EMBEDDER_H
#define EMBEDDER_H

#pragma once
#include <vector>
#include <optional>
#include <arrow_embed.h>


class Embedder {
public:
    explicit Embedder(
    const std::string_view &modelPath = "models/all-MiniLM-L6-v2.onnx",
    const std::string_view &tokenizerName = "sentence-transformers/all-MiniLM-L6-v2"
    );

    std::vector<float> embed(const char* text);

    /// Dataset loading result containing text chunks and embeddings
    struct DatasetLoadResult {
        std::vector<std::string> chunks;
        std::vector<std::vector<float>> embeddings;
    };

    /// Load OpenWebText dataset from text and embeddings files
    /// @param textPath Path to text file (one chunk per line)
    /// @param embeddingsPath Path to binary embeddings file (float32 format)
    /// @param numChunks Number of chunks to load
    /// @param minLength Minimum text length (characters)
    /// @param maxLength Maximum text length (characters)
    /// @return Optional containing dataset or empty if failed
    std::optional<DatasetLoadResult> loadOpenWebText(
        const std::string_view &textPath,
        const std::string_view &embeddingsPath,
        size_t numChunks = 200000,
        size_t minLength = 40,
        size_t maxLength = 200
    );

    /// Download OpenWebText from HuggingFace and return embeddings
    /// @param modelPath Path to the ONNX model file
    /// @param tokenizerName HuggingFace tokenizer name
    /// @param numSamples Number of samples to download and embed
    /// @param outputTextPath Optional path to save text chunks file (like Python embed.py)
    /// @return Optional containing dataset or empty if failed
    static std::optional<DatasetLoadResult> downloadAndEmbed(
        const std::string_view &modelPath,
        const std::string_view &tokenizerName,
        size_t numSamples = 10000,
        const std::string_view &outputTextPath = ""
    );

    inline bool ok() { return ok_ ;}

    // TODO
    // std::vector<std::vector<float>> embedBatch( const std::vector<std::string>& texts);
private:
    bool ok_;
};


#endif // EMBEDDER_H

