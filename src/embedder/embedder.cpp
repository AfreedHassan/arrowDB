#include "embedder.h"
#include "iostream"
#include <cstring>


extern "C" {
int32_t arrow_embed_init(const char *model_path, const char *tokenizer_name);
EmbeddingResult arrow_embed_text(const char *text);
void arrow_embed_free(EmbeddingResult result);
size_t arrow_embed_dimension();
DatasetLoadResult arrow_dataset_load_openwebtext(
    const char *text_path,
    const char *embeddings_path,
    size_t num_chunks,
    size_t min_length,
    size_t max_length
);
void arrow_dataset_free(DatasetLoadResult result);
DatasetLoadResult arrow_dataset_download_and_embed(
    const char *model_path,
    const char *tokenizer_name,
    size_t num_samples,
    const char *output_text_path
);
}

Embedder::Embedder(const std::string_view &modelPath,
                   const std::string_view &tokenizerName) {
  int32_t res = arrow_embed_init(modelPath.data(), tokenizerName.data());
  if (res != 0) {
    std::cerr << "Error: Failed to initialize embedder (code: " << res << ")\n";
    ok_ = false;
    return;
  }
  ok_ = true;
}

std::vector<float> Embedder::embed(const char* text) {
  EmbeddingResult res = arrow_embed_text(text);

  if (res.error_code != 0) {
    std::cerr << "Error: Failed to embed query text (code: "
              << res.error_code << ")\n";
    return {};
  }

  if (res.len != EMBEDDING_DIM) {
    std::cerr << "Error: Embedding dimension mismatch. Expected "
              <<  EMBEDDING_DIM << ", got " << res.len << "\n";
    arrow_embed_free(res);
    return {};
  }

  // Copy embedding to vector
  std::vector<float> embedding(res.data, res.data + res.len);
  arrow_embed_free(res);

  return embedding;
}

std::optional<Embedder::DatasetLoadResult> Embedder::loadOpenWebText(
    const std::string_view &textPath,
    const std::string_view &embeddingsPath,
    size_t numChunks,
    size_t minLength,
    size_t maxLength
) {
  // Call Rust FFI function to load dataset
  ::DatasetLoadResult rustResult = arrow_dataset_load_openwebtext(
      textPath.data(),
      embeddingsPath.data(),
      numChunks,
      minLength,
      maxLength
  );

  // Check for errors
  if (rustResult.error_code != 0) {
    std::cerr << "Error: Failed to load OpenWebText dataset (code: "
              << rustResult.error_code << ")\n";
    return std::nullopt;
  }

  if (rustResult.num_chunks == 0) {
    std::cerr << "Error: No chunks loaded from dataset\n";
    return std::nullopt;
  }

  // Copy chunks from C strings to std::vector<std::string>
  std::vector<std::string> chunks;
  chunks.reserve(rustResult.num_chunks);

  for (size_t i = 0; i < rustResult.num_chunks; ++i) {
    if (rustResult.chunks_ptr[i] != nullptr) {
      chunks.emplace_back(rustResult.chunks_ptr[i]);
    }
  }

  // Copy embeddings from flat C array to std::vector<std::vector<float>>
  std::vector<std::vector<float>> embeddings;
  embeddings.reserve(rustResult.num_chunks);

  for (size_t i = 0; i < rustResult.num_chunks; ++i) {
    std::vector<float> embedding(
        rustResult.embeddings_ptr + (i * rustResult.embedding_dim),
        rustResult.embeddings_ptr + ((i + 1) * rustResult.embedding_dim)
    );
    embeddings.push_back(std::move(embedding));
  }

  // Verify embedding dimension matches expected value
  if (rustResult.embedding_dim != EMBEDDING_DIM) {
    std::cerr << "Error: Embedding dimension mismatch. Expected "
              << EMBEDDING_DIM << ", got " << rustResult.embedding_dim << "\n";
    arrow_dataset_free(rustResult);
    return std::nullopt;
  }

  // Free Rust-allocated memory
  arrow_dataset_free(rustResult);

  // Return the dataset
  return DatasetLoadResult{
      .chunks = std::move(chunks),
      .embeddings = std::move(embeddings)
  };
}

std::optional<Embedder::DatasetLoadResult> Embedder::downloadAndEmbed(
    const std::string_view &modelPath,
    const std::string_view &tokenizerName,
    size_t numSamples,
    const std::string_view &outputTextPath
) {
  // Call Rust FFI function to download and embed
  ::DatasetLoadResult rustResult = arrow_dataset_download_and_embed(
      modelPath.data(),
      tokenizerName.data(),
      numSamples,
      outputTextPath.empty() ? nullptr : outputTextPath.data()
  );

  // Check for errors
  if (rustResult.error_code != 0) {
    std::cerr << "Error: Failed to download and embed dataset (code: "
              << rustResult.error_code << ")\n";
    return std::nullopt;
  }

  if (rustResult.num_chunks == 0) {
    std::cerr << "Error: No chunks downloaded\n";
    return std::nullopt;
  }

  // Copy chunks from C strings to std::vector<std::string>
  std::vector<std::string> chunks;
  chunks.reserve(rustResult.num_chunks);

  for (size_t i = 0; i < rustResult.num_chunks; ++i) {
    if (rustResult.chunks_ptr[i] != nullptr) {
      chunks.emplace_back(rustResult.chunks_ptr[i]);
    }
  }

  // Copy embeddings from flat C array to std::vector<std::vector<float>>
  std::vector<std::vector<float>> embeddings;
  embeddings.reserve(rustResult.num_chunks);

  for (size_t i = 0; i < rustResult.num_chunks; ++i) {
    std::vector<float> embedding(
        rustResult.embeddings_ptr + (i * rustResult.embedding_dim),
        rustResult.embeddings_ptr + ((i + 1) * rustResult.embedding_dim)
    );
    embeddings.push_back(std::move(embedding));
  }

  // Free Rust-allocated memory
  arrow_dataset_free(rustResult);

  // Return the dataset
  return DatasetLoadResult{
      .chunks = std::move(chunks),
      .embeddings = std::move(embeddings)
  };
}

