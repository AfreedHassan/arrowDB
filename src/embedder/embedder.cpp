#include "embedder.h"
#include "iostream"


extern "C" {
int32_t arrow_embed_init(const char *model_path, const char *tokenizer_name);
EmbeddingResult arrow_embed_text(const char *text);
void arrow_embed_free(EmbeddingResult result);
size_t arrow_embed_dimension();
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


