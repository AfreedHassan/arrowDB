#ifndef EMBEDDER_H
#define EMBEDDER_H

#pragma once
#include <vector>
#include <arrow_embed.h>


class Embedder {
public:
    explicit Embedder(
    const std::string_view &modelPath = "models/all-MiniLM-L6-v2.onnx",
    const std::string_view &tokenizerName = "sentence-transformers/all-MiniLM-L6-v2"
    );

    std::vector<float> embed(const char* text);

    inline bool ok() { return ok_ ;}

    // TODO
    // std::vector<std::vector<float>> embedBatch( const std::vector<std::string>& texts);
private:
    bool ok_; 
};


#endif // EMBEDDER_H

