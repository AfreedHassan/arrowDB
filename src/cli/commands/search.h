// Copyright 2025 ArrowDB
#ifndef ARROW_CLI_COMMANDS_SEARCH_H
#define ARROW_CLI_COMMANDS_SEARCH_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "arrow/arrow.h"
#include "embedder/embedder.h"

namespace arrow::cli {

/// Helper function to read a specific line from a text file by line number.
inline std::string getLineFromFile(const std::string& filePath, size_t lineNumber) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    return "ERROR: Could not open file";
  }

  std::string line;
  for (size_t i = 0; i <= lineNumber; ++i) {
    if (!std::getline(file, line)) {
      return "ERROR: Line not found";
    }
  }

  file.close();
  return line;
}

/// Search with a pre-computed query vector from file.
///
/// @param collectionPath Path to the collection directory
/// @param queryPath Path to binary query vector file
/// @param textPath Path to text file for displaying results
/// @param k Number of results to return
/// @param ef Search beam width
inline void searchWithQueryFile(const std::string& collectionPath,
                                const std::string& queryPath,
                                const std::string& textPath,
                                uint32_t k = 10,
                                uint32_t ef = 200) {
  // Load the collection
  auto resultOrError = Collection::load(collectionPath);
  if (!resultOrError.ok()) {
    std::cerr << "Error loading collection: "
              << resultOrError.status().message() << "\n";
    return;
  }

  Collection& collection = resultOrError.value();
  std::cout << "Loaded collection: " << collection.name() << "\n";
  std::cout << "  Dimensions: " << collection.dimension() << "\n";
  std::cout << "  Total vectors: " << collection.size() << "\n\n";

  // Read query vector from binary file
  std::ifstream queryFile(queryPath, std::ios::binary);
  if (!queryFile.is_open()) {
    std::cerr << "Error: Failed to open query file: " << queryPath << "\n";
    return;
  }

  std::vector<float> query(collection.dimension());
  queryFile.read(reinterpret_cast<char*>(query.data()),
                 collection.dimension() * sizeof(float));

  if (queryFile.gcount() != static_cast<std::streamsize>(
      collection.dimension() * sizeof(float))) {
    std::cerr << "Error: Query vector has incorrect size. Expected "
              << collection.dimension() << " floats, got "
              << (queryFile.gcount() / sizeof(float)) << "\n";
    return;
  }

  queryFile.close();
  std::cout << "Loaded query vector from: " << queryPath << "\n";
  std::cout << "  Searching for " << k << " nearest neighbors...\n\n";

  // Perform search
  auto searchResults = collection.search(query, k, ef);

  std::ofstream outputFile("output.txt");
  std::cout << "Search Results:\n";
  std::cout << std::string(80, '=') << "\n";
  for (size_t i = 0; i < searchResults.size(); ++i) {
    const auto& sr = searchResults[i];
    std::string text = getLineFromFile(textPath, sr.id);
    outputFile << text << "\n";

    std::cout << (i + 1) << ". [Score: " << sr.score << "] "
              << text << "\n";
  }
  std::cout << std::string(80, '=') << "\n";
  outputFile.close();
}

/// Search with text query using embedder.
///
/// @param queryText The text query to embed and search
/// @param collectionPath Path to the collection directory
/// @param textPath Path to text file for displaying results
/// @param modelPath Path to the ONNX embedding model
/// @param k Number of results to return
/// @param ef Search beam width
inline void searchWithText(const std::string& queryText,
                           const std::string& collectionPath = "owt_collection",
                           const std::string& textPath = "openwebtext.txt",
                           const std::string& modelPath = "",
                           uint32_t k = 10,
                           uint32_t ef = 200) {
  // Initialize embedder
  Embedder embedder(modelPath);
  if (!embedder.ok()) return;

  // Load collection
  auto resultOrError = Collection::load(collectionPath);
  if (!resultOrError.ok()) {
    std::cerr << "Error loading collection: "
              << resultOrError.status().message() << "\n";
    return;
  }

  Collection& collection = resultOrError.value();
  std::cout << "Loaded collection: " << collection.name() << "\n";
  std::cout << "  Dimensions: " << collection.dimension() << "\n";
  std::cout << "  Total vectors: " << collection.size() << "\n\n";

  // Embed query text
  std::cout << "Embedding query: \"" << queryText << "\"\n";
  std::vector<float> query = embedder.embed(queryText.c_str());

  if (query.empty()) return;

  std::cout << "Query embedded successfully\n";
  std::cout << "Searching for " << k << " nearest neighbors...\n\n";

  // Perform search
  auto searchResults = collection.search(query, k, ef);

  std::cout << "Search Results:\n";
  std::cout << std::string(80, '=') << "\n";
  for (size_t i = 0; i < searchResults.size(); ++i) {
    const auto& sr = searchResults[i];
    std::string text = getLineFromFile(textPath, sr.id);

    std::cout << (i + 1) << ". [Score: " << sr.score << "] "
              << text << "\n";
  }
  std::cout << std::string(80, '=') << "\n";
}

} // namespace arrow::cli

#endif // ARROW_CLI_COMMANDS_SEARCH_H
