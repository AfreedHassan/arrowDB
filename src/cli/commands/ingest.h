// Copyright 2025 ArrowDB
#ifndef ARROW_CLI_COMMANDS_INGEST_H
#define ARROW_CLI_COMMANDS_INGEST_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "arrow/arrow.h"

namespace arrow::cli {

/// Ingest embeddings from binary file into a collection.
///
/// @param embeddingsPath Path to binary file with float32 embeddings
/// @param textPath Path to text file (one line per vector)
/// @param idsPath Path to IDs file (unused currently)
/// @param outputPath Path to save the collection
inline void ingest(const std::string& embeddingsPath = "embeddings.bin",
                   const std::string& textPath = "wikitext.txt",
                   const std::string& idsPath = "",
                   const std::string& outputPath = "wiki_collection") {
  const size_t dims = 384;
  const size_t batchSize = 10000;  // Insert 10K vectors at a time

  std::cout << "Starting ingestion from " << embeddingsPath << " and " << textPath
            << "...\n";

  // Create collection
  CollectionConfig cfg{.name = "owt", .dimensions = static_cast<uint32_t>(dims), .metric = DistanceMetric::L2};
  Collection collection(cfg);

  auto startTime = std::chrono::high_resolution_clock::now();

  // Open embeddings file
  std::ifstream embeddingsFile(embeddingsPath, std::ios::binary);
  if (!embeddingsFile.is_open()) {
    std::cerr << "Error: Failed to open " << embeddingsPath << "\n";
    return;
  }

  // Open text file
  std::ifstream textFile(textPath);
  if (!textFile.is_open()) {
    std::cerr << "Error: Failed to open " << textPath << "\n";
    return;
  }

  VectorID vectorId = 0;
  std::vector<std::pair<VectorID, std::vector<float>>> batch;
  batch.reserve(batchSize);

  std::string textLine;
  while (std::getline(textFile, textLine)) {
    // Read embedding (384 float32 values)
    std::vector<float> embedding(dims);
    embeddingsFile.read(reinterpret_cast<char*>(embedding.data()),
                       dims * sizeof(float));

    if (embeddingsFile.gcount() != static_cast<std::streamsize>(dims * sizeof(float))) {
      std::cerr << "Error: Failed to read complete embedding for vector "
                << vectorId << "\n";
      break;
    }

    // Add to batch
    batch.push_back({vectorId, embedding});

    // Set metadata with text content
    Metadata meta;
    meta["text"] = textLine;
    collection.setMetadata(vectorId, meta);

    vectorId++;

    // Insert batch when full
    if (batch.size() >= batchSize) {
      auto result = collection.insertBatch(batch);
      if (!result.ok()) {
        std::cerr << "Error: Batch insert failed at vector "
                  << (vectorId - batch.size()) << ": "
                  << result.status().message() << "\n";
        return;
      }

      size_t successCount = result.value().successCount;
      std::cout << "Inserted batch: vectors " << (vectorId - batch.size() + 1)
                << "-" << vectorId << " (" << successCount << " successful)\n";

      batch.clear();
    }
  }

  // Insert remaining vectors
  if (!batch.empty()) {
    auto result = collection.insertBatch(batch);
    if (!result.ok()) {
      std::cerr << "Error: Final batch insert failed: "
                << result.status().message() << "\n";
      return;
    }

    size_t successCount = result.value().successCount;
    std::cout << "Inserted final batch: vectors " << (vectorId - batch.size() + 1)
              << "-" << vectorId << " (" << successCount << " successful)\n";
  }

  embeddingsFile.close();
  textFile.close();

  // Save collection
  std::cout << "Saving collection to " << outputPath << "...\n";
  auto saveStatus = collection.save(outputPath);
  if (!saveStatus.ok()) {
    std::cerr << "Error: Failed to save collection: "
              << saveStatus.message() << "\n";
    return;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(
      endTime - startTime);

  std::cout << "Ingestion complete!\n";
  std::cout << "  Total vectors: " << vectorId << "\n";
  std::cout << "  Time elapsed: " << duration.count() << "s\n";
  if (duration.count() > 0) {
    std::cout << "  Throughput: " << (vectorId / duration.count())
              << " vectors/second\n";
  }
}

} // namespace arrow::cli

#endif // ARROW_CLI_COMMANDS_INGEST_H
