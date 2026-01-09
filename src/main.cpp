// Copyright 2025 ArrowDB
#include "arrow/collection.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

typedef enum {
  META_COMMAND_SUCCESS,
  META_COMMAND_UNRECOGNIZED_COMMAND
} MetaCommandResult;

typedef enum { PREPARE_SUCCESS, PREPARE_UNRECOGNIZED_STATEMENT } PrepareResult;

class InputBuffer {
public:
  std::string buffer;
  void read() {
    if (!std::getline(std::cin, buffer)) {
      if (buffer[0] == '.') {
        switch (doMetaCommand()) {
        case (META_COMMAND_SUCCESS):
          printf("SUCCESS");
        case (META_COMMAND_UNRECOGNIZED_COMMAND):
          printf("Unrecognized command '%s'\n", buffer.c_str());
        }
      }
    }
  }
  MetaCommandResult doMetaCommand() {
    return MetaCommandResult::META_COMMAND_SUCCESS;
  }
};

void prompt() { std::cout << "db > "; }

int repl() {
  InputBuffer input;
  while (true) {
    prompt();
    input.read();

    if (input.buffer == ".exit") {
      return EXIT_SUCCESS;
    } else {
      std::cout << "Unrecognized command '" << input.buffer << "'.\n";
    }
  }
}

using arrow::Collection;
using arrow::CollectionConfig;
using arrow::DataType;
using arrow::DistanceMetric;
using arrow::Metadata;
using arrow::VectorID;

void test() {
  // Create a simple collection demo
  CollectionConfig cfg("demo_collection", 128, DistanceMetric::Cosine,
                       DataType::Float32);

  Collection collection(cfg);
  //collection.printCollectionInfo();

  std::cout << "ArrowDB demo - Collection created successfully.\n";
  std::cout << "Run './tests' to execute integration tests.\n";

  // Example: Create some metadata for vectors
  std::unordered_map<VectorID, Metadata> metadataMap;

  // Vector 1: Image with tags and score
  Metadata vec1Meta;
  vec1Meta["category"] = std::string("image");
  vec1Meta["tags"] = std::string("dog,pet,animal");
  vec1Meta["score"] = 0.95;
  vec1Meta["active"] = true;
  metadataMap[1] = vec1Meta;

  // Vector 2: Text document
  Metadata vec2Meta;
  vec2Meta["category"] = std::string("text");
  vec2Meta["author"] = std::string("John Doe");
  vec2Meta["word_count"] = static_cast<int64_t>(1250);
  metadataMap[2] = vec2Meta;

  // Vector 3: Audio file
  Metadata vec3Meta;
  vec3Meta["category"] = std::string("audio");
  vec3Meta["duration"] = 180.5; // seconds
  vec3Meta["format"] = std::string("mp3");
  vec3Meta["active"] = false;
  metadataMap[3] = vec3Meta;

  // Export to JSON
  try {
    arrow::utils::exportMetadataToJson(metadataMap, "metadata_export.json");
    std::cout << "✓ Successfully exported metadata to metadata_export.json\n";

    // Print the JSON to console
    arrow::utils::json j = arrow::utils::json::object();
    for (const auto &[id, metadata] : metadataMap) {
      j[std::to_string(id)] = arrow::utils::metadataToJson(metadata);
    }
    std::cout << "\nExported JSON:\n";
    std::cout << j.dump(2) << "\n\n";

    // Test import
    auto imported =
        arrow::utils::importMetadataFromJson("metadata_export.json");
    std::cout << "✓ Successfully imported " << imported.size()
              << " metadata entries\n";

    // Verify imported data
    if (imported.size() == metadataMap.size()) {
      std::cout << "✓ Imported entry count matches\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
  }
}

/*
void write_test() {
  using namespace arrow;
  std::filesystem::path testPath = "/tmp/test_wal_dir";
  std::filesystem::create_directories(testPath);
  std::string walFile = (testPath / "db.wal").string();

  wal::Header header;
  header.magic = 0x41574C01;
  header.version = 1;
  header.flags = 0;
  header.creationTime = 1234567890;
  header.headerCrc32 = header.computeCrc32();
  header.padding = 0;

  std::ofstream file(walFile, std::ios::binary | std::ios::trunc);
  BinaryWriter writer(file);
  wal::WriteHeader(header, writer);
  writer.flush();
  file.close();

  std::cout << "File size after writing header: "
            << std::filesystem::file_size(walFile) << " bytes" << std::endl;

  auto f = std::make_unique<std::ifstream>(walFile, std::ios::binary);
  BinaryReader reader(std::move(f));
  auto result = wal::ParseHeader(reader);
  if (result.ok()) {
    std::cout << "ParseHeader succeeded" << std::endl;
  } else {
    std::cerr << "ParseHeader failed: " << result.status().message()
              << std::endl;
  }

  std::filesystem::remove_all(testPath);
}
*/


using namespace arrow::wal;

// Helper function to read a specific line from a text file by line number
std::string getLineFromFile(const std::string& filePath, size_t lineNumber) {
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

void ingest() {
  const size_t dims = 384;
  const std::string embeddingsPath = "embeddings.bin";
  const std::string textPath = "wikitext.txt";
  const std::string outputPath = "wiki_collection";
  const size_t batchSize = 10000;  // Insert 10K vectors at a time

  std::cout << "Starting ingestion from " << embeddingsPath << " and " << textPath
            << "...\n";

  // Create collection
  CollectionConfig cfg("wiki", dims, DistanceMetric::Cosine, DataType::Float32);
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

  std::cout << "✓ Ingestion complete!\n";
  std::cout << "  Total vectors: " << vectorId << "\n";
  std::cout << "  Time elapsed: " << duration.count() << "s\n";
  std::cout << "  Throughput: " << (vectorId / duration.count())
            << " vectors/second\n";
}

void searchWithQueryFile(const std::string& collectionPath,
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
  std::cout << "✓ Loaded collection: " << collection.name() << "\n";
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
  std::cout << "✓ Loaded query vector from: " << queryPath << "\n";
  std::cout << "  Searching for " << k << " nearest neighbors...\n\n";

  // Perform search
  auto searchResults = collection.search(query, k, ef);

  std::cout << "Search Results:\n";
  std::cout << std::string(80, '=') << "\n";
  for (size_t i = 0; i < searchResults.size(); ++i) {
    const auto& sr = searchResults[i];
    std::string text = getLineFromFile(textPath, sr.id);

    // Truncate text to 75 chars for display
    if (text.length() > 75) {
      text = text.substr(0, 75) + "...";
    }

    std::cout << (i + 1) << ". [Score: " << sr.score << "] "
              << text << "\n";
  }
  std::cout << std::string(80, '=') << "\n";
}

void searchExample() {
  // Default example with random query
  auto resultOrError = Collection::load("wiki_collection");
  if (!resultOrError.ok()) {
    std::cerr << "Error loading collection: "
              << resultOrError.status().message() << "\n";
    return;
  }

  Collection& collection = resultOrError.value();
  std::cout << "✓ Loaded collection: " << collection.name() << "\n";
  std::cout << "  Dimensions: " << collection.dimension() << "\n";
  std::cout << "  Metric: Cosine\n";
  std::cout << "  Total vectors: " << collection.size() << "\n\n";

  // Generate random query vector for demo
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<float> query(collection.dimension());
  for (uint32_t i = 0; i < collection.dimension(); ++i) {
    query[i] = uniform(rng);
  }

  std::cout << "Searching for 10 nearest neighbors...\n\n";
  auto searchResults = collection.search(query, 10, 200);

  std::cout << "Top 10 results:\n";
  for (size_t i = 0; i < searchResults.size(); ++i) {
    const auto& sr = searchResults[i];
    std::cout << (i + 1) << ". Vector ID: " << sr.id
              << ", Score: " << -sr.score << "\n";
  }
}

int main(int argc, char* argv[]) {
  if (argc >= 3 && std::string(argv[1]) == "query") {
    // Usage: ./arrowDB query <query_file> [k] [ef]
    std::string queryFile = argv[2];
    uint32_t k = (argc >= 4) ? std::stoul(argv[3]) : 10;
    uint32_t ef = (argc >= 5) ? std::stoul(argv[4]) : 200;
    searchWithQueryFile("wiki_collection", queryFile, "wikitext.txt", k, ef);
  } else if (argc > 1 && std::string(argv[1]) == "search") {
    searchExample();
  } else {
    ingest();
  }
  return 0;
}
