// Copyright 2025 ArrowDB
#include "arrow/arrow.h"
#include "arrow_embed.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <chrono>
#include <embedder/embedder.h>

#ifndef ARROW_EMBEDDING_MODEL_DIR
#define ARROW_EMBEDDING_MODEL_DIR ""
#endif

#ifndef ARROW_EMBEDDING_DEFAULT_MODEL
#define ARROW_EMBEDDING_DEFAULT_MODEL "all-MiniLM-L6-v2"
#endif

class CLIArgs {
public:
  std::unordered_map<std::string, std::string> flags;
  std::string command;

  CLIArgs(int argc, char* argv[]) {
    if (argc < 2) return;
    command = argv[1];

    // Parse flags in format: -flag value
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg[0] == '-' && i + 1 < argc) {
        std::string flag = arg.substr(1);  // Remove leading '-'
        std::string value = argv[++i];
        flags[flag] = value;
      }
    }
  }

  std::string get(const std::string& flag, const std::string& defaultValue = "") const {
    auto it = flags.find(flag);
    return (it != flags.end()) ? it->second : defaultValue;
  }

  bool has(const std::string& flag) const {
    return flags.find(flag) != flags.end();
  }
};

// Rust FFI function declarations (from arrow_embed library)
extern "C" {
  int32_t arrow_embed_init(const char* model_path, const char* tokenizer_name);
  EmbeddingResult arrow_embed_text(const char* text);
  void arrow_embed_free(EmbeddingResult result);
  size_t arrow_embed_dimension();
}

#ifndef ARROW_EMBEDDING_MODEL_DIR
#define ARROW_EMBEDDING_MODEL_DIR = ""
#endif

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

using arrow::Collection;
using arrow::InternalCollectionConfig;
using arrow::DataType;
using arrow::DistanceMetric;
using arrow::Metadata;
using arrow::VectorID;

void test() {
  // Create a simple collection demo
  InternalCollectionConfig cfg("demo_collection", 128, DistanceMetric::Cosine,
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

void ingest(const std::string& embeddingsPath = "embeddings.bin",
            const std::string& textPath = "wikitext.txt",
            const std::string& idsPath = "",
            const std::string& outputPath = "wiki_collection") {
  const size_t dims = 384;
  const size_t batchSize = 10000;  // Insert 10K vectors at a time

  std::cout << "Starting ingestion from " << embeddingsPath << " and " << textPath
            << "...\n";

  // Create collection
  InternalCollectionConfig cfg("owt", dims, DistanceMetric::L2, DataType::Float32);
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

  std::ofstream outputFile("output.txt");
  std::cout << "Search Results:\n";
  std::cout << std::string(80, '=') << "\n";
  for (size_t i = 0; i < searchResults.size(); ++i) {
    const auto& sr = searchResults[i];
    std::string text = getLineFromFile(textPath, sr.id);
    outputFile << text << "\n";

    /* Truncate text to 75 chars for display
    if (text.length() > 75) {
      text = text.substr(0, 75) + "...";
    }
    */

    std::cout << (i + 1) << ". [Score: " << sr.score << "] "
              << text << "\n";
  }
  std::cout << std::string(80, '=') << "\n";
  outputFile.close();
}


std::string defaultModelPath() {
  std::string defaultModelDir = ARROW_EMBEDDING_MODEL_DIR;
  std::string defaultModel = ARROW_EMBEDDING_DEFAULT_MODEL;
  std::string defaultPath = defaultModelDir+"/"+defaultModel;
  return defaultPath;
}
// Initialize the Rust embedder with model and tokenizer
bool initEmbedder(const std::string& modelPath = "models/all-MiniLM-L6-v2.onnx",
                  const std::string& tokenizerName = "sentence-transformers/all-MiniLM-L6-v2") {
  std::cout << "Initializing embedder...\n";
  int32_t result = arrow_embed_init(modelPath.c_str(), tokenizerName.c_str());
  if (result != 0) {
    std::cerr << "Error: Failed to initialize embedder (code: " << result << ")\n";
    return false;
  }
  std::cout << "✓ Embedder initialized (dim=" << arrow_embed_dimension() << ")\n";
  return true;
}

void searchWithText(const std::string& queryText,
                    const std::string& collectionPath = "owt_collection",
                    const std::string& textPath = "openwebtext.txt",
                    const std::string& modelPath = "",
                    uint32_t k = 10,
                    uint32_t ef = 200) {
  // Resolve to default model 
  if (modelPath == "") {

  }
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
  std::cout << "✓ Loaded collection: " << collection.name() << "\n";
  std::cout << "  Dimensions: " << collection.dimension() << "\n";
  std::cout << "  Total vectors: " << collection.size() << "\n\n";

  // Embed query text
  std::cout << "Embedding query: \"" << queryText << "\"\n";
  std::vector<float> query = embedder.embed(queryText.c_str());

  if (query.empty()) return;

  std::cout << "✓ Query embedded successfully\n";
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

int main(int argc, char* argv[]) {
  CLIArgs args(argc, argv);

  if (args.command == "query") {
    std::string queryFile = args.get("f");
    std::string collectionPath = args.get("c", "wiki_collection");
    std::string textFile = args.get("t", "wikitext.txt");

    if (queryFile.empty()) {
      std::cerr << "Error: query command requires -f <query_file>\n";
      std::cerr << "Usage: ./arrowDB query -f <query_file> [-c <collection_path>] "
                   "[-t <text_file>]\n";
      return 1;
    }

    searchWithQueryFile(collectionPath, queryFile, textFile);
  } else if (args.command == "ingest") {
    std::string embeddingsFile = args.get("e");
    std::string idsFile = args.get("i");
    std::string textFile = args.get("t");
    std::string outputPath = args.get("o");
    

    if (embeddingsFile.empty() || idsFile.empty() || textFile.empty()) {
      std::cerr << "Error: ingest command requires -e, -i, and -t flags\n";
      std::cerr << "Usage: ./arrowDB ingest -e <embeddings_file> -i <ids_file> "
                   "-t <text_file> -o <output_path>\n";
      return 1;
    }

    ingest(embeddingsFile, textFile, idsFile, outputPath);
  } else if (args.command == "search") {
    // Collect all remaining arguments as the query text
    std::string queryText;
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      // Skip flags and their values
      if (arg[0] == '-' && i + 1 < argc) {
        ++i;  // Skip the flag value
        continue;
      }
      if (!queryText.empty()) queryText += " ";
      queryText += arg;
    }

    if (queryText.empty()) {
      std::cerr << "Error: search command requires a query string\n";
      std::cerr << "Usage: ./arrowDB search <query_text> [-c <collection_path>] "
                   "[-t <text_file>] [-m <model_path>]\n";
      return 1;
    }

    std::string collectionPath = args.get("c", "owt_collection");
    std::string textFile = args.get("t", "openwebtext.txt");
    std::string modelPath = args.get("m", "models/all-MiniLM-L6-v2.onnx");

    searchWithText(queryText, collectionPath, textFile, modelPath, 10, 200);
  } else {
    std::cerr << "Unknown command: " << args.command << "\n";
    std::cerr << "Usage:\n";
    std::cerr << "  ./arrowDB search <query_text> [-c <collection>] [-t <text_file>] "
                 "[-m <model.onnx>]\n";
    std::cerr << "  ./arrowDB query -f <query_file> [-c <collection>] "
                 "[-t <text_file>]\n";
    std::cerr << "  ./arrowDB ingest -e <embeddings_file> -i <ids_file> "
                 "-t <text_file>\n";
    return 1;
  }

  return 0;
}
