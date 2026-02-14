// Copyright 2025 ArrowDB
//
// ArrowDB REPL - Interactive command-line interface for vector database
// operations.
//
// Commands:
//   .search <query_text>  - Search for similar vectors
//   .exit                 - Exit the REPL

#include "args.h"
#include "commands/ingest.h"
#include "commands/search.h"
#include "embedder/embedder.h"
#include <arrow/arrow.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using arrow::utils::Result;

namespace {

  std::vector<std::string> questions = {
      "What type of organism is commonly used in preparation of foods such as cheese and yogurt?",
      "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?",
      "Changes from a less-ordered state to a more-ordered state (such as a liquid to a solid) are always what?",
      "What is the least dangerous radioactive decay?",
      "Kilauea in hawaii is the world’s most continuously active volcano. very active volcanoes characteristically eject red-hot rocks and lava rather than this?",
      "When a meteoroid reaches earth, what is the remaining object called?",
      "What kind of a reaction occurs when a substance reacts quickly with oxygen?",
      "Organisms categorized by what species descriptor demonstrate a version of allopatric speciation and have limited regions of overlap with one another, but where they overlap they interbreed successfully?.",
      "Alpha emission is a type of what?",
      "What is the stored food in a seed called?",
      "Zinc is more easily oxidized than iron because zinc has a lower reduction potential. since zinc has a lower reduction potential, it is a more what?",
      "What is controlled by both genes and experiences in a given envionment?"
  };

// Default configuration
constexpr const char *DEFAULT_COLLECTION_PATH = "owt_collection";
constexpr const char *DEFAULT_TEXT_FILE = "openwebtext.txt";
constexpr const char *DEFAULT_MODEL_PATH = "models/all-MiniLM-L6-v2.onnx";
constexpr const char *DEFAULT_TOKENIZER_NAME =
    "sentence-transformers/all-MiniLM-L6-v2";
constexpr size_t DEFAULT_NUM_SAMPLES = 10000;

void printWelcome() {
  std::cout
      << "\n╔════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║              ArrowDB Vector Database REPL                  ║\n";
  std::cout
      << "╚════════════════════════════════════════════════════════════╝\n\n";
  std::cout << "Commands:\n";
  std::cout << "  .search <query>      - Search for similar vectors\n";
  std::cout << "  .exit                - Exit the REPL\n";
  std::cout << "  .help                - Show this help message\n\n";
}

void printHelp() {
  std::cout << "\nArrowDB Commands:\n";
  std::cout << "  .search <query_text>\n";
  std::cout << "    Search the collection for vectors similar to the query.\n";
  std::cout << "    Uses default collection (owt_collection) and text file "
               "(openwebtext.txt).\n";
  std::cout << "    Returns 10 nearest neighbors.\n\n";
  std::cout << "  .exit\n";
  std::cout << "    Exit the REPL.\n\n";
  std::cout << "  .help\n";
  std::cout << "    Display this help message.\n\n";
}

bool checkCollectionExists(const std::string &path) {
  return std::filesystem::exists(path) &&
         std::filesystem::exists(path + "/meta.json");
}

bool checkFileExists(const std::string &path) {
  return std::filesystem::exists(path);
}

void handleSearch(const std::string &queryText,
                  const arrow::Collection &collection,
                  const std::string &textFile, const std::string &modelPath) {

  // Check text file exists
  if (!checkFileExists(textFile)) {
    std::cerr << "Error: Text file not found at " << textFile << "\n";
    return;
  }

  // Perform search
  arrow::cli::searchWithText(queryText, collection, textFile, modelPath, 10,
                             200);
}

void handleSearch(const std::string &queryText,
                  const std::string &collectionPath,
                  const std::string &textFile, const std::string &modelPath) {
  // Check collection exists
  if (!checkCollectionExists(collectionPath)) {
    std::cerr << "Error: Collection not found at " << collectionPath << "\n";
    std::cerr << "Please create a collection first.\n";
    return;
  }

  // Check text file exists
  if (!checkFileExists(textFile)) {
    std::cerr << "Error: Text file not found at " << textFile << "\n";
    return;
  }

  // Perform search
  arrow::cli::searchWithText(queryText, collectionPath, textFile, modelPath, 10,
                             200);
}

std::string trim(const std::string &str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  if (start == std::string::npos)
    return "";
  size_t end = str.find_last_not_of(" \t\n\r");
  return str.substr(start, end - start + 1);
}

/// Save text chunks to file (one per line)
bool saveTextFile(const std::vector<std::string> &chunks,
                  const std::string &path) {
  std::ofstream file(path);
  if (!file.is_open())
    return false;
  for (const auto &chunk : chunks) {
    file << chunk << "\n";
  }
  return true;
}

/// Ensure the OpenWebText collection exists, download and create if not
bool ensureCollectionExists(const std::string &collectionPath,
                            const std::string &modelPath,
                            const std::string &tokenizerName) {
  if (checkCollectionExists(collectionPath)) {
    std::cout << "Collection found at: " << collectionPath << "\n";
    return true;
  }

  std::cout << "Collection not found. Downloading OpenWebText dataset...\n";
  std::cout << "This may take a few minutes on first run.\n\n";

  // Create collection directory first so Rust can save text file there
  std::filesystem::create_directories(collectionPath);

  // Text file path for Rust to save
  std::string textPath = std::string(collectionPath) + "/" + DEFAULT_TEXT_FILE;

  // Call Rust to download and embed (also saves text file)
  auto datasetOpt = Embedder::downloadAndEmbed(modelPath, tokenizerName,
                                               DEFAULT_NUM_SAMPLES, textPath);

  if (!datasetOpt.has_value()) {
    std::cerr << "Error: Failed to download and embed dataset\n";
    return false;
  }

  auto &dataset = datasetOpt.value();
  std::cout << "\nDownloaded " << dataset.chunks.size() << " text chunks\n";

  // Note: Text file already saved by Rust during download/embed

  // Create the Arrow Collection
  arrow::CollectionConfig config;
  config.name = "openwebtext";
  config.dimensions = 384;
  config.space = arrow::Space::Cosine;

  arrow::Collection collection(config);

  // Insert all vectors
  std::cout << "Inserting " << dataset.embeddings.size()
            << " vectors into collection...\n";
  for (size_t i = 0; i < dataset.embeddings.size(); ++i) {
    auto status = collection.insert(std::to_string(i), dataset.embeddings[i]);
    if (!status.ok()) {
      std::cerr << "Warning: Failed to insert vector " << i << "\n";
    }
    if (i % 1000 == 0 && i > 0) {
      std::cout << "  Inserted " << i << "/" << dataset.embeddings.size()
                << " vectors\r";
      std::cout.flush();
    }
  }
  std::cout << "  Inserted " << dataset.embeddings.size() << "/"
            << dataset.embeddings.size() << " vectors\n";

  // Save the collection
  auto saveStatus = collection.save(collectionPath);
  if (!saveStatus.ok()) {
    std::cerr << "Error: Failed to save collection: " << saveStatus.message()
              << "\n";
    return false;
  }

  std::cout << "Collection saved to: " << collectionPath << "\n";
  std::cout << "Ready for search!\n\n";
  return true;
}

} // namespace

/*
int example(int argc, char *argv[]) {
  // If arguments provided, run in command mode
  if (argc > 1) {
    arrow::cli::CLIArgs args(argc, argv);

    if (args.command.empty()) {
      std::cerr << "ArrowDB - Vector Database CLI\n\n";
      std::cerr << "Usage (CLI mode):\n";
      std::cerr << "  ./arrowDB search <query_text> [-c <collection>] [-t "
                   "<text_file>]\n";
      std::cerr << "  ./arrowDB ingest -e <embeddings_file> -i <ids_file> -t "
                   "<text_file>\n\n";
      std::cerr << "Or run without arguments for interactive mode:\n";
      std::cerr << "  ./arrowDB\n";
      return 1;
    }

    if (args.command == "search") {
      std::string queryText;
      for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-' && i + 1 < argc) {
          ++i;
          continue;
        }
        if (!queryText.empty())
          queryText += " ";
        queryText += arg;
      }

      if (queryText.empty()) {
        std::cerr << "Error: search command requires a query string\n";
        return 1;
      }

      std::string collectionPath = args.get("c", "owt_collection");
      std::string textFile = args.get("t", "openwebtext.txt");
      std::string modelPath = args.get("m", "models/all-MiniLM-L6-v2.onnx");

      arrow::cli::searchWithText(queryText, collectionPath, textFile, modelPath,
                                 10, 200);

    } else if (args.command == "ingest") {
      std::string embeddingsFile = args.get("e");
      std::string idsFile = args.get("i");
      std::string textFile = args.get("t");
      std::string collectionName = args.get("-c");
      std::string outputPath = args.get("o", "collection_output");

      std::cout << "Collection name: " << collectionName << "\n";
      if (embeddingsFile.empty() || textFile.empty()) {
        std::cerr << "Error: ingest command requires -e and -t flags\n";
        return 1;
      }

      arrow::cli::ingest(embeddingsFile, textFile, idsFile, collectionName);
    } else {
      std::cerr << "Unknown command: " << args.command << "\n";
      return 1;
    }

    return 0;
  }

  // Interactive REPL mode
  printWelcome();

  std::string collectionPath = DEFAULT_COLLECTION_PATH;
  std::string textFile =
      std::string(DEFAULT_COLLECTION_PATH) + "/" + DEFAULT_TEXT_FILE;
  std::string modelPath = DEFAULT_MODEL_PATH;

  // Ensure collection exists (download and create if needed)
  if (!ensureCollectionExists(collectionPath, modelPath,
                              DEFAULT_TOKENIZER_NAME)) {
    std::cerr << "Error: Could not initialize collection. Exiting.\n";
    return 1;
  }

  std::string line;
  while (true) {
    std::cout << "arrowdb> ";
    std::cout.flush();

    if (!std::getline(std::cin, line)) {
      // EOF or read error
      std::cout << "\n";
      break;
    }

    // Trim input
    line = trim(line);

    // Skip empty lines
    if (line.empty()) {
      continue;
    }

    // Handle exit command
    if (line == ".exit" || line == "exit") {
      std::cout << "Goodbye!\n";
      break;
    }

    // Handle help command
    if (line == ".help" || line == "help") {
      printHelp();
      continue;
    }

    // Handle search command
    if (line.substr(0, 7) == ".search") {
      std::string query = trim(line.substr(7));

      if (query.empty()) {
        std::cerr << "Error: .search requires a query string\n";
        std::cerr << "Usage: .search <your query text here>\n";
        continue;
      }

      std::cout << "\n";
      handleSearch(query, collectionPath, textFile, modelPath);
      std::cout << "\n";
      continue;
    }

    // Unknown command
    if (line[0] == '.') {
      std::cerr << "Unknown command: " << line << "\n";
      std::cerr << "Type .help for available commands.\n";
      continue;
    }

    // Treat as search query (implicit .search)
    std::cout << "\n";
    handleSearch(line, collectionPath, textFile, modelPath);
    std::cout << "\n";
  }

  return 0;
}
*/

int main(int argc, char *argv[]) {
  // If arguments provided, run in command mode
  if (argc > 1) {
    arrow::cli::CLIArgs args(argc, argv);

    if (args.command.empty()) {
      std::cerr << "ArrowDB - Vector Database CLI\n\n";
      std::cerr << "Usage (CLI mode):\n";
      std::cerr << "  ./arrowDB search <query_text> [-c <collection>] [-t "
                   "<text_file>]\n";
      std::cerr << "  ./arrowDB ingest -e <embeddings_file> -i <ids_file> -t "
                   "<text_file>\n\n";
      std::cerr << "Or run without arguments for interactive mode:\n";
      std::cerr << "  ./arrowDB\n";
      return 1;
    }

    if (args.command == "search") {
      std::string queryText;
      for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-' && i + 1 < argc) {
          ++i;
          continue;
        }
        if (!queryText.empty())
          queryText += " ";
        queryText += arg;
      }

      if (queryText.empty()) {
        std::cerr << "Error: search command requires a query string\n";
        return 1;
      }

      std::string collectionPath = args.get("c", "owt_collection");
      std::string textFile = args.get("t", "openwebtext.txt");
      std::string modelPath = args.get("m", "models/all-MiniLM-L6-v2.onnx");

      arrow::cli::searchWithText(queryText, collectionPath, textFile, modelPath,
                                 10, 200);

    } else if (args.command == "ingest") {
      std::string embeddingsFile = args.get("e");
      std::string idsFile = args.get("i");
      std::string textFile = args.get("t");
      std::string collectionName = args.get("c");
      std::string outputPath = args.get("o", "collection_output");

      if (embeddingsFile.empty() || textFile.empty()) {
        std::cerr << "Error: ingest command requires -e and -t flags\n";
        return 1;
      }

      arrow::cli::ingest(embeddingsFile, textFile, idsFile, collectionName);

    } else {
      std::cerr << "Unknown command: " << args.command << "\n";
      return 1;
    }

    return 0;
  }

  // Interactive REPL mode
  printWelcome();

  std::string collectionPath = "sciq_collection";
  std::string textFile =
      std::string(DEFAULT_COLLECTION_PATH) + "/" + DEFAULT_TEXT_FILE;
  std::string modelPath = DEFAULT_MODEL_PATH;
  std::string tokenizerName = DEFAULT_TOKENIZER_NAME;

  std::string line;

  using namespace arrow;

  collectionPath = "sciq_collection";
  Embedder embedder(modelPath);
  if (!embedder.ok()) {
    std::cerr << "Error: Failed to initialize embedder\n";
    return 1;
  }

  Result<Collection> res = Collection::load(collectionPath);
  if (!res.ok()) {
    std::cerr << "Error loading collection: " << res.status().message()
              << "\n";
    return 1;
  }
  Collection &collection = res.value();

  std::cout << "Loaded collection: " << collection.name() << "\n";
  std::cout << "  Dimensions: " << collection.dimension() << "\n";
  std::cout << "  Total vectors: " << collection.size() << "\n\n";

  std::string textPath = "sciq.txt";
  for (std::string &q : questions) {
    std::cout << "Question: " << q << "\n";
    auto embedding = embedder.embed(q.c_str());
    std::vector<IndexSearchResult> sr = collection.search(embedding, 1, 200);
    if (sr.empty()) {
      std::cout << "No results found\n";
    } else {
      IndexSearchResult &result = sr[0];
      auto metaResult = collection.getMetadata(std::to_string(result.id));
      std::string text = metaResult.ok() ? std::get<std::string>(metaResult.value()["text"]) : "(no metadata)";
      std::cout << "Retrieved: " << text << "\n";
    }
    std::cout << "\n";
  }
  /*
  while (true) {
    std::cout << "arrowdb> ";
    std::cout.flush();

    if (!std::getline(std::cin, line)) {
      // EOF or read error
      std::cout << "\n";
      break;
    }

    // Trim input
    line = trim(line);

    // Skip empty lines
    if (line.empty()) {
      continue;
    }

    // Handle exit command
    if (line == ".exit" || line == "exit") {
      std::cout << "Goodbye!\n";
      break;
    }

    // Handle help command
    if (line == ".help" || line == "help") {
      printHelp();
      continue;
    }

    // Handle search command
    if (line.substr(0, 7) == ".search") {
      std::string query = trim(line.substr(7));

      if (query.empty()) {
        std::cerr << "Error: .search requires a query string\n";
        std::cerr << "Usage: .search <your query text here>\n";
        continue;
      }

      std::cout << "\n";
      textFile = "sciq.txt";
      handleSearch(query, collectionPath, textFile, modelPath);
      std::cout << "\n";
      continue;


    // Unknown command
    if (line[0] == '.') {
      std::cerr << "Unknown command: " << line << "\n";
      std::cerr << "Type .help for available commands.\n";
      continue;
    }

    // Treat as search query (implicit .search)
    std::cout << "\n";
    handleSearch(line, collectionPath, textFile, modelPath);
    std::cout << "\n";
  }

  if (line.substr(0, 7) == ".load") {
    std::string name = trim(line.substr(7));


    if (name.empty()) {
      std::cerr << "Error: .load requires a collection name\n";
      std::cerr << "Usage: .search <your query text here>\n";
      continue;
    }

    if (!std::filesystem::exists(name + "_collection")) {
        std::cerr << "Error: Collection not found at " << name + "_collection"
<< "\n";
    }
    colRes = arrow::Collection::load(name + "_collection");
    if (!colRes.ok()) {
      std::cerr << "Error loading collection: "
        << colRes.status().message() << "\n";
      return 1;
    }
    Collection& collection = colRes.value();

    std::cout << "\n";
    handleSearch(name, collectionPath, textFile, modelPath);
    std::cout << "\n";
    continue;
  }

  // Unknown command
  if (line[0] == '.') {
    std::cerr << "Unknown command: " << line << "\n";
    std::cerr << "Type .help for available commands.\n";
    continue;
  }

  // Treat as search query (implicit .search)
  std::cout << "\n";
  handleSearch(line, collectionPath, textFile, modelPath);
  std::cout << "\n";
}
*/

  return 0;
}
