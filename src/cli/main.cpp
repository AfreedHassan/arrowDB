// Copyright 2025 ArrowDB
//
// ArrowDB REPL - Interactive command-line interface for vector database operations.
//
// Commands:
//   .search <query_text>  - Search for similar vectors
//   .exit                 - Exit the REPL

#include "args.h"
#include <arrow/arrow.h>
#include "commands/ingest.h"
#include "commands/search.h"

#include <iostream>
#include <string>
#include <filesystem>

namespace {

void printWelcome() {
  std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
  std::cout << "║              ArrowDB Vector Database REPL                  ║\n";
  std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
  std::cout << "Commands:\n";
  std::cout << "  .search <query>      - Search for similar vectors\n";
  std::cout << "  .exit                - Exit the REPL\n";
  std::cout << "  .help                - Show this help message\n\n";
}

void printHelp() {
  std::cout << "\nArrowDB Commands:\n";
  std::cout << "  .search <query_text>\n";
  std::cout << "    Search the collection for vectors similar to the query.\n";
  std::cout << "    Uses default collection (owt_collection) and text file (openwebtext.txt).\n";
  std::cout << "    Returns 10 nearest neighbors.\n\n";
  std::cout << "  .exit\n";
  std::cout << "    Exit the REPL.\n\n";
  std::cout << "  .help\n";
  std::cout << "    Display this help message.\n\n";
}

bool checkCollectionExists(const std::string& path) {
  return std::filesystem::exists(path) &&
         std::filesystem::exists(path + "/meta.json");
}

bool checkFileExists(const std::string& path) {
  return std::filesystem::exists(path);
}

void handleSearch(const std::string& queryText,
                 const std::string& collectionPath,
                 const std::string& textFile,
                 const std::string& modelPath) {
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
  arrow::cli::searchWithText(queryText, collectionPath, textFile, modelPath, 10, 200);
}

std::string trim(const std::string& str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  size_t end = str.find_last_not_of(" \t\n\r");
  return str.substr(start, end - start + 1);
}

} // namespace

int main(int argc, char* argv[]) {
  // If arguments provided, run in command mode
  if (argc > 1) {
    arrow::cli::CLIArgs args(argc, argv);

    if (args.command.empty()) {
      std::cerr << "ArrowDB - Vector Database CLI\n\n";
      std::cerr << "Usage (CLI mode):\n";
      std::cerr << "  ./arrowDB search <query_text> [-c <collection>] [-t <text_file>]\n";
      std::cerr << "  ./arrowDB ingest -e <embeddings_file> -i <ids_file> -t <text_file>\n\n";
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
        if (!queryText.empty()) queryText += " ";
        queryText += arg;
      }

      if (queryText.empty()) {
        std::cerr << "Error: search command requires a query string\n";
        return 1;
      }

      std::string collectionPath = args.get("c", "owt_collection");
      std::string textFile = args.get("t", "openwebtext.txt");
      std::string modelPath = args.get("m", "models/all-MiniLM-L6-v2.onnx");

      arrow::cli::searchWithText(queryText, collectionPath, textFile, modelPath, 10, 200);

    } else if (args.command == "ingest") {
      std::string embeddingsFile = args.get("e");
      std::string idsFile = args.get("i");
      std::string textFile = args.get("t");
      std::string outputPath = args.get("o", "collection_output");

      if (embeddingsFile.empty() || textFile.empty()) {
        std::cerr << "Error: ingest command requires -e and -t flags\n";
        return 1;
      }

      arrow::cli::ingest(embeddingsFile, textFile, idsFile, outputPath);

    } else {
      std::cerr << "Unknown command: " << args.command << "\n";
      return 1;
    }

    return 0;
  }

  // Interactive REPL mode
  printWelcome();

  std::string collectionPath = "owt_collection";
  std::string textFile = "openwebtext.txt";
  std::string modelPath = "models/all-MiniLM-L6-v2.onnx";

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
