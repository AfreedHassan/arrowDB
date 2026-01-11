// Copyright 2025 ArrowDB
//
// ArrowDB CLI - Command-line interface for vector database operations.
//
// Usage:
//   ./arrowDB search <query_text> [-c <collection>] [-t <text_file>] [-m <model.onnx>]
//   ./arrowDB query -f <query_file> [-c <collection>] [-t <text_file>]
//   ./arrowDB ingest -e <embeddings_file> -i <ids_file> -t <text_file> [-o <output>]

#include "args.h"
#include <arrow/arrow.h>>
#include "commands/ingest.h"
#include "commands/search.h"

#include <iostream>
#include <string>

namespace {

void printUsage() {
  std::cerr << "ArrowDB - Vector Database CLI\n\n";
  std::cerr << "Usage:\n";
  std::cerr << "  ./arrowDB search <query_text> [-c <collection>] [-t <text_file>] "
               "[-m <model.onnx>]\n";
  std::cerr << "  ./arrowDB query -f <query_file> [-c <collection>] "
               "[-t <text_file>]\n";
  std::cerr << "  ./arrowDB ingest -e <embeddings_file> -i <ids_file> "
               "-t <text_file> [-o <output>]\n";
}

} // namespace

int main(int argc, char* argv[]) {
  arrow::cli::CLIArgs args(argc, argv);

  if (args.command.empty()) {
    printUsage();
    return 1;
  }

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

    arrow::cli::searchWithQueryFile(collectionPath, queryFile, textFile);

  } else if (args.command == "ingest") {
    std::string embeddingsFile = args.get("e");
    std::string idsFile = args.get("i");
    std::string textFile = args.get("t");
    std::string outputPath = args.get("o", "collection_output");

    if (embeddingsFile.empty() || textFile.empty()) {
      std::cerr << "Error: ingest command requires -e and -t flags\n";
      std::cerr << "Usage: ./arrowDB ingest -e <embeddings_file> "
                   "-t <text_file> [-o <output_path>]\n";
      return 1;
    }

    arrow::cli::ingest(embeddingsFile, textFile, idsFile, outputPath);

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

    arrow::cli::searchWithText(queryText, collectionPath, textFile, modelPath, 10, 200);

  } else {
    std::cerr << "Unknown command: " << args.command << "\n\n";
    printUsage();
    return 1;
  }

  return 0;
}
