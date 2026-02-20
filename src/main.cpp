// ArrowDB SCI-Q Demo
// Ingests text from sciq.txt with embeddings, stores text as metadata,
// then runs question-answering retrieval showing retrieved passages.

#include <arrow/arrow.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace arrow;

namespace {

const std::vector<std::string> questions = {
    "What type of organism is commonly used in preparation of foods such as cheese and yogurt?",
    "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?",
    "Changes from a less-ordered state to a more-ordered state (such as a liquid to a solid) are always what?",
    "What is the least dangerous radioactive decay?",
    "Kilauea in hawaii is the world's most continuously active volcano. very active volcanoes characteristically eject red-hot rocks and lava rather than this?",
    "When a meteoroid reaches earth, what is the remaining object called?",
    "What kind of a reaction occurs when a substance reacts quickly with oxygen?",
    "Organisms categorized by what species descriptor demonstrate a version of allopatric speciation and have limited regions of overlap with one another, but where they overlap they interbreed successfully?.",
    "Alpha emission is a type of what?",
    "What is the stored food in a seed called?",
    "Zinc is more easily oxidized than iron because zinc has a lower reduction potential. since zinc has a lower reduction potential, it is a more what?",
    "What is controlled by both genes and experiences in a given envionment?"
};

std::vector<std::string> readLines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream file(path);
  if (!file.is_open()) return lines;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) lines.push_back(std::move(line));
  }
  return lines;
}

} // namespace

int main() {
  constexpr const char* corpus = "sciq.txt";

  // Create collection with text metadata schema
  Client client("/tmp/sciq_demo");
  MetadataSchema schema;
  schema.field("text", FieldType::String, kRequiredField);

  CollectionConfig config = {
    .name = "sciq_demo",
    .dimensions = 384,
    .space = Space::Cosine,
    .schema = schema,
  };

  auto colRes = client.getOrCreateCollection("/tmp/sciq_demo", config);
  if (!colRes.ok()) {
    std::cerr << "Error creating collection: " << colRes.status().message() << "\n";
    return 1;
  }
  Collection* col = *colRes;

  // Ingest: embed each line and insert with text as metadata
  if (col->stats().vectorCount == 0) {
    std::cout << "Buidling collection...\n";
    auto res = col->insertBatch(readLines(corpus));
    if (!res) {
      std::cerr << res.status().message();
      return 1;
    }
  } else {
    std::cout << "Loaded existing collection.\n";
    col->printStats();
  }


  // Search: query() embeds the text internally and returns results with metadata
  for (const auto& q : questions) {
    std::cout << "Q: " << q << "\n";
    auto hits = col->query(q, 1, 200);
    if (hits.hits.empty()) {
      std::cout << "A: (no results)\n\n";
      continue;
    }
    ScoredDocument& hit = hits.hits[0];
    if (hit.metadata.contains("text")) {
      std::cout << "A: " << hit.metadata.at("text").asString()
                << "  (score=" << hit.score << ")\n\n";
    } else {
      std::cout << "A: (no text metadata, score=" << hit.score << ")\n\n";
    }
  }
  client.close();

  std::cout << "ArrowDB - A lightweight vector database in C++23\n"
            << "\n"
            << "ArrowDB is a library, not a standalone server. Use it via:\n"
            << "\n"
            << "  C++     #include <arrow/arrow.h>\n"
            << "  Python  import arrowdb\n"
            << "  Node.js const arrowdb = require('arrowdb')\n"
            << "\n"
            << "Quick start (C++):\n"
            << "\n"
            << "  arrow::CollectionConfig config;\n"
            << "  config.name = \"my_collection\";\n"
            << "  config.dimensions = 384;\n"
            << "  config.space = arrow::Space::Cosine;\n"
            << "\n"
            << "  auto collection = arrow::Collection(config);\n"
            << "  collection.insert(\"vec1\", embedding);\n"
            << "  auto results = collection.search(query, 10);\n"
            << "\n"
            << "Run tests:      ./tests\n"
            << "Run benchmarks: ./benchmarks\n"
            << "\n"
            << "See README.md for full documentation.\n";
  return 0;
}
