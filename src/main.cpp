// Copyright 2025 ArrowDB
#include "arrow/collection.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

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
                       DataType::Float16);

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

int main() {
  //write_test();
  return 0;
}
