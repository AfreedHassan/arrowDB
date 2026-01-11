// Copyright 2025 ArrowDB
#ifndef ARROW_CLI_ARGS_H
#define ARROW_CLI_ARGS_H

#include <string>
#include <unordered_map>

namespace arrow::cli {

/// Command-line argument parser for ArrowDB CLI.
///
/// Parses arguments in format: command -flag value -flag value
/// Example: ./arrowDB search "query text" -c collection -m model.onnx
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

  /// Get a flag value, or default if not present.
  std::string get(const std::string& flag, const std::string& defaultValue = "") const {
    auto it = flags.find(flag);
    return (it != flags.end()) ? it->second : defaultValue;
  }

  /// Check if a flag is present.
  bool has(const std::string& flag) const {
    return flags.find(flag) != flags.end();
  }
};

} // namespace arrow::cli

#endif // ARROW_CLI_ARGS_H
