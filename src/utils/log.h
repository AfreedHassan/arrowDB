#ifndef ARROW_LOG_H
#define ARROW_LOG_H

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

namespace arrow {

enum class LogLevel { kDebug, kInfo, kWarn, kError };

inline LogLevel& globalLogLevel() {
  static LogLevel level = LogLevel::kInfo;
  return level;
}

inline void log(LogLevel level, std::string_view component, std::string_view msg) {
  if (level < globalLogLevel()) return;
  static constexpr const char* levelStr[] = {"DEBUG", "INFO", "WARN", "ERROR"};
  // Build the full line first, then write atomically with a single write call
  // to prevent interleaving from concurrent threads.
  std::string line = std::string("[") + levelStr[static_cast<int>(level)] + "] "
                   + "[" + std::string(component) + "] " + std::string(msg) + "\n";
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);
  std::cerr << line;
}

#define ARROW_LOG_DEBUG(comp, msg) ::arrow::log(::arrow::LogLevel::kDebug, comp, msg)
#define ARROW_LOG_INFO(comp, msg) ::arrow::log(::arrow::LogLevel::kInfo, comp, msg)
#define ARROW_LOG_WARN(comp, msg) ::arrow::log(::arrow::LogLevel::kWarn, comp, msg)
#define ARROW_LOG_ERROR(comp, msg) ::arrow::log(::arrow::LogLevel::kError, comp, msg)

}  // namespace arrow

#endif  // ARROW_LOG_H
