#ifndef ARROW_LOG_H
#define ARROW_LOG_H

#include <iostream>
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
  std::cerr << "[" << levelStr[static_cast<int>(level)] << "] "
            << "[" << component << "] " << msg << "\n";
}

#define ARROW_LOG_DEBUG(comp, msg) ::arrow::log(::arrow::LogLevel::kDebug, comp, msg)
#define ARROW_LOG_INFO(comp, msg) ::arrow::log(::arrow::LogLevel::kInfo, comp, msg)
#define ARROW_LOG_WARN(comp, msg) ::arrow::log(::arrow::LogLevel::kWarn, comp, msg)
#define ARROW_LOG_ERROR(comp, msg) ::arrow::log(::arrow::LogLevel::kError, comp, msg)

}  // namespace arrow

#endif  // ARROW_LOG_H
