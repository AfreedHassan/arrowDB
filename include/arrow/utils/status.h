#ifndef ARROW_UTILS_STATUS_H
#define ARROW_UTILS_STATUS_H

#include <cstdint>
#include <string>

namespace arrow::utils {

enum class StatusCode : uint8_t {
  kOk = 0,

  // Generic
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kUnimplemented,

  kDimensionMismatch,

  // I/O & persistence
  kIoError,
  kEof,
  kCorruption,
  kChecksumMismatch,

  // WAL / recovery
  kBadRecord,
  kBadHeader,
  kVersionMismatch,

  // Internal invariants
  kInternal,
};

class Status {
 public:
  Status() noexcept : code_(StatusCode::kOk) {}
  explicit Status(StatusCode code) noexcept : code_(code) {}
  Status(StatusCode code, const std::string& msg) noexcept
      : code_(code), message_(msg) {}
  Status(StatusCode code, std::string&& msg) noexcept
      : code_(code), message_(std::move(msg)) {}

  bool ok() const noexcept {
    return code_ == StatusCode::kOk;
  }

  StatusCode code() const noexcept { return code_; }
  const std::string& message() const noexcept { return message_; }

 private:
  StatusCode code_;
  std::string message_;
};

inline Status OkStatus() noexcept { return Status(); }
}  // namespace arrow::utils

#endif  // ARROW_UTILS_STATUS_H
