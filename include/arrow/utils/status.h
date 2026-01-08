#ifndef ARROW_UTILS_STATUS_H
#define ARROW_UTILS_STATUS_H

#include <cstdint>
#include <string_view>

namespace arrow::utils {

enum class StatusCode : uint8_t {
  kOk = 0,

  // Generic
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kUnimplemented,

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
  constexpr Status() noexcept : code_(StatusCode::kOk) {}
  constexpr explicit Status(StatusCode code) noexcept : code_(code) {}
  Status(StatusCode code, std::string_view msg = "") noexcept
      : code_(code), message_(std::move(msg)) {}

  constexpr bool ok() const noexcept {
    return code_ == StatusCode::kOk;
  }

  constexpr StatusCode code() const noexcept { return code_; }
  const std::string_view message() const noexcept { return message_; }

 private:
  StatusCode code_;
  std::string_view message_;
};

inline constexpr Status OkStatus() noexcept { return Status(); }
}  // namespace arrow::utils

#endif  // ARROW_UTILS_STATUS_H
