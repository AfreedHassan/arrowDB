#ifndef ARROW_UTILS_RESULT_H
#define ARROW_UTILS_RESULT_H

#include "status.h"
#include <expected>

namespace arrow::utils {
template <typename T>
class Result {
 public:
  using resType = std::expected<T, Status>;

  // Value
  Result(const T& v) : res_(v) {}
  Result(T&& v) : res_(std::move(v)) {}

  // Error
  Result(const Status& s) : res_(std::unexpected(s)) {}
  Result(Status&& s) : res_(std::unexpected(std::move(s))) {}

  inline bool ok() const noexcept { return res_.has_value(); }

  // Status access
  const Status& status() const & {
    static const Status kOk = OkStatus();
    return ok() ? kOk : res_.error();
  }

  Status status() && {
    return ok() ? OkStatus() : std::move(res_.error());
  }

  // Value access (explicit)
  T& value() & { return res_.value(); }
  const T& value() const & { return res_.value(); }
  T&& value() && { return std::move(res_.value()); }

 private:
  resType res_;
};
}

#endif // ARROW_UTILS_RESULT_H
