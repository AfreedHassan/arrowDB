#ifndef ARROW_UTILS_RESULT_H
#define ARROW_UTILS_RESULT_H

#include "status.h"
#include <expected>
#include <type_traits>

namespace arrow::utils {

/// Result<T> wraps std::expected<T, Status> for ergonomic error handling.
///
/// Usage:
///   auto result = doSomething();
///   if (result) {  // or: if (result.ok())
///     auto value = *result;  // or: result.value()
///   }
///
/// Monadic operations (functional style):
///   auto result = doSomething()
///       .transform([](auto v) { return process(v); })
///       .inspect([](auto v) { log(v); })
///       .recover([](Status s) { return defaultValue(); });
///
/// Error handling:
///   - operator*, operator->, value() throw std::bad_expected_access if !ok()
///
template <typename T>
class Result {
 public:
  using resType = std::expected<T, Status>;

  Result(const T& v) : res_(v) {}
  Result(T&& v) : res_(std::move(v)) {}

  Result(const Status& s) : res_(std::unexpected(s)) {}
  Result(Status&& s) : res_(std::unexpected(std::move(s))) {}

  Result(const Result&) = default;
  Result(Result&&) = default;
  Result& operator=(const Result&) = default;
  Result& operator=(Result&&) = default;

  inline bool ok() const noexcept { return res_.has_value(); }
  explicit operator bool() const noexcept { return ok(); }

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

  T& operator*() & { return res_.value(); }
  const T& operator*() const & { return res_.value(); }
  T&& operator*() && { return std::move(res_.value()); }

  // Member access operators
  T* operator->() requires std::is_class_v<T> {
    return &res_.value();
  }
  const T* operator->() const requires std::is_class_v<T> {
    return &res_.value();
  }

  // Monadic operations

  /// Transform the success value using function f.
  /// If ok(), applies f to the value and returns Result<U>.
  /// If !ok(), propagates the error status.
  template <typename F>
  auto transform(F&& f) -> Result<std::invoke_result_t<F, T>> {
    using U = std::invoke_result_t<F, T>;
    if (ok()) {
      return Result<U>(std::forward<F>(f)(value()));
    }
    return Result<U>(status());
  }

  /// Run function f for side effects on success, without changing the value.
  /// If ok(), calls f with the value, then returns *this.
  /// If !ok(), does nothing.
  template <typename F>
  Result<T>& inspect(F&& f) {
    if (ok()) {
      std::forward<F>(f)(value());
    }
    return *this;
  }

  /// Recover from an error by providing an alternative Result.
  /// If ok(), returns *this unchanged.
  /// If !ok(), calls f with the error status and returns its result.
  template <typename F>
  auto recover(F&& f) -> Result<T> {
    if (!ok()) {
      return std::forward<F>(f)(status());
    }
    return *this;
  }

 private:
  resType res_;
};
}

#endif // ARROW_UTILS_RESULT_H
