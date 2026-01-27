#include "arrow/utils/result.h"
#include "arrow/utils/status.h"
#include <gtest/gtest.h>
#include <string>

using namespace arrow::utils;

class ResultTest : public ::testing::Test {};

TEST_F(ResultTest, OperatorStar) {
  Result<int> r(42);
  EXPECT_EQ(*r, 42);

  Result<int> err(Status(StatusCode::kInternal, "fail"));
  EXPECT_THROW(*err, std::bad_expected_access<Status>);
}

TEST_F(ResultTest, OperatorStarConst) {
  const Result<int> r(42);
  EXPECT_EQ(*r, 42);
}

TEST_F(ResultTest, OperatorStarRvalueRef) {
  Result<std::string> r("hello");
  std::string moved = *std::move(r);
  EXPECT_EQ(moved, "hello");
}

TEST_F(ResultTest, OperatorStarRvalueRefThrowsOnError) {
  Result<std::string> err(Status(StatusCode::kInternal, "fail"));
  EXPECT_THROW(*std::move(err), std::bad_expected_access<Status>);
}

TEST_F(ResultTest, OperatorArrow) {
  struct Data { int value = 42; };
  Result<Data> r(Data{});
  EXPECT_EQ(r->value, 42);
}

TEST_F(ResultTest, OperatorArrowConst) {
  struct Data { int value = 42; };
  const Result<Data> r(Data{});
  EXPECT_EQ(r->value, 42);
}

TEST_F(ResultTest, OperatorArrowThrowsOnError) {
  struct Data { int value = 42; };
  Result<Data> err(Status(StatusCode::kInternal, "fail"));
  EXPECT_THROW(err->value, std::bad_expected_access<Status>);
}

TEST_F(ResultTest, OperatorBool) {
  Result<int> r(42);
  EXPECT_TRUE(static_cast<bool>(r));

  Result<int> err(Status(StatusCode::kInternal, "fail"));
  EXPECT_FALSE(static_cast<bool>(err));
}

TEST_F(ResultTest, OkMethod) {
  Result<int> r(42);
  EXPECT_TRUE(r.ok());

  Result<int> err(Status(StatusCode::kInternal, "fail"));
  EXPECT_FALSE(err.ok());
}

TEST_F(ResultTest, AndThen) {
  auto r = Result<int>(5)
      .transform([](int x) { return x * 2; });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, 10);

  auto err = Result<int>(Status(StatusCode::kInternal, "fail"))
      .transform([](int x) { return x * 2; });
  EXPECT_FALSE(err.ok());
}

TEST_F(ResultTest, AndThenTypeChange) {
  auto r = Result<int>(42)
      .transform([](int x) { return std::to_string(x); });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, "42");
}

TEST_F(ResultTest, AndThenWithRef) {
  auto r = Result<int>(5)
      .transform([](const int& x) { return x + 1; });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, 6);
}

TEST_F(ResultTest, AndThenPreservesError) {
  auto err = Result<int>(Status(StatusCode::kNotFound, "not found"))
      .transform([](int x) { return x * 2; });
  EXPECT_FALSE(err.ok());
  EXPECT_EQ(err.status().code(), StatusCode::kNotFound);
  EXPECT_EQ(err.status().message(), "not found");
}

TEST_F(ResultTest, Inspect) {
  int called = 0;
  auto r = Result<int>(42)
      .inspect([&](int x) { called = x; });
  EXPECT_EQ(called, 42);
  EXPECT_EQ(*r, 42);
}

TEST_F(ResultTest, InspectWithRef) {
  bool called = false;
  auto r = Result<int>(42)
      .inspect([&](const int& x) {
        called = true;
        EXPECT_EQ(x, 42);
      });
  EXPECT_TRUE(called);
}

TEST_F(ResultTest, InspectNotCalledOnError) {
  bool called = false;
  auto r = Result<int>(Status(StatusCode::kInternal, "fail"))
      .inspect([&](int x) { called = true; });
  EXPECT_FALSE(called);
  EXPECT_FALSE(r.ok());
}

TEST_F(ResultTest, InspectReturnsReference) {
  auto r = Result<int>(42);
  auto& ref = r.inspect([](int x) {});
  EXPECT_EQ(&ref, &r);
}

TEST_F(ResultTest, Recover) {
  auto r = Result<int>(Status(StatusCode::kInternal, "fail"))
      .recover([](Status s) { return Result<int>(99); });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, 99);
}

TEST_F(ResultTest, RecoverNotCalledOnSuccess) {
  bool called = false;
  auto r = Result<int>(42)
      .recover([&](Status s) {
        called = true;
        return Result<int>(99);
      });
  EXPECT_FALSE(called);
  EXPECT_EQ(*r, 42);
}

TEST_F(ResultTest, RecoverPreservesErrorOnNoRecovery) {
  auto r = Result<int>(Status(StatusCode::kNotFound, "not found"))
      .recover([](Status s) { return Result<int>(s); });
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status().code(), StatusCode::kNotFound);
}

TEST_F(ResultTest, RecoverCanTransformToDifferentValue) {
  auto r = Result<int>(Status(StatusCode::kInternal, "fail"))
      .recover([](Status s) { return Result<int>(s.code() == StatusCode::kInternal ? 0 : -1); });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, 0);
}

TEST_F(ResultTest, Chaining) {
  auto r = Result<int>(5)
      .transform([](int x) { return x * 2; })
      .inspect([](int x) {})
      .transform([](int x) { return std::to_string(x); });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, "10");
}

TEST_F(ResultTest, ChainingWithRecover) {
  auto r = Result<int>(Status(StatusCode::kInternal, "fail"))
      .recover([](Status s) { return Result<int>(0); })
      .transform([](int x) { return x + 10; });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, 10);
}

TEST_F(ResultTest, ChainingWithMultipleAndThens) {
  auto r = Result<int>(1)
      .transform([](int x) { return x + 1; })
      .transform([](int x) { return x * 2; })
      .transform([](int x) { return x * 3; });
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(*r, 12);
}

TEST_F(ResultTest, ChainingErrorPropagation) {
  auto r = Result<int>(Status(StatusCode::kNotFound, "not found"))
      .transform([](int x) { return x * 2; })
      .inspect([](int x) {})
      .transform([](int x) { return std::to_string(x); });
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status().code(), StatusCode::kNotFound);
}

TEST_F(ResultTest, CopyConstructor) {
  Result<int> original(42);
  Result<int> copy = original;
  EXPECT_TRUE(copy.ok());
  EXPECT_EQ(*copy, 42);
  EXPECT_TRUE(original.ok());
  EXPECT_EQ(*original, 42);
}

TEST_F(ResultTest, MoveConstructor) {
  Result<std::string> original("hello");
  Result<std::string> moved = std::move(original);
  EXPECT_TRUE(moved.ok());
  EXPECT_EQ(*moved, "hello");
}

TEST_F(ResultTest, CopyAssignment) {
  Result<int> original(42);
  Result<int> copy(0);
  copy = original;
  EXPECT_TRUE(copy.ok());
  EXPECT_EQ(*copy, 42);
  EXPECT_TRUE(original.ok());
  EXPECT_EQ(*original, 42);
}

TEST_F(ResultTest, MoveAssignment) {
  Result<std::string> original("hello");
  Result<std::string> moved("");
  moved = std::move(original);
  EXPECT_TRUE(moved.ok());
  EXPECT_EQ(*moved, "hello");
}

TEST_F(ResultTest, StatusAccessOnSuccess) {
  Result<int> r(42);
  const auto& status = r.status();
  EXPECT_TRUE(status.ok());
}

TEST_F(ResultTest, StatusAccessOnError) {
  Result<int> err(Status(StatusCode::kNotFound, "not found"));
  const auto& status = err.status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kNotFound);
  EXPECT_EQ(status.message(), "not found");
}

TEST_F(ResultTest, StatusMoveOnError) {
  Result<int> err(Status(StatusCode::kNotFound, "not found"));
  auto status = std::move(err).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kNotFound);
  EXPECT_EQ(status.message(), "not found");
}

TEST_F(ResultTest, ValueMethodStillWorks) {
  Result<int> r(42);
  EXPECT_EQ(r.value(), 42);
}

TEST_F(ResultTest, ValueMethodConst) {
  const Result<int> r(42);
  EXPECT_EQ(r.value(), 42);
}

TEST_F(ResultTest, ValueMethodRvalue) {
  Result<std::string> r("hello");
  std::string moved = std::move(r).value();
  EXPECT_EQ(moved, "hello");
}

TEST_F(ResultTest, ValueMethodThrowsOnError) {
  Result<int> err(Status(StatusCode::kInternal, "fail"));
  EXPECT_THROW(err.value(), std::bad_expected_access<Status>);
}

