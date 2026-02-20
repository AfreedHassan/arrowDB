#include "arrow/filter.h"
#include <gtest/gtest.h>

using namespace arrow;

TEST(FilterTest, EqString) {
  Metadata meta{{"category", std::string("tech")}};
  EXPECT_TRUE(MetadataFilter::Eq("category", std::string("tech"))(meta));
  EXPECT_FALSE(MetadataFilter::Eq("category", std::string("science"))(meta));
}

TEST(FilterTest, EqInt64) {
  Metadata meta{{"year", int64_t(2024)}};
  EXPECT_TRUE(MetadataFilter::Eq("year", int64_t(2024))(meta));
  EXPECT_FALSE(MetadataFilter::Eq("year", int64_t(2023))(meta));
}

TEST(FilterTest, EqDouble) {
  Metadata meta{{"score", 0.95}};
  EXPECT_TRUE(MetadataFilter::Eq("score", 0.95)(meta));
  EXPECT_FALSE(MetadataFilter::Eq("score", 0.5)(meta));
}

TEST(FilterTest, EqBool) {
  Metadata meta{{"active", true}};
  EXPECT_TRUE(MetadataFilter::Eq("active", true)(meta));
  EXPECT_FALSE(MetadataFilter::Eq("active", false)(meta));
}

TEST(FilterTest, Neq) {
  Metadata meta{{"category", std::string("tech")}};
  EXPECT_FALSE(MetadataFilter::Neq("category", std::string("tech"))(meta));
  EXPECT_TRUE(MetadataFilter::Neq("category", std::string("science"))(meta));
}

TEST(FilterTest, GtInt) {
  Metadata meta{{"year", int64_t(2024)}};
  EXPECT_TRUE(MetadataFilter::Gt("year", int64_t(2020))(meta));
  EXPECT_FALSE(MetadataFilter::Gt("year", int64_t(2024))(meta));
  EXPECT_FALSE(MetadataFilter::Gt("year", int64_t(2025))(meta));
}

TEST(FilterTest, GteInt) {
  Metadata meta{{"year", int64_t(2024)}};
  EXPECT_TRUE(MetadataFilter::Gte("year", int64_t(2024))(meta));
  EXPECT_TRUE(MetadataFilter::Gte("year", int64_t(2020))(meta));
  EXPECT_FALSE(MetadataFilter::Gte("year", int64_t(2025))(meta));
}

TEST(FilterTest, LtDouble) {
  Metadata meta{{"score", 0.3}};
  EXPECT_TRUE(MetadataFilter::Lt("score", 0.5)(meta));
  EXPECT_FALSE(MetadataFilter::Lt("score", 0.3)(meta));
  EXPECT_FALSE(MetadataFilter::Lt("score", 0.1)(meta));
}

TEST(FilterTest, LteDouble) {
  Metadata meta{{"score", 0.5}};
  EXPECT_TRUE(MetadataFilter::Lte("score", 0.5)(meta));
  EXPECT_TRUE(MetadataFilter::Lte("score", 0.9)(meta));
  EXPECT_FALSE(MetadataFilter::Lte("score", 0.3)(meta));
}

TEST(FilterTest, CrossTypeNumeric) {
  Metadata meta{{"year", int64_t(2024)}};
  EXPECT_TRUE(MetadataFilter::Gt("year", 2020.0)(meta));
  EXPECT_FALSE(MetadataFilter::Gt("year", 2024.0)(meta));

  Metadata meta2{{"score", 2024.0}};
  EXPECT_TRUE(MetadataFilter::Gt("score", int64_t(2020))(meta2));
}

TEST(FilterTest, InValues) {
  Metadata meta{{"tag", std::string("b")}};
  auto f = MetadataFilter::In("tag", {MetadataValue(std::string("a")),
                               MetadataValue(std::string("b")),
                               MetadataValue(std::string("c"))});
  EXPECT_TRUE(f(meta));

  Metadata meta2{{"tag", std::string("d")}};
  EXPECT_FALSE(f(meta2));
}

TEST(FilterTest, And) {
  Metadata meta{{"category", std::string("tech")}, {"year", int64_t(2024)}};
  auto f = MetadataFilter::And(MetadataFilter::Eq("category", std::string("tech")),
                       MetadataFilter::Gt("year", int64_t(2020)));
  EXPECT_TRUE(f(meta));

  Metadata meta2{{"category", std::string("tech")}, {"year", int64_t(2019)}};
  EXPECT_FALSE(f(meta2));
}

TEST(FilterTest, Or) {
  auto f = MetadataFilter::Or(MetadataFilter::Eq("category", std::string("tech")),
                      MetadataFilter::Eq("category", std::string("sci")));
  EXPECT_TRUE(f(Metadata{{"category", std::string("tech")}}));
  EXPECT_TRUE(f(Metadata{{"category", std::string("sci")}}));
  EXPECT_FALSE(f(Metadata{{"category", std::string("art")}}));
}

TEST(FilterTest, Not) {
  auto f = MetadataFilter::Not(MetadataFilter::Eq("category", std::string("tech")));
  EXPECT_FALSE(f(Metadata{{"category", std::string("tech")}}));
  EXPECT_TRUE(f(Metadata{{"category", std::string("sci")}}));
}

TEST(FilterTest, NestedLogic) {
  auto f = MetadataFilter::And(
      MetadataFilter::Or(MetadataFilter::Eq("category", std::string("tech")),
                 MetadataFilter::Eq("category", std::string("sci"))),
      MetadataFilter::Not(MetadataFilter::Eq("active", false)));

  EXPECT_TRUE(f(Metadata{{"category", std::string("tech")}, {"active", true}}));
  EXPECT_FALSE(f(Metadata{{"category", std::string("tech")}, {"active", false}}));
  EXPECT_FALSE(f(Metadata{{"category", std::string("art")}, {"active", true}}));
}

TEST(FilterTest, MissingField) {
  Metadata meta{{"other", std::string("value")}};
  EXPECT_FALSE(MetadataFilter::Eq("nonexistent", std::string("x"))(meta));
  EXPECT_FALSE(MetadataFilter::Gt("nonexistent", int64_t(0))(meta));
  EXPECT_FALSE(MetadataFilter::In("nonexistent", {MetadataValue(std::string("a"))})(meta));
}

TEST(FilterTest, VariadicAnd) {
  auto f = MetadataFilter::And({
      MetadataFilter::Eq("a", int64_t(1)),
      MetadataFilter::Eq("b", int64_t(2)),
      MetadataFilter::Eq("c", int64_t(3))
  });
  EXPECT_TRUE(f(Metadata{{"a", int64_t(1)}, {"b", int64_t(2)}, {"c", int64_t(3)}}));
  EXPECT_FALSE(f(Metadata{{"a", int64_t(1)}, {"b", int64_t(2)}, {"c", int64_t(99)}}));
}

TEST(FilterTest, EmptyMetadata) {
  Metadata empty;
  EXPECT_FALSE(MetadataFilter::Eq("field", std::string("x"))(empty));
  EXPECT_FALSE(MetadataFilter::Gt("field", int64_t(0))(empty));
  EXPECT_FALSE(MetadataFilter::Neq("field", std::string("x"))(empty));
}

TEST(FilterTest, WhereStringLength) {
  auto f = MetadataFilter::Where<std::string>("name", [](const std::string& s) {
    return s.size() > 5;
  });
  EXPECT_TRUE(f(Metadata{{"name", std::string("longname")}}));
  EXPECT_FALSE(f(Metadata{{"name", std::string("hi")}}));
}

TEST(FilterTest, WhereInt64Range) {
  auto f = MetadataFilter::Where<int64_t>("year", [](int64_t y) {
    return y > 2020 && y < 2025;
  });
  EXPECT_TRUE(f(Metadata{{"year", int64_t(2023)}}));
  EXPECT_FALSE(f(Metadata{{"year", int64_t(2019)}}));
  EXPECT_FALSE(f(Metadata{{"year", int64_t(2025)}}));
}

TEST(FilterTest, WhereDouble) {
  auto f = MetadataFilter::Where<double>("score", [](double d) { return d > 0.5; });
  EXPECT_TRUE(f(Metadata{{"score", 0.9}}));
  EXPECT_FALSE(f(Metadata{{"score", 0.3}}));
}

TEST(FilterTest, WhereBool) {
  auto f = MetadataFilter::Where<bool>("active", [](bool b) { return b; });
  EXPECT_TRUE(f(Metadata{{"active", true}}));
  EXPECT_FALSE(f(Metadata{{"active", false}}));
}

TEST(FilterTest, WhereMissingField) {
  auto f = MetadataFilter::Where<std::string>("name", [](const std::string&) { return true; });
  EXPECT_FALSE(f(Metadata{}));
  EXPECT_FALSE(f(Metadata{{"other", std::string("x")}}));
}

TEST(FilterTest, WhereWrongType) {
  auto f = MetadataFilter::Where<std::string>("val", [](const std::string&) { return true; });
  EXPECT_FALSE(f(Metadata{{"val", int64_t(42)}}));
}

TEST(FilterTest, WhereComposedWithDSL) {
  auto f = MetadataFilter::And(
      MetadataFilter::Where<std::string>("name", [](const std::string& s) { return s.size() > 3; }),
      MetadataFilter::Eq("category", std::string("tech"))
  );
  EXPECT_TRUE(f(Metadata{{"name", std::string("longname")}, {"category", std::string("tech")}}));
  EXPECT_FALSE(f(Metadata{{"name", std::string("hi")}, {"category", std::string("tech")}}));
  EXPECT_FALSE(f(Metadata{{"name", std::string("longname")}, {"category", std::string("art")}}));
}

TEST(FilterTest, VariadicOr) {
  auto f = MetadataFilter::Or({
      MetadataFilter::Eq("tag", std::string("a")),
      MetadataFilter::Eq("tag", std::string("b")),
      MetadataFilter::Eq("tag", std::string("c"))
  });
  EXPECT_TRUE(f(Metadata{{"tag", std::string("a")}}));
  EXPECT_TRUE(f(Metadata{{"tag", std::string("b")}}));
  EXPECT_TRUE(f(Metadata{{"tag", std::string("c")}}));
  EXPECT_FALSE(f(Metadata{{"tag", std::string("d")}}));
}

TEST(FilterTest, CompareNumericNonNumeric) {
  // compareNumeric returns 0 for non-numeric types, so:
  // Gt (>0) and Lt (<0) return false; Gte (>=0) and Lte (<=0) return true
  Metadata meta{{"field", std::string("hello")}};
  EXPECT_FALSE(MetadataFilter::Gt("field", std::string("abc"))(meta));
  EXPECT_FALSE(MetadataFilter::Lt("field", std::string("xyz"))(meta));
  EXPECT_TRUE(MetadataFilter::Gte("field", std::string("abc"))(meta));
  EXPECT_TRUE(MetadataFilter::Lte("field", std::string("xyz"))(meta));
}
