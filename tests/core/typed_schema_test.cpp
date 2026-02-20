// Copyright 2025 ArrowDB
#include "arrow/typed_schema.h"
#include "arrow/arrow.h"
#include "test_util.h"
#include <gtest/gtest.h>

using namespace arrow;

// ── Test schema definitions ─────────────────────────────────

using TestSchema = Schema<
    Field<"category", FieldType::String, true>,
    Field<"score",    FieldType::Double, true>,
    Field<"count",    FieldType::Int64,  true>
>;

using MixedSchema = Schema<
    Field<"name",     FieldType::String, true>,
    Field<"rating",   FieldType::Double>,
    Field<"active",   FieldType::Bool>
>;

// ── Helper: build a SearchResult from scratch ───────────────

static SearchResult makeResult(
    std::vector<std::pair<std::string, Metadata>> docs,
    float baseScore = 0.9f) {
    SearchResult sr;
    float s = baseScore;
    for (auto& [id, meta] : docs) {
        sr.hits.push_back(ScoredDocument{
            .id = std::move(id),
            .score = s,
            .metadata = std::move(meta),
        });
        s -= 0.1f;
    }
    return sr;
}

// ── Tests ───────────────────────────────────────────────────

class TypedSchemaTest : public ::testing::Test {};

TEST_F(TypedSchemaTest, BindRequiredFields) {
    auto sr = makeResult({
        {"doc1", {{"category", "tech"}, {"score", 0.95}, {"count", int64_t(42)}}},
    });

    auto results = bind<TestSchema>(sr);
    ASSERT_EQ(results.hits.size(), 1u);

    auto& hit = results.hits[0];
    EXPECT_EQ(hit.get<"category">(), "tech");
    EXPECT_DOUBLE_EQ(hit.get<"score">(), 0.95);
    EXPECT_EQ(hit.get<"count">(), 42);
}

TEST_F(TypedSchemaTest, BindOptionalFieldPresent) {
    auto sr = makeResult({
        {"doc1", {{"name", "Alice"}, {"rating", 4.5}, {"active", true}}},
    });

    auto results = bind<MixedSchema>(sr);
    ASSERT_EQ(results.hits.size(), 1u);

    auto& hit = results.hits[0];
    EXPECT_EQ(hit.get<"name">(), "Alice");
    ASSERT_TRUE(hit.get<"rating">().has_value());
    EXPECT_DOUBLE_EQ(*hit.get<"rating">(), 4.5);
    ASSERT_TRUE(hit.get<"active">().has_value());
    EXPECT_TRUE(*hit.get<"active">());
}

TEST_F(TypedSchemaTest, BindOptionalFieldMissing) {
    auto sr = makeResult({
        {"doc1", {{"name", "Bob"}}},
    });

    auto results = bind<MixedSchema>(sr);
    ASSERT_EQ(results.hits.size(), 1u);

    auto& hit = results.hits[0];
    EXPECT_EQ(hit.get<"name">(), "Bob");
    EXPECT_FALSE(hit.get<"rating">().has_value());
    EXPECT_FALSE(hit.get<"active">().has_value());
}

TEST_F(TypedSchemaTest, BindOptionalFieldWrongType) {
    auto sr = makeResult({
        {"doc1", {{"name", "Carol"}, {"rating", "not_a_double"}, {"active", int64_t(1)}}},
    });

    auto results = bind<MixedSchema>(sr);
    ASSERT_EQ(results.hits.size(), 1u);

    auto& hit = results.hits[0];
    EXPECT_EQ(hit.get<"name">(), "Carol");
    EXPECT_FALSE(hit.get<"rating">().has_value());
    EXPECT_FALSE(hit.get<"active">().has_value());
}

TEST_F(TypedSchemaTest, BindRequiredFieldMissing) {
    auto sr = makeResult({
        {"doc1", {{"category", "tech"}, {"score", 0.5}}},
        // "count" is missing but required
    });

    EXPECT_THROW(bind<TestSchema>(sr), std::runtime_error);
}

TEST_F(TypedSchemaTest, BindRequiredFieldWrongType) {
    auto sr = makeResult({
        {"doc1", {{"category", "tech"}, {"score", 0.5}, {"count", "not_an_int"}}},
    });

    EXPECT_THROW(bind<TestSchema>(sr), std::bad_variant_access);
}

TEST_F(TypedSchemaTest, TryBindError) {
    auto sr = makeResult({
        {"doc1", {{"category", "tech"}}},
        // "score" and "count" missing
    });

    auto result = tryBind<TestSchema>(sr);
    EXPECT_FALSE(result.ok());
}

TEST_F(TypedSchemaTest, TryBindSuccess) {
    auto sr = makeResult({
        {"doc1", {{"category", "tech"}, {"score", 0.8}, {"count", int64_t(10)}}},
    });

    auto result = tryBind<TestSchema>(sr);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->hits.size(), 1u);
    EXPECT_EQ(result->hits[0].get<"category">(), "tech");
}

TEST_F(TypedSchemaTest, PreservesIdAndScore) {
    auto sr = makeResult({
        {"my-vec-id", {{"category", "art"}, {"score", 1.0}, {"count", int64_t(7)}}},
    }, 0.75f);

    auto results = bind<TestSchema>(sr);
    ASSERT_EQ(results.hits.size(), 1u);
    EXPECT_EQ(results.hits[0].id, "my-vec-id");
    EXPECT_FLOAT_EQ(results.hits[0].score, 0.75f);
}

TEST_F(TypedSchemaTest, EmptyResult) {
    SearchResult sr;
    auto results = bind<TestSchema>(sr);
    EXPECT_TRUE(results.hits.empty());
}

TEST_F(TypedSchemaTest, ToRuntimeSchema) {
    auto schema = TestSchema::toRuntimeSchema();
    ASSERT_EQ(schema.fields.size(), 3u);

    EXPECT_EQ(schema.fields[0].name, "category");
    EXPECT_EQ(schema.fields[0].type, FieldType::String);
    EXPECT_TRUE(schema.fields[0].required);

    EXPECT_EQ(schema.fields[1].name, "score");
    EXPECT_EQ(schema.fields[1].type, FieldType::Double);
    EXPECT_TRUE(schema.fields[1].required);

    EXPECT_EQ(schema.fields[2].name, "count");
    EXPECT_EQ(schema.fields[2].type, FieldType::Int64);
    EXPECT_TRUE(schema.fields[2].required);
}

TEST_F(TypedSchemaTest, MultipleHits) {
    auto sr = makeResult({
        {"d1", {{"category", "A"}, {"score", 1.0}, {"count", int64_t(1)}}},
        {"d2", {{"category", "B"}, {"score", 2.0}, {"count", int64_t(2)}}},
        {"d3", {{"category", "C"}, {"score", 3.0}, {"count", int64_t(3)}}},
    });

    auto results = bind<TestSchema>(sr);
    ASSERT_EQ(results.hits.size(), 3u);
    EXPECT_EQ(results.hits[0].get<"category">(), "A");
    EXPECT_EQ(results.hits[1].get<"category">(), "B");
    EXPECT_EQ(results.hits[2].get<"category">(), "C");
    EXPECT_EQ(results.hits[2].get<"count">(), 3);
}

TEST_F(TypedSchemaTest, QueryFreeFunction) {
    constexpr uint32_t dim = 8;
    CollectionConfig config{
        .name = "typed_test",
        .dimensions = dim,
        .space = Space::Cosine,
    };
    Collection col(config);

    std::mt19937 gen(42);

    // Insert vectors with metadata
    for (int i = 0; i < 5; ++i) {
        auto vec = arrow::testing::RandomVector(dim, gen);
        Metadata meta{
            {"category", std::string(i % 2 == 0 ? "even" : "odd")},
            {"score", static_cast<double>(i) * 0.1},
            {"count", static_cast<int64_t>(i)},
        };
        auto status = col.insert("v" + std::to_string(i), vec, std::move(meta));
        ASSERT_TRUE(status.ok()) << status.message();
    }

    // Query using the typed free function
    auto queryVec = arrow::testing::RandomVector(dim, gen);
    auto results = arrow::query<TestSchema>(col, queryVec, 3);
    ASSERT_EQ(results.hits.size(), 3u);

    for (auto& hit : results.hits) {
        // All required fields must be present and correctly typed
        std::string cat = hit.get<"category">();
        EXPECT_TRUE(cat == "even" || cat == "odd");

        double s = hit.get<"score">();
        EXPECT_GE(s, 0.0);
        EXPECT_LE(s, 0.4);

        int64_t c = hit.get<"count">();
        EXPECT_GE(c, 0);
        EXPECT_LE(c, 4);
    }
}

// Compile-time safety: uncommenting the line below should produce a compile error.
// static_assert requires ValidFieldName to be satisfied.
//
// void compileErrorDemo() {
//     Hit<TestSchema> hit;
//     hit.get<"typo">();  // ERROR: constraint not satisfied
// }
