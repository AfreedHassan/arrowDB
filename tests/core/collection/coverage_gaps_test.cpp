#include "common.h"
#include "arrow/client.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Dimension mismatch in search/query ──────────────────────

TEST_F(CollectionTest, SearchDimensionMismatch) {
  CollectionConfig cfg{.name = "dim_mismatch", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
  col.insert("v1", vec);

  // Wrong query dimension
  std::vector<float> query = {1.0f, 2.0f};
  auto results = col.search(query, 5);
  EXPECT_TRUE(results.empty());
}

TEST_F(CollectionTest, SearchWithFilterDimensionMismatch) {
  CollectionConfig cfg{.name = "filter_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  col.setMetadata("v1", {{"tag", std::string("a")}});

  auto filter = MetadataFilter::Eq("tag", std::string("a"));
  std::vector<float> badQuery = {1.0f, 2.0f};
  auto results = col.search(badQuery, 5, filter);
  EXPECT_TRUE(results.empty());
}

TEST_F(CollectionTest, SearchWithFilterEmptyCollection) {
  CollectionConfig cfg{.name = "filter_empty", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  auto filter = MetadataFilter::Eq("tag", std::string("a"));
  std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
  auto results = col.search(query, 5, filter);
  EXPECT_TRUE(results.empty());
}

TEST_F(CollectionTest, QueryVecDimensionMismatch) {
  CollectionConfig cfg{.name = "query_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<float> badQuery = {1.0f, 2.0f};
  auto result = col.query(badQuery, 5);
  EXPECT_TRUE(result.hits.empty());
}

TEST_F(CollectionTest, QueryWithFilterDimensionMismatch) {
  CollectionConfig cfg{.name = "qf_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  col.setMetadata("v1", {{"tag", std::string("a")}});

  auto filter = MetadataFilter::Eq("tag", std::string("a"));
  std::vector<float> badQuery = {1.0f};
  auto result = col.query(badQuery, 5, filter);
  EXPECT_TRUE(result.hits.empty());
}

TEST_F(CollectionTest, QueryWithFilterEmptyIndex) {
  CollectionConfig cfg{.name = "qf_empty", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  auto filter = MetadataFilter::Eq("tag", std::string("a"));
  std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
  auto result = col.query(query, 5, filter);
  EXPECT_TRUE(result.hits.empty());
}

// ── PreparedFilter dimension mismatch ───────────────────────

TEST_F(CollectionTest, PreparedFilterSearchDimensionMismatch) {
  CollectionConfig cfg{.name = "pf_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  col.setMetadata("v1", {{"x", int64_t(1)}});

  auto pf = col.prepareFilter(MetadataFilter::Eq("x", int64_t(1)));

  std::vector<float> badQuery = {1.0f};
  auto results = col.search(badQuery, 5, pf);
  EXPECT_TRUE(results.empty());
}

TEST_F(CollectionTest, PreparedFilterQueryDimensionMismatch) {
  CollectionConfig cfg{.name = "pf_qdim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  col.setMetadata("v1", {{"x", int64_t(1)}});

  auto pf = col.prepareFilter(MetadataFilter::Eq("x", int64_t(1)));

  std::vector<float> badQuery = {1.0f};
  auto result = col.query(badQuery, 5, pf);
  EXPECT_TRUE(result.hits.empty());
}

// ── getMetadata for non-existent vector ─────────────────────

TEST_F(CollectionTest, GetMetadataNonexistent) {
  CollectionConfig cfg{.name = "gm_nonexist", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  auto result = col.getMetadata("missing");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

// ── getMetadata for vector without metadata (empty) ─────────

TEST_F(CollectionTest, GetMetadataEmptyMeta) {
  CollectionConfig cfg{.name = "gm_empty", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  auto result = col.getMetadata("v1");
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result.value().empty());
}

// ── setMetadata with schema validation ──────────────────────

TEST_F(CollectionTest, SetMetadataSchemaValidation) {
  MetadataSchema schema;
  schema.field("category", FieldType::String, true);

  CollectionConfig cfg{
    .name = "schema_setmeta",
    .dimensions = 4,
    .space = Space::Cosine,
    .schema = schema
  };
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  // Wrong type for schema field
  Metadata badMeta{{"category", int64_t(42)}};
  auto s = col.setMetadata("v1", badMeta);
  EXPECT_FALSE(s.ok());
}

// ── searchBatch with dimension mismatch ─────────────────────

TEST_F(CollectionTest, SearchBatchDimensionMismatch) {
  CollectionConfig cfg{.name = "sb_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::vector<float>> queries = {
    {1.0f, 2.0f, 3.0f, 4.0f},
    {1.0f, 2.0f}  // wrong dimension
  };
  auto result = col.searchBatch(queries, 5);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kDimensionMismatch);
}

// ── searchBatch multi-threaded path (needs >1 query) ────────

TEST_F(CollectionTest, SearchBatchMultipleQueries) {
  CollectionConfig cfg{.name = "sb_multi", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 100; ++i) {
    col.insert(std::to_string(i), RandomVector(4, gen));
  }

  // Create many queries to trigger parallel search
  std::vector<std::vector<float>> queries;
  for (int i = 0; i < 10; ++i) {
    queries.push_back(RandomVector(4, gen));
  }

  auto result = col.searchBatch(queries, 5);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().size(), 10u);
  for (const auto& r : result.value()) {
    EXPECT_EQ(r.size(), 5u);
  }
}

// ── printStats ──────────────────────────────────────────────

TEST_F(CollectionTest, PrintStats) {
  CollectionConfig cfg{.name = "print_stats", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_NO_THROW(col.printStats());
}

// ── Document-based insert ───────────────────────────────────

TEST_F(CollectionTest, InsertDocumentWithMetadata) {
  CollectionConfig cfg{.name = "doc_insert_meta", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  Document doc;
  doc.embedding = {1.0f, 2.0f, 3.0f, 4.0f};
  doc.metadata = {{"tag", std::string("test")}};

  auto result = col.insert(doc);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().size(), 36u);  // UUID
  EXPECT_EQ(col.size(), 1);

  auto meta = col.getMetadata(result.value());
  ASSERT_TRUE(meta.ok());
  EXPECT_EQ(std::get<std::string>(meta.value().at("tag")), "test");
}

TEST_F(CollectionTest, InsertDocumentExplicitID) {
  CollectionConfig cfg{.name = "doc_insert_eid", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  Document doc;
  doc.id = "my-doc";
  doc.embedding = {1.0f, 2.0f, 3.0f, 4.0f};

  auto result = col.insert(doc);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), "my-doc");
}

// ── Persistence: WAL replay on dirty shutdown ───────────────

TEST_F(CollectionTest, WalReplayOnDirtyShutdown) {
  std::string path = GetTestPath("dirty_shutdown");

  // Create persistent collection, insert data
  {
    CollectionConfig cfg{.name = "dirty", .dimensions = 4, .space = Space::Cosine};
    auto cr = Collection::create(cfg, path);
    ASSERT_TRUE(cr.ok());
    Collection col = std::move(cr.value());

    ASSERT_TRUE(col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f}).ok());
    ASSERT_TRUE(col.insert("v2", {5.0f, 6.0f, 7.0f, 8.0f}).ok());
    ASSERT_TRUE(col.save(path).ok());
  }

  // Simulate dirty shutdown
  auto metaPath = std::filesystem::path(path) / "meta.json";
  {
    std::ifstream in(metaPath);
    json j;
    in >> j;
    in.close();
    j["recovery"]["cleanShutdown"] = false;
    std::ofstream out(metaPath);
    out << j.dump(2);
  }

  // Load - should trigger full WAL replay
  auto loadResult = Collection::load(path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection col = std::move(loadResult.value());

  EXPECT_EQ(col.size(), 2u);
  EXPECT_TRUE(col.recoveredFromWal());
}

// ── Persistence: WAL replay with INSERT after checkpoint ────

TEST_F(CollectionTest, WalReplayIncrementalInsert) {
  std::string path = GetTestPath("incremental_wal");

  // Create, insert 2, save (checkpoint)
  {
    CollectionConfig cfg{.name = "incr", .dimensions = 4, .space = Space::Cosine};
    auto cr = Collection::create(cfg, path);
    ASSERT_TRUE(cr.ok());
    Collection col = std::move(cr.value());

    col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
    col.insert("v2", {5.0f, 6.0f, 7.0f, 8.0f});
    col.save(path);

    // Insert more after checkpoint (these are only in WAL)
    col.insert("v3", {9.0f, 10.0f, 11.0f, 12.0f});
  }

  // Load - should replay v3 from WAL
  auto loadResult = Collection::load(path);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection col = std::move(loadResult.value());

  EXPECT_EQ(col.size(), 3u);
  auto v3 = col.get("v3");
  ASSERT_TRUE(v3.ok());
}

// ── Persistence: WAL replay DELETE ──────────────────────────

TEST_F(CollectionTest, DeletePersistsAcrossSaveLoad) {
  std::string path = GetTestPath("delete_persist");

  {
    CollectionConfig cfg{.name = "del", .dimensions = 4, .space = Space::Cosine};
    auto cr = Collection::create(cfg, path);
    ASSERT_TRUE(cr.ok());
    Collection col = std::move(cr.value());

    col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
    col.insert("v2", {5.0f, 6.0f, 7.0f, 8.0f});
    col.remove("v1");
    col.save(path);
  }

  auto loadResult = Collection::load(path);
  ASSERT_TRUE(loadResult.ok());
  Collection col = std::move(loadResult.value());

  // v1 should be gone after load
  auto v1 = col.get("v1");
  EXPECT_FALSE(v1.ok());

  auto v2 = col.get("v2");
  EXPECT_TRUE(v2.ok());
}

// ── Optimize (SQ + BFS reorder) ─────────────────────────────

TEST_F(CollectionTest, OptimizeWithQuantizationSearchable) {
  CollectionConfig cfg{
    .name = "optimize",
    .dimensions = 4,
    .space = Space::Cosine,
    .index_config = {
      .quantization = Quantization::INT8,
    }
  };
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 100; ++i) {
    col.insert(std::to_string(i), RandomVector(4, gen));
  }

  auto s = col.optimize();
  EXPECT_TRUE(s.ok());

  // Search should still work after optimization
  auto results = col.search(RandomVector(4, gen), 5);
  EXPECT_EQ(results.size(), 5u);
}

TEST_F(CollectionTest, OptimizeWithoutQuantization) {
  CollectionConfig cfg{.name = "no_quant", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  auto s = col.optimize();
  EXPECT_TRUE(s.ok());  // No-op
}

TEST_F(CollectionTest, OptimizeEmptyCollection) {
  CollectionConfig cfg{
    .name = "opt_empty",
    .dimensions = 4,
    .space = Space::Cosine,
    .index_config = {.quantization = Quantization::INT8}
  };
  Collection col(cfg);

  auto s = col.optimize();
  EXPECT_TRUE(s.ok());  // No-op
}

// ── Persistence: bad schema version in meta.json ────────────

TEST_F(CollectionTest, LoadFutureSchemaVersion) {
  std::string path = GetTestPath("future_schema");
  std::filesystem::create_directories(path);

  std::ofstream(std::filesystem::path(path) / "meta.json") << R"({
    "name": "test",
    "dimensions": 128,
    "space": "Cosine",
    "dtype": "Float32",
    "idxType": "HNSW",
    "schemaVersion": 999,
    "hnsw": {"maxElements": 10000, "M": 16, "efConstruction": 200},
    "recovery": {"lastPersistedLsn": 0, "lastPersistedTxid": 0, "cleanShutdown": true}
  })";

  auto result = Collection::load(path);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kCorruption);
}

// ── Persistence: invalid dimensions in meta.json ────────────

TEST_F(CollectionTest, LoadInvalidDimensions) {
  std::string path = GetTestPath("bad_dims");
  std::filesystem::create_directories(path);

  std::ofstream(std::filesystem::path(path) / "meta.json") << R"({
    "name": "test",
    "dimensions": 0,
    "space": "Cosine",
    "dtype": "Float32",
    "idxType": "HNSW",
    "schemaVersion": 3,
    "hnsw": {"maxElements": 10000, "M": 16, "efConstruction": 200},
    "recovery": {"lastPersistedLsn": 0, "lastPersistedTxid": 0, "cleanShutdown": true}
  })";

  auto result = Collection::load(path);
  EXPECT_FALSE(result.ok());
}

// ── Persistence: invalid M in meta.json ─────────────────────

TEST_F(CollectionTest, LoadInvalidM) {
  std::string path = GetTestPath("bad_m");
  std::filesystem::create_directories(path);

  std::ofstream(std::filesystem::path(path) / "meta.json") << R"({
    "name": "test",
    "dimensions": 128,
    "space": "Cosine",
    "dtype": "Float32",
    "idxType": "HNSW",
    "schemaVersion": 3,
    "hnsw": {"maxElements": 10000, "M": 0, "efConstruction": 200},
    "recovery": {"lastPersistedLsn": 0, "lastPersistedTxid": 0, "cleanShutdown": true}
  })";

  auto result = Collection::load(path);
  EXPECT_FALSE(result.ok());
}

// ── Persistence: invalid efConstruction ─────────────────────

TEST_F(CollectionTest, LoadInvalidEfConstruction) {
  std::string path = GetTestPath("bad_ef");
  std::filesystem::create_directories(path);

  std::ofstream(std::filesystem::path(path) / "meta.json") << R"({
    "name": "test",
    "dimensions": 128,
    "space": "Cosine",
    "dtype": "Float32",
    "idxType": "HNSW",
    "schemaVersion": 3,
    "hnsw": {"maxElements": 10000, "M": 16, "efConstruction": 0},
    "recovery": {"lastPersistedLsn": 0, "lastPersistedTxid": 0, "cleanShutdown": true}
  })";

  auto result = Collection::load(path);
  EXPECT_FALSE(result.ok());
}

// ── Client: dropCollection non-existent ─────────────────────

TEST_F(CollectionTest, ClientDropNonexistent) {
  ClientOptions options{.dataDir = testDir};
  Client db(options);

  auto status = db.dropCollection("missing");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), utils::StatusCode::kNotFound);
}

// ── Client: move semantics ──────────────────────────────────

TEST_F(CollectionTest, ClientMoveConstructor) {
  ClientOptions options{.dataDir = testDir};
  Client db1(options);

  CollectionConfig cfg{.name = "test", .dimensions = 4, .space = Space::Cosine};
  db1.createCollection("test", cfg);

  Client db2(std::move(db1));
  EXPECT_TRUE(db2.hasCollection("test"));
}

// ── Non-4-aligned dimensions (NEON remainder loops) ─────────

TEST_F(CollectionTest, NonAlignedDimensions) {
  // Dimension 13 triggers NEON remainder loops
  CollectionConfig cfg{.name = "non_aligned", .dimensions = 13, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    col.insert(std::to_string(i), RandomVector(13, gen));
  }

  auto results = col.search(RandomVector(13, gen), 5);
  EXPECT_EQ(results.size(), 5u);
}

TEST_F(CollectionTest, NonAlignedDimensionsL2) {
  CollectionConfig cfg{.name = "non_aligned_l2", .dimensions = 17, .space = Space::L2};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    col.insert(std::to_string(i), RandomVector(17, gen));
  }

  auto results = col.search(RandomVector(17, gen), 5);
  EXPECT_EQ(results.size(), 5u);
}

TEST_F(CollectionTest, NonAlignedDimensionsIP) {
  CollectionConfig cfg{.name = "non_aligned_ip", .dimensions = 11, .space = Space::InnerProduct};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    col.insert(std::to_string(i), RandomVector(11, gen));
  }

  auto results = col.search(RandomVector(11, gen), 5);
  EXPECT_EQ(results.size(), 5u);
}

// ── Search with filter on vectors without metadata ──────────

TEST_F(CollectionTest, FilteredSearchVectorsWithoutMetadata) {
  CollectionConfig cfg{.name = "filter_no_meta", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  // Insert some vectors with metadata and some without
  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  col.insert("v2", {5.0f, 6.0f, 7.0f, 8.0f});
  col.setMetadata("v1", {{"category", std::string("tech")}});
  // v2 has no metadata

  // Filter that passes on empty metadata
  auto filter = MetadataFilter::Not(MetadataFilter::Eq("category", std::string("tech")));
  auto results = col.search({1.0f, 2.0f, 3.0f, 4.0f}, 5, filter);

  // v2 should pass (it has empty metadata, NOT tech)
  EXPECT_FALSE(results.empty());
}

// ── Query with filter returning results with metadata ───────

TEST_F(CollectionTest, QueryWithFilterReturnsMetadata) {
  CollectionConfig cfg{.name = "qf_meta", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  col.setMetadata("v1", {{"tag", std::string("a")}, {"score", 0.95}});
  col.insert("v2", {5.0f, 6.0f, 7.0f, 8.0f});
  col.setMetadata("v2", {{"tag", std::string("b")}, {"score", 0.5}});

  auto filter = MetadataFilter::Eq("tag", std::string("a"));
  auto result = col.query({1.0f, 2.0f, 3.0f, 4.0f}, 5, filter);

  EXPECT_FALSE(result.hits.empty());
  for (const auto& hit : result.hits) {
    EXPECT_EQ(std::get<std::string>(hit.metadata.at("tag")), "a");
  }
}

// ── InsertBatch with metadata validation failures ───────────

TEST_F(CollectionTest, InsertBatchMetadataValidation) {
  MetadataSchema schema;
  schema.field("category", FieldType::String, true);

  CollectionConfig cfg{
    .name = "batch_schema",
    .dimensions = 4,
    .space = Space::Cosine,
    .schema = schema
  };
  Collection col(cfg);

  std::vector<Document> docs = {
    {"d1", {1.0f, 2.0f, 3.0f, 4.0f}, {{"category", std::string("ok")}}},
    {"d2", {5.0f, 6.0f, 7.0f, 8.0f}, {{"category", int64_t(42)}}},  // wrong type
  };

  auto result = col.insertBatch(std::move(docs));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, 1u);
  EXPECT_EQ(result.value().failureCount, 1u);
}

// ── InsertBatch with dimension mismatch ─────────────────────

TEST_F(CollectionTest, InsertBatchDimensionMismatch) {
  CollectionConfig cfg{.name = "batch_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<Document> docs = {
    {"d1", {1.0f, 2.0f, 3.0f, 4.0f}, {}},
    {"d2", {1.0f, 2.0f}, {}},  // wrong dimension
  };

  auto result = col.insertBatch(std::move(docs));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, 1u);
  EXPECT_EQ(result.value().failureCount, 1u);
}

// ── Persistence: save/load with all metadata field types ────

TEST_F(CollectionTest, PersistenceAllFieldTypes) {
  std::string path = GetTestPath("all_field_types");

  MetadataSchema schema;
  schema.field("intField", FieldType::Int64, false)
        .field("dblField", FieldType::Double, false)
        .field("strField", FieldType::String, false)
        .field("boolField", FieldType::Bool, false);

  CollectionConfig cfg{
    .name = "all_types",
    .dimensions = 4,
    .space = Space::Cosine,
    .schema = schema
  };

  {
    auto cr = Collection::create(cfg, path);
    ASSERT_TRUE(cr.ok());
    Collection col = std::move(cr.value());

    Metadata meta{
      {"intField", int64_t(42)},
      {"dblField", 3.14},
      {"strField", std::string("hello")},
      {"boolField", true}
    };
    col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f}, meta);
    col.save(path);
  }

  auto loadResult = Collection::load(path);
  ASSERT_TRUE(loadResult.ok());
  Collection col = std::move(loadResult.value());

  auto metaResult = col.getMetadata("v1");
  ASSERT_TRUE(metaResult.ok());
  auto& m = metaResult.value();
  EXPECT_EQ(std::get<int64_t>(m.at("intField")), 42);
  EXPECT_DOUBLE_EQ(std::get<double>(m.at("dblField")), 3.14);
  EXPECT_EQ(std::get<std::string>(m.at("strField")), "hello");
  EXPECT_EQ(std::get<bool>(m.at("boolField")), true);
}

// ── Update with metadata ────────────────────────────────────

TEST_F(CollectionTest, UpdateWithMetadata) {
  CollectionConfig cfg{.name = "update_meta", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f}, {{"tag", std::string("old")}});

  auto s = col.update("v1", {5.0f, 6.0f, 7.0f, 8.0f}, {{"tag", std::string("new")}});
  EXPECT_TRUE(s.ok());

  auto meta = col.getMetadata("v1");
  ASSERT_TRUE(meta.ok());
  EXPECT_EQ(std::get<std::string>(meta.value().at("tag")), "new");
}

// ── Create persistent collection ────────────────────────────

// ── SearchBatch single-query sequential fallback ────────────

TEST_F(CollectionTest, SearchBatchSingleQuery) {
  CollectionConfig cfg{.name = "sb_single", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::mt19937 gen(42);
  for (int i = 0; i < 50; ++i) {
    col.insert(std::to_string(i), RandomVector(4, gen));
  }

  // Single query triggers sequential path (numQueries <= 1)
  std::vector<std::vector<float>> queries = {RandomVector(4, gen)};
  auto result = col.searchBatch(queries, 5);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0].size(), 5u);
}

// ── SearchBatch multi-query with one dimension mismatch ─────

TEST_F(CollectionTest, SearchBatchMultiDimMismatch) {
  CollectionConfig cfg{.name = "sb_multi_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::vector<float>> queries = {
    {1.0f, 2.0f, 3.0f, 4.0f},
    {1.0f, 2.0f}
  };
  auto result = col.searchBatch(queries, 5);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kDimensionMismatch);
}

// ── Collection move assignment ──────────────────────────────

TEST_F(CollectionTest, CollectionMoveAssignment) {
  CollectionConfig cfg1{.name = "move_src", .dimensions = 4, .space = Space::Cosine};
  Collection a(cfg1);
  a.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  CollectionConfig cfg2{.name = "move_dst", .dimensions = 4, .space = Space::Cosine};
  Collection b(cfg2);

  b = std::move(a);
  EXPECT_EQ(b.size(), 1u);
  EXPECT_EQ(b.name(), "move_src");
}

// ── InsertBatch exceeds max batch size ──────────────────────

TEST_F(CollectionTest, InsertBatchExceedsMaxSize) {
  CollectionConfig cfg{.name = "batch_max", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  std::vector<Document> docs(1000001);
  for (size_t i = 0; i < docs.size(); ++i) {
    docs[i].embedding = {1.0f, 2.0f, 3.0f, 4.0f};
  }

  auto result = col.insertBatch(std::move(docs));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), utils::StatusCode::kInvalidArgument);
}

// ── InsertBatch per-doc bad metadata ────────────────────────

TEST_F(CollectionTest, InsertBatchBadMetadataPerDoc) {
  CollectionConfig cfg{.name = "batch_bad_meta", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  Metadata badMeta;
  for (int i = 0; i < 257; ++i) {
    badMeta["key_" + std::to_string(i)] = int64_t(i);
  }

  std::vector<Document> docs = {
    {"good", {1.0f, 2.0f, 3.0f, 4.0f}, {}},
    {"bad", {5.0f, 6.0f, 7.0f, 8.0f}, badMeta},
  };

  auto result = col.insertBatch(std::move(docs));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().successCount, 1u);
  EXPECT_EQ(result.value().failureCount, 1u);
}

// ── Update not found ────────────────────────────────────────

TEST_F(CollectionTest, UpdateNotFound) {
  CollectionConfig cfg{.name = "update_nf", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  auto s = col.update("nonexistent", {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kNotFound);
}

// ── Update dimension mismatch ───────────────────────────────

TEST_F(CollectionTest, UpdateDimensionMismatch) {
  CollectionConfig cfg{.name = "update_dim", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  auto s = col.update("v1", {1.0f, 2.0f});
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kDimensionMismatch);
}

// ── Update with NaN vector ──────────────────────────────────

TEST_F(CollectionTest, UpdateNaNVectorCoverage) {
  CollectionConfig cfg{.name = "update_nan", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  auto s = col.update("v1", {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f});
  EXPECT_FALSE(s.ok());
}

// ── Remove not found ────────────────────────────────────────

TEST_F(CollectionTest, RemoveNotFound) {
  CollectionConfig cfg{.name = "remove_nf", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});

  auto s = col.remove("nonexistent");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), utils::StatusCode::kNotFound);
}

// ── PreparedFilter move semantics ───────────────────────────

TEST_F(CollectionTest, PreparedFilterMoveSemantics) {
  CollectionConfig cfg{.name = "pf_move", .dimensions = 4, .space = Space::Cosine};
  Collection col(cfg);

  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  col.setMetadata("v1", {{"x", int64_t(1)}});

  auto pf1 = col.prepareFilter(MetadataFilter::Eq("x", int64_t(1)));

  // Move construct
  PreparedFilter pf2(std::move(pf1));
  auto results = col.search({1.0f, 2.0f, 3.0f, 4.0f}, 5, pf2);
  EXPECT_EQ(results.size(), 1u);

  // Move assign
  auto pf3 = col.prepareFilter(MetadataFilter::Eq("x", int64_t(99)));
  pf3 = std::move(pf2);
  results = col.search({1.0f, 2.0f, 3.0f, 4.0f}, 5, pf3);
  EXPECT_EQ(results.size(), 1u);
}

// ── Create persistent collection ────────────────────────────

TEST_F(CollectionTest, CreatePersistentCollection) {
  std::string path = GetTestPath("persistent");

  CollectionConfig cfg{.name = "persistent", .dimensions = 4, .space = Space::Cosine};
  auto result = Collection::create(cfg, path);
  ASSERT_TRUE(result.ok());

  Collection col = std::move(result.value());
  EXPECT_EQ(col.name(), "persistent");
  EXPECT_EQ(col.dimension(), 4u);

  // Insert and save to verify persistence works
  col.insert("v1", {1.0f, 2.0f, 3.0f, 4.0f});
  auto saveStatus = col.save(path);
  EXPECT_TRUE(saveStatus.ok());
}
