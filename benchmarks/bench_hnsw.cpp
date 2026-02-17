// Copyright 2025 ArrowDB
#include <benchmark/benchmark.h>
#include "arrow/client.h"
#include "arrow/collection.h"

#include <memory>
#include <random>
#include <vector>

namespace {

std::vector<float> randomVector(size_t dim, std::mt19937& gen) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> vec(dim);
  float norm = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    vec[i] = dist(gen);
    norm += vec[i] * vec[i];
  }
  norm = std::sqrt(norm);
  for (auto& v : vec) v /= norm;
  return vec;
}

// ─────────────────────────────────────────────────────────────
// Shared client + collection, persisted to /tmp for reuse
// ─────────────────────────────────────────────────────────────

static constexpr size_t kDim = 384;
static constexpr size_t kN = 100000;
static const std::string kDataDir = "/tmp/arrow_bench_db";

struct SharedState {
  std::unique_ptr<arrow::Client> client;
  arrow::Collection* collection = nullptr;  // owned by client
  std::vector<std::vector<float>> queries;

  static SharedState& instance() {
    static SharedState inst;
    return inst;
  }

  void ensureReady() {
    if (collection) return;

    client = std::make_unique<arrow::Client>(kDataDir);

    arrow::CollectionConfig config;
    config.name = "bench_384d_100k";
    config.dimensions = kDim;
    config.space = arrow::Space::L2;
    config.index_config.max_elements = kN;
    config.index_config.quantization = arrow::Quantization::INT8;
    config.index_config.hnsw_params.M = 16;
    config.index_config.hnsw_params.ef_construction = 200;

    auto result = client->getOrCreateCollection(config.name, config);
    if (!result.ok()) {
      fprintf(stderr, "Failed to get/create collection: %s\n",
              result.status().message().c_str());
      return;
    }
    collection = result.value();

    if (collection->size() >= kN) {
      fprintf(stderr, "Loaded existing collection: %zu vectors\n",
              collection->size());
    } else {
      fprintf(stderr, "Building collection: %zuD x %zu vectors...\n", kDim, kN);
      std::mt19937 gen(42);
      for (size_t i = collection->size(); i < kN; ++i) {
        auto vec = randomVector(kDim, gen);
        collection->insert(std::to_string(i), vec);
        if (i % 10000 == 0 && i > 0)
          fprintf(stderr, "  inserted %zu/%zu\n", i, kN);
      }
      // Save so next run loads instantly
      client->close();
      // Reopen
      client = std::make_unique<arrow::Client>(kDataDir);
      auto r2 = client->getCollection(config.name);
      collection = r2.value();
      fprintf(stderr, "Collection built and saved: %zu vectors\n",
              collection->size());
    }

    // Generate queries
    std::mt19937 qgen(123);
    queries.reserve(1000);
    for (int i = 0; i < 1000; ++i)
      queries.push_back(randomVector(kDim, qgen));
  }
};

// ─────────────────────────────────────────────────────────────
// Multi-threaded throughput through Client → Collection::search()
// ─────────────────────────────────────────────────────────────

static void BM_SearchThroughput(benchmark::State& state) {
  auto& shared = SharedState::instance();
  if (state.thread_index() == 0) {
    shared.ensureReady();
  }

  size_t idx = state.thread_index() * 137;
  const size_t nQueries = shared.queries.size();

  for (auto _ : state) {
    auto results = shared.collection->search(
        shared.queries[idx % nQueries], 10, 118);
    benchmark::DoNotOptimize(results);
    ++idx;
  }

  state.SetItemsProcessed(state.iterations());
  state.SetLabel("Collection::search SQ8 ef=118 " +
                 std::to_string(state.threads()) + "T");
}

BENCHMARK(BM_SearchThroughput)
->Unit(benchmark::kMicrosecond)
->Threads(1)
->Threads(4)
->Threads(12)
->UseRealTime()
->MeasureProcessCPUTime();

}  // namespace

BENCHMARK_MAIN();
