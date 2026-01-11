// Copyright 2025 ArrowDB
#include <benchmark/benchmark.h>
#include "internal/hnsw_index.h"
#include "arrow/collection.h"

#include <memory>
#include <random>
#include <vector>

namespace {

// Generate a random normalized vector
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

// Fixture for HNSW benchmarks with pre-built index
class HNSWBenchmark : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    dim_ = 128;  // Common embedding dimension
    size_t n = state.range(0);

    std::mt19937 gen(42);

    index_ = std::make_unique<arrow::HNSWIndex>(
        dim_,
        arrow::DistanceMetric::Cosine,
        arrow::HNSWConfig{n, 16, 200});

    // Build index
    for (size_t i = 0; i < n; ++i) {
      index_->insert(i, randomVector(dim_, gen));
    }

    // Pre-generate queries
    queries_.clear();
    for (int i = 0; i < 100; ++i) {
      queries_.push_back(randomVector(dim_, gen));
    }
  }

  void TearDown(const benchmark::State&) override {
    index_.reset();
    queries_.clear();
  }

 protected:
  size_t dim_;
  std::unique_ptr<arrow::HNSWIndex> index_;
  std::vector<std::vector<float>> queries_;
};

// ─────────────────────────────────────────────────────────────
// Search benchmarks at various ef values
// ─────────────────────────────────────────────────────────────

BENCHMARK_DEFINE_F(HNSWBenchmark, SearchEf10)(benchmark::State& state) {
  size_t queryIdx = 0;
  for (auto _ : state) {
    auto results = index_->search(queries_[queryIdx % queries_.size()], 10, 10);
    benchmark::DoNotOptimize(results);
    ++queryIdx;
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(HNSWBenchmark, SearchEf50)(benchmark::State& state) {
  size_t query_idx = 0;
  for (auto _ : state) {
    auto results = index_->search(queries_[query_idx % queries_.size()], 10, 50);
    benchmark::DoNotOptimize(results);
    ++query_idx;
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(HNSWBenchmark, SearchEf100)(benchmark::State& state) {
  size_t query_idx = 0;
  for (auto _ : state) {
    auto results = index_->search(queries_[query_idx % queries_.size()], 10, 100);
    benchmark::DoNotOptimize(results);
    ++query_idx;
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(HNSWBenchmark, SearchEf200)(benchmark::State& state) {
  size_t query_idx = 0;
  for (auto _ : state) {
    auto results = index_->search(queries_[query_idx % queries_.size()], 10, 200);
    benchmark::DoNotOptimize(results);
    ++query_idx;
  }
  state.SetItemsProcessed(state.iterations());
}

// Register benchmarks with different dataset sizes
// Args: number of vectors in index
BENCHMARK_REGISTER_F(HNSWBenchmark, SearchEf10)
->Arg(1000)
->Arg(10000)
->Arg(100000)
->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(HNSWBenchmark, SearchEf50)
->Arg(1000)
->Arg(10000)
->Arg(100000)
->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(HNSWBenchmark, SearchEf100)
->Arg(1000)
->Arg(10000)
->Arg(100000)
->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(HNSWBenchmark, SearchEf200)
->Arg(1000)
->Arg(10000)
->Arg(100000)
->Unit(benchmark::kMicrosecond);

// ─────────────────────────────────────────────────────────────
// Insert benchmark
// ─────────────────────────────────────────────────────────────

static void BM_HNSWInsert(benchmark::State& state) {
  const size_t dim = 128;
  const size_t batch_size = state.range(0);

  std::mt19937 gen(42);

  // Pre-generate vectors
  std::vector<std::vector<float>> vectors;
  vectors.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    vectors.push_back(randomVector(dim, gen));
  }

  for (auto _ : state) {
    state.PauseTiming();
    arrow::HNSWIndex index(dim, arrow::DistanceMetric::Cosine,
                           {batch_size, 16, 200});
    state.ResumeTiming();

    for (size_t i = 0; i < batch_size; ++i) {
      index.insert(i, vectors[i]);
    }
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK(BM_HNSWInsert)
->Arg(1000)
->Arg(10000)
->Arg(50000)
->Unit(benchmark::kMillisecond);

// ─────────────────────────────────────────────────────────────
// Different dimensions benchmark
// ─────────────────────────────────────────────────────────────

static void BM_HNSWSearchDimensions(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t n = 10000;

  std::mt19937 gen(42);

  arrow::HNSWIndex index(dim, arrow::DistanceMetric::Cosine, {n, 16, 200});

  // Build index
  for (size_t i = 0; i < n; ++i) {
    index.insert(i, randomVector(dim, gen));
  }

  // Pre-generate queries
  std::vector<std::vector<float>> queries;
  for (int i = 0; i < 100; ++i) {
    queries.push_back(randomVector(dim, gen));
  }

  size_t query_idx = 0;
  for (auto _ : state) {
    auto results = index.search(queries[query_idx % queries.size()], 10, 100);
    benchmark::DoNotOptimize(results);
    ++query_idx;
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_HNSWSearchDimensions)
->Arg(64)    // Small embeddings
->Arg(128)   // Common
->Arg(384)   // sentence-transformers
->Arg(768)   // BERT/OpenAI
->Arg(1536)  // text-embedding-ada-002
->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();

