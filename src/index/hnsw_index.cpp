// Copyright 2025 ArrowDB
#include "internal/hnsw_index.h"
#include "arrow/utils/status.h"

#include "index/hnsw/hnsw.cpp"
#include "index/hnsw/space_ip.h"
#include "index/hnsw/space_l2.h"
#include "index/hnsw/scalar_quantizer.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace arrow {

HNSWIndex::HNSWIndex(size_t dim, Space space,
                      const HNSWConfig& config)
    : dim_(dim), spaceKind_(space) {
  // Create space based on space kind
    switch (space) {
    case Space::Cosine:
    case Space::InnerProduct:
      space_ = std::make_unique<hnsw::InnerProductSpace>(dim);
      break;
    case Space::L2:
      space_ = std::make_unique<hnsw::L2Space>(dim);
      break;
    default:
      throw std::invalid_argument("Unsupported space type");
  }

  auto sqDistType = (space == Space::L2)
      ? hnsw::SQDistType::L2 : hnsw::SQDistType::IP;

  hnsw_ = std::make_unique<hnsw::HierarchicalNSW<float>>(
      space_.get(),
      config.maxElements,
      config.M,
      config.efConstruction,
      42,           // randSeed
      false,        // allowReuseDeleted
      config.quantize,
      sqDistType);
}

HNSWIndex::~HNSWIndex() = default;

HNSWIndex::HNSWIndex(HNSWIndex&&) noexcept = default;
HNSWIndex& HNSWIndex::operator=(HNSWIndex&&) noexcept = default;

bool HNSWIndex::insert(InternalID id, const std::vector<float>& vec) {
  if (vec.size() != dim_) {
    std::cerr << "Vector dimension mismatch: expected " << dim_
              << ", got " << vec.size() << "\n";
    return false;
  }
  // Auto-grow if at capacity
  if (hnsw_->size() >= capacity()) {
    size_t newMax = capacity() * 2;
    hnsw_->resizeIndex(newMax);
  }
  hnsw_->addPoint(vec.data(), static_cast<hnsw::label_t>(id));
  return true;
}

std::vector<HNSWSearchResult> HNSWIndex::search(
    const std::vector<float>& query,
    size_t k,
    size_t ef) const {
  if (query.size() != dim_) {
    std::cerr << "Query dimension mismatch: expected " << dim_
              << ", got " << query.size() << "\n";
    return {};
  }

  using QueueItem = std::pair<float, hnsw::label_t>;  // (distance, label)
  std::priority_queue<QueueItem> resultsQueue =
      hnsw_->searchKnn(query.data(), k, nullptr, ef);

  std::vector<HNSWSearchResult> results;
  results.reserve(resultsQueue.size());

  // For InnerProduct/Cosine, hnswlib returns negative similarity (1 - cosine)
  // For L2, hnswlib returns positive distance
  // We negate for InnerProduct/Cosine to get proper similarity scores (higher = better)
  int8_t distToScoreConverter = (spaceKind_ == Space::L2) ? 1 : -1;

  // Results come out in worst-to-best order (max heap), reverse them
  while (!resultsQueue.empty()) {
    auto [dist, label] = resultsQueue.top();
    resultsQueue.pop();
    float score = distToScoreConverter * dist;
    results.push_back({static_cast<InternalID>(label), score});
  }
  std::reverse(results.begin(), results.end());
  return results;
}

namespace {

/// Adapts HNSWIndex::IDFilter to hnsw::BaseFilterFunctor so the HNSW
/// internals never leak past the HNSWIndex boundary.
class IDFilterAdapter final : public hnsw::BaseFilterFunctor {
public:
  explicit IDFilterAdapter(const HNSWIndex::IDFilter& filter) : filter_(filter) {}
  bool operator()(hnsw::label_t id) override {
    return filter_(static_cast<InternalID>(id));
  }
private:
  const HNSWIndex::IDFilter& filter_;
};

}  // namespace

std::vector<HNSWSearchResult> HNSWIndex::search(
    const std::vector<float>& query,
    size_t k,
    const IDFilter& filter,
    size_t ef) const {
  if (query.size() != dim_) {
    std::cerr << "Query dimension mismatch: expected " << dim_
              << ", got " << query.size() << "\n";
    return {};
  }

  IDFilterAdapter adapter(filter);
  using QueueItem = std::pair<float, hnsw::label_t>;
  std::priority_queue<QueueItem> resultsQueue =
      hnsw_->searchKnn(query.data(), k, &adapter, ef);

  std::vector<HNSWSearchResult> results;
  results.reserve(resultsQueue.size());

  int8_t distToScoreConverter = (spaceKind_ == Space::L2) ? 1 : -1;

  while (!resultsQueue.empty()) {
    auto [dist, label] = resultsQueue.top();
    resultsQueue.pop();
    float score = distToScoreConverter * dist;
    results.push_back({static_cast<InternalID>(label), score});
  }
  std::reverse(results.begin(), results.end());
  return results;
}

utils::Status HNSWIndex::saveIndex(const std::string& path) const {
  try {
    hnsw_->saveIndex(path);
    return utils::OkStatus();
  } catch (const std::exception& e) {
    return utils::Status(utils::StatusCode::kIoError,
      std::string("saveIndex failed: ") + e.what());
  }
}

utils::Status HNSWIndex::loadIndex(const std::string& path) {
  try {
    hnsw_->loadIndex(path, space_.get());
    // Set SQ kernels if quantization was loaded from the index file
    if (hnsw_->isQuantizationEnabled()) {
        auto sqDistType = (spaceKind_ == Space::L2)
            ? hnsw::SQDistType::L2 : hnsw::SQDistType::IP;
        hnsw_->setSQDistType(sqDistType);
        // For global SQ, also select integer-domain kernels
        if (hnsw_->isGlobalSQ()) {
            hnsw_->selectSQIntKernels();
        }
    }
    return utils::OkStatus();
  } catch (const std::exception& e) {
    return utils::Status(utils::StatusCode::kCorruption,
      std::string("loadIndex failed: ") + e.what());
  }
}

size_t HNSWIndex::size() const {
    return hnsw_->size();
}

void HNSWIndex::reserve(size_t max_elements) {
    hnsw_->resizeIndex(max_elements);
}

utils::Status HNSWIndex::markDelete(InternalID id) {
    try {
      hnsw_->markDelete(static_cast<hnsw::label_t>(id));
    } catch (const std::exception& e) {
      if (std::string_view(e.what()) == "Label not found") {
        return utils::Status(utils::StatusCode::kNotFound, e.what());
      }
      return utils::Status(utils::StatusCode::kInternal, e.what());
    }
    return utils::OkStatus();
}

const float* HNSWIndex::getVectorData(InternalID id) const {
    try {
      return hnsw_->getDataByLabel(static_cast<hnsw::label_t>(id));
    } catch (...) {
      return nullptr;
    }
}

size_t HNSWIndex::capacity() const {
    return hnsw_->getMaxElements();
}

void HNSWIndex::reorderBFS() {
    hnsw_->reorderBFS();
}

void HNSWIndex::computeGlobalSQ() {
    hnsw_->computeGlobalSQ();
}

bool HNSWIndex::isGlobalSQ() const {
    return hnsw_->isGlobalSQ();
}

}  // namespace arrow
