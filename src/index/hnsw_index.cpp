// Copyright 2025 ArrowDB
#include "internal/hnsw_index.h"
#include "arrow/utils/status.h"

#include "index/hnsw/hnsw.cpp"
#include "index/hnsw/space_ip.h"
#include "index/hnsw/space_l2.h"
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

  hnsw_ = std::make_unique<hnsw::HierarchicalNSW<float>>(
      space_.get(),
      config.maxElements,
      config.M,
      config.efConstruction);
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
  hnsw_->addPoint(vec.data(), static_cast<hnsw::label_t>(id));
  return true;
}

std::vector<IndexSearchResult> HNSWIndex::search(
    const std::vector<float>& query,
    size_t k,
    size_t ef) const {
  if (query.size() != dim_) {
    throw std::invalid_argument("Query dimension mismatch");
  }

  hnsw_->setEF(ef);

  using QueueItem = std::pair<float, hnsw::label_t>;  // (distance, label)
  std::priority_queue<QueueItem> resultsQueue =
      hnsw_->searchKnn(query.data(), k);

  std::vector<IndexSearchResult> results;
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

void HNSWIndex::saveIndex(const std::string& path) const {
  hnsw_->saveIndex(path);
}

void HNSWIndex::loadIndex(const std::string& path) {
  hnsw_->loadIndex(path, space_.get());
}

size_t HNSWIndex::size() const {
    return hnsw_->size();
}

void HNSWIndex::reserve(size_t max_elements) {
    hnsw_->resizeIndex(max_elements);
}

utils::Status HNSWIndex::markDelete(InternalID id) {
    const std::string_view labelNotFoundError = "Label not found";
    try {
      hnsw_->markDelete(static_cast<hnsw::label_t>(id));
    } catch (const std::exception& e) {
      if (e.what() == labelNotFoundError) {
        return utils::Status(utils::StatusCode::kNotFound, e.what());
      }
    }
    return utils::OkStatus();
}
}  // namespace arrow
