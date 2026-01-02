#include "arrow/hnsw_index.h"
#include "arrow/collection.h"  // For DistanceMetric enum

#include <hnswlib/hnswlib.h>
#include <stdexcept>
#include <algorithm>

namespace arrow {

HNSWIndex::HNSWIndex(size_t dim, DistanceMetric metric, const HNSWConfig& config)
    : dim_(dim), metric_(metric) {
    
    // Create space based on metric
    switch (metric) {
        case DistanceMetric::Cosine:
        case DistanceMetric::InnerProduct:
            space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
            break;
        case DistanceMetric::L2:
            space_ = std::make_unique<hnswlib::L2Space>(dim);
            break;
        default:
            throw std::invalid_argument("Unsupported distance metric");
    }
    
    hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(),
        config.maxElements,
        config.M,
        config.efConstruction
    );
}

HNSWIndex::~HNSWIndex() = default;

HNSWIndex::HNSWIndex(HNSWIndex&&) noexcept = default;
HNSWIndex& HNSWIndex::operator=(HNSWIndex&&) noexcept = default;

void HNSWIndex::insert(VectorID id, const std::vector<float>& vec) {
    if (vec.size() != dim_) {
        throw std::invalid_argument("Dimension mismatch");
    }
    hnsw_->addPoint(vec.data(), static_cast<hnswlib::labeltype>(id));
}

std::vector<SearchResult> HNSWIndex::search(
    const std::vector<float>& query,
    size_t k,
    size_t ef
) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("Query dimension mismatch");
    }
    
    hnsw_->setEf(ef);
    
    using QueueItem = std::pair<float, hnswlib::labeltype>;  // (distance, label)
    std::priority_queue<QueueItem> resultsQueue = hnsw_->searchKnn(query.data(), k);
    
    std::vector<SearchResult> results;
    results.reserve(resultsQueue.size());

		int8_t distToScoreConverter = (metric_ == DistanceMetric::L2) ? 1 : -1;
    
    // Results come out in worst-to-best order, reverse them
    while (!resultsQueue.empty()) {
				// dist -> float, label -> hnswlib::labeltype
        auto [dist, label] = resultsQueue.top();
        resultsQueue.pop();
        // For InnerProduct/Cosine, hnswlib returns negative distance (smaller = better)
        // For L2, hnswlib returns positive distance (smaller = better)
        float score = distToScoreConverter * dist;
        results.push_back({static_cast<VectorID>(label), score});
    }
    std::reverse(results.begin(), results.end());
    return results;
}

void HNSWIndex::saveIndex(const std::string& path) const {
		hnsw_->saveIndex(path);
}

void HNSWIndex::loadIndex(const std::string& path) {
	hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
		space_.get(),
		path
	);
}

size_t HNSWIndex::size() const {
    return hnsw_->cur_element_count;
}

void HNSWIndex::reserve(size_t max_elements) {
    hnsw_->resizeIndex(max_elements);
}
}  // namespace arrow
