#ifndef FLAT_SEARCH_H
#define FLAT_SEARCH_H

#include "vector_store.h"
#include <algorithm>
#include <vector>
#include <queue>
#include <utility>

namespace arrow {

struct SearchResult {
    uint64_t id;
    float score;
};

inline float dotProduct(
    const float* a,
    const float* b,
    size_t dim
) {
    float res = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

inline std::vector<SearchResult> flatSearch(
    const VectorStore& store,
    const std::vector<float>& query,
    size_t k
) {
    using Item = std::pair<float, size_t>; // (score, index)
    std::priority_queue<Item, std::vector<Item>, std::greater<>> minHeap;

    for (size_t i = 0; i < store.size(); ++i) {
        float score = dotProduct(
            store.vectorAt(i),
            query.data(),
            store.dimension()
        );

        if (minHeap.size() < k) {
            minHeap.emplace(score, i);
        } else if (score > minHeap.top().first) {
            minHeap.pop();
            minHeap.emplace(score, i);
        }
    }

    std::vector<SearchResult> results;
    while (!minHeap.empty()) {
        auto [score, idx] = minHeap.top();
        minHeap.pop();
        results.push_back({store.id_at(idx), score});
    }

    std::reverse(results.begin(), results.end());
    return results;
}

} // namespace arrow

#endif // FLAT_SEARCH_H

