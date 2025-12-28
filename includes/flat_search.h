#ifndef FLAT_SEARCH_H
#define FLAT_SEARCH_H

#include "vector_store.h"
#include <algorithm>
#include <vector>
#include <queue>
#include <utility>

namespace arrow {

/**
 * @brief Result of a vector search operation.
 */
struct SearchResult {
    uint64_t id;    ///< The unique identifier of the matched vector.
    float score;    ///< The similarity score (dot product) of the match.
};

/**
 * @brief Computes the dot product of two vectors.
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param dim The dimension of both vectors.
 * @return The dot product (sum of element-wise products).
 */
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

/**
 * @brief Performs a flat (brute-force) search for top-k nearest neighbors.
 * 
 * This function searches through all vectors in the store using dot product
 * similarity. For normalized vectors, dot product is equivalent to cosine similarity.
 * Results are returned in descending order of similarity score.
 * 
 * @param store The VectorStore containing the vectors to search.
 * @param query The query vector. Should be normalized for cosine similarity.
 * @param k The number of nearest neighbors to return.
 * @return A vector of SearchResult containing the top-k nearest neighbors,
 *         sorted by score in descending order.
 * @note Time complexity is O(n*d + k*log(k)) where n is the number of vectors
 *       and d is the dimension.
 */
inline std::vector<SearchResult> flatSearch(
    const VectorStore& store,
    const std::vector<float>& query,
    size_t k
) {
    using Item = std::pair<float, size_t>; // (score, index)
    std::priority_queue<Item, std::vector<Item>, std::greater<>> minHeap;

    for (size_t i = 0; i < store.size(); ++i) {
        float score = dotProduct(
            store.vecAt(i),
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
        results.push_back({store.vecIdAt(idx), score});
    }

    std::reverse(results.begin(), results.end());
    return results;
}

} // namespace arrow

#endif // FLAT_SEARCH_H

