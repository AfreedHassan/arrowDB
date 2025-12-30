#ifndef HNSW_INDEX_H
#define HNSW_INDEX_H

#include <vector>
#include <memory>
#include "types.h"

// Forward declare hnswlib types to avoid header pollution
namespace hnswlib {
    template<typename T> class HierarchicalNSW;
    template<typename T> class SpaceInterface;
}

namespace arrow {

struct SearchResult {
    VectorID id;
    float score;
};

enum class DistanceMetric;

/// Configuration for HNSW index construction.
struct HNSWConfig {
    size_t maxElements = 1000000;  	// Initial capacity
    size_t M = 16;                  // Max connections per node
    size_t efConstruction = 200;   	// Construction beam width
};

/// HNSW index wrapper around hnswlib.
/// Owns vector data internally - no separate VectorStore needed.
class HNSWIndex {
private:
    size_t dim_;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
public:
    HNSWIndex(size_t dim, DistanceMetric metric, const HNSWConfig& config = {});
    ~HNSWIndex();
    
    // Non-copyable, movable
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;
    HNSWIndex(HNSWIndex&&) noexcept;
    HNSWIndex& operator=(HNSWIndex&&) noexcept;
    
    /// Insert a vector with the given ID.
    void insert(VectorID id, const std::vector<float>& vec);
    
    /// Search for k nearest neighbors.
    /// @param ef Search beam width (higher = better recall, slower)
    std::vector<SearchResult> search(
        const std::vector<float>& query, 
        size_t k, 
        size_t ef = 100
    ) const;
    
    /// Vector dimension.
    inline size_t dimension() const { return dim_; }

    // Number of vectors in the index.
    size_t size() const; 

    /// Resize index capacity.
    void reserve(size_t max_elements);
};

}  // namespace arrow

#endif  // HNSW_INDEX_H
