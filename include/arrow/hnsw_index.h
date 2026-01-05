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
/// 
/// Default values optimized for 100K+ vectors based on benchmark results:
/// - M=64: Provides 91-92% recall@10 for 100K vectors (vs 74-78% with M=32)
/// - efConstruction=200: Minimal impact on recall (200 vs 400 vs 800), balanced build time
/// 
/// For smaller datasets (<10K), M=32 may be sufficient and uses less memory.
/// For very large datasets (1M+), consider M=64 with efConstruction=400.
struct HNSWConfig {
    size_t maxElements = 1000000;  	// Initial capacity
    size_t M = 64;                  // Max connections per node (optimized for 100K+ vectors)
    size_t efConstruction = 200;   	// Construction beam width
};

/// HNSW index wrapper around hnswlib.
/// Owns vector data internally - no separate VectorStore needed.
class HNSWIndex {
private:
    size_t dim_;
    DistanceMetric metric_;
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
    bool insert(VectorID id, const std::vector<float>& vec);
    
    /// Search for k nearest neighbors.
    /// @param ef Search beam width (higher = better recall, slower)
    ///            Default 200 provides ~91% recall@10 for 100K vectors with M=64
    std::vector<SearchResult> search(
        const std::vector<float>& query, 
        size_t k, 
        size_t ef = 200  // Optimized for 100K+ vectors (benchmark-optimized)
    ) const;
    
    /// Vector dimension.
    inline size_t dimension() const { return dim_; }

	/// Save the index to disk.
	/// @param path File path where the index will be saved
	void saveIndex(const std::string& path) const;
	
	/// Load an index from disk.
	/// @param path File path to load the index from
	/// @note This replaces the current index with the loaded one.
	///       The dimension and metric must match the saved index.
	void loadIndex(const std::string& path);

    // Number of vectors in the index.
    size_t size() const; 

    /// Resize index capacity.
    void reserve(size_t max_elements);
};

}  // namespace arrow

#endif  // HNSW_INDEX_H
