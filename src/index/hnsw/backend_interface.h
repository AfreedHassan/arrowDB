#pragma once
#include <cstddef>
#include "cpu_features.h"

namespace hnsw {

/// Single-pair distance function pointer type.
/// Computes distance between two vectors of given dimension.
template<typename metric_t>
using pdistfunc_t = float (*)(const metric_t* a, const metric_t* b, std::size_t dim);

/// Batch distance function pointer type.
/// Computes distances from one query to N target vectors.
/// @param query     Query vector
/// @param targets   Array of pointers to target vectors
/// @param numTargets Number of target vectors
/// @param dim       Dimension of vectors
/// @param outDistances Output array for computed distances (must have space for numTargets)
template<typename metric_t>
using pbatchdistfunc_t = void (*)(
    const metric_t* query,
    const metric_t* const* targets,
    std::size_t numTargets,
    std::size_t dim,
    metric_t* outDistances);

template<typename T>
struct DistanceBackend {
    // Single-pair distance functions
    pdistfunc_t<T> l2;
    pdistfunc_t<T> ip;
    pdistfunc_t<T> l2_aligned16;  // Optimized for dim % 16 == 0
    pdistfunc_t<T> ip_aligned16;  // Optimized for dim % 16 == 0
    pdistfunc_t<T> l2_aligned4;   // Optimized for dim % 4 == 0
    pdistfunc_t<T> ip_aligned4;   // Optimized for dim % 4 == 0

    // Batch distance functions (1-to-N)
    pbatchdistfunc_t<T> l2_batch;
    pbatchdistfunc_t<T> ip_batch;

    const char* name;
    SimdLevel simd_level;
};

// Primary template - produces compile error for unsupported types.
// Only float is specialized (in backend_registry.h).
template<typename T>
const DistanceBackend<T>& selectBackend() {
    static_assert(sizeof(T) == 0,
        "selectBackend<T> only supports float type. "
        "Other types (double, int, etc.) are not implemented.");
}

} // namespace hnsw
