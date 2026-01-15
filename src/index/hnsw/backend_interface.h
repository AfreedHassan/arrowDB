#pragma once
#include <cstddef>
#include "cpu_features.h"

namespace hnsw {

template<typename metric_t>
using pdistfunc_t = float (*)(const metric_t* a, const metric_t* b, std::size_t dim);

template<typename T>
struct DistanceBackend {
    pdistfunc_t<T> l2;
    pdistfunc_t<T> ip;
    pdistfunc_t<T> l2_aligned16;  // Optimized for dim % 16 == 0
    pdistfunc_t<T> ip_aligned16;  // Optimized for dim % 16 == 0
    pdistfunc_t<T> l2_aligned4;   // Optimized for dim % 4 == 0
    pdistfunc_t<T> ip_aligned4;   // Optimized for dim % 4 == 0
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
