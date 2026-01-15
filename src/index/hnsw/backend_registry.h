#pragma once
#include <cstddef>
#include "impl_kernels.h"
#include "cpu_features.h"
#include "backend_interface.h"

namespace hnsw {

template<>
inline const DistanceBackend<float>& selectBackend<float>() {
    using namespace hnsw::impl;

    // Use C++11 magic statics for thread-safe initialization.
    // Static local initialization is guaranteed to be thread-safe.
    static const DistanceBackend<float> backend = []() {
        DistanceBackend<float> b;
        auto simd = CPUFeatures::get().level;

#if defined(__aarch64__) || defined(_M_ARM64)
        if (simd >= SimdLevel::NEON) {
            b.l2 = l2_neon;
            b.ip = ip_neon;
            b.l2_aligned16 = l2_neon_aligned16;
            b.ip_aligned16 = ip_neon_aligned16;
            b.l2_aligned4 = l2_neon_aligned4;
            b.ip_aligned4 = ip_neon_aligned4;
            b.name = "neon";
            b.simd_level = SimdLevel::NEON;
            return b;
        }
#endif

#if defined(__AVX512F__)
        if (simd >= SimdLevel::AVX512) {
            b.l2 = l2_avx512;
            b.ip = ip_avx512;
            b.l2_aligned16 = l2_avx512_aligned16;
            b.ip_aligned16 = ip_avx512_aligned16;
            b.l2_aligned4 = l2_avx2_aligned4;
            b.ip_aligned4 = ip_avx2_aligned4;
            b.name = "avx512";
            b.simd_level = SimdLevel::AVX512;
            return b;
        }
#endif

#if defined(__AVX2__) && defined(__FMA__)
        if (simd >= SimdLevel::AVX2) {
            b.l2 = l2_avx2;
            b.ip = ip_avx2;
            b.l2_aligned16 = l2_avx2_aligned16;
            b.ip_aligned16 = ip_avx2_aligned16;
            b.l2_aligned4 = l2_avx2_aligned4;
            b.ip_aligned4 = ip_avx2_aligned4;
            b.name = "avx2";
            b.simd_level = SimdLevel::AVX2;
            return b;
        }
#endif

#if defined(__AVX__)
        if (simd >= SimdLevel::AVX) {
            b.l2 = l2_avx;
            b.ip = ip_avx;
            b.l2_aligned16 = l2_avx_aligned16;
            b.ip_aligned16 = ip_avx_aligned16;
            b.l2_aligned4 = l2_avx_aligned4;
            b.ip_aligned4 = ip_avx_aligned4;
            b.name = "avx";
            b.simd_level = SimdLevel::AVX;
            return b;
        }
#endif

#if defined(__SSE__)
        if (simd >= SimdLevel::SSE) {
            b.l2 = l2_sse;
            b.ip = ip_sse;
            b.l2_aligned16 = l2_sse_aligned16;
            b.ip_aligned16 = ip_sse_aligned16;
            b.l2_aligned4 = l2_sse_aligned4;
            b.ip_aligned4 = ip_sse_aligned4;
            b.name = "sse";
            b.simd_level = SimdLevel::SSE;
            return b;
        }
#endif

        // scalar fallback
        b.l2 = l2_scalar;
        b.ip = ip_scalar;
        b.l2_aligned16 = l2_scalar;
        b.ip_aligned16 = ip_scalar;
        b.l2_aligned4 = l2_scalar;
        b.ip_aligned4 = ip_scalar;
        b.name = "scalar";
        b.simd_level = SimdLevel::NONE;
        return b;
    }();

    return backend;
}

} // namespace hnsw
