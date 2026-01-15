#pragma once

#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#else
#define PORTABLE_ALIGN32
#define PORTABLE_ALIGN64
#endif

namespace hnsw::impl {

// Horizontal sum helpers
#if defined(__SSE3__)
inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#elif defined(__SSE__)
inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

#if defined(__AVX__)
inline float hsum_avx(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    return hsum_sse(sum128);
}
#endif

#if defined(__AVX512F__)
inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

// Scalar fallback
inline float l2_scalar(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline float ip_scalar(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i)
        sum += a[i] * b[i];
    return sum;
}

// SSE (128-bit)
#if defined(__SSE__)
inline float l2_sse(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    std::size_t i = 0;

    // Process 16 elements per iteration (4 SIMD ops unrolled)
    for (; i + 16 <= dim; i += 16) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(diff, diff));

        va = _mm_loadu_ps(a + i + 4);
        vb = _mm_loadu_ps(b + i + 4);
        diff = _mm_sub_ps(va, vb);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(diff, diff));

        va = _mm_loadu_ps(a + i + 8);
        vb = _mm_loadu_ps(b + i + 8);
        diff = _mm_sub_ps(va, vb);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(diff, diff));

        va = _mm_loadu_ps(a + i + 12);
        vb = _mm_loadu_ps(b + i + 12);
        diff = _mm_sub_ps(va, vb);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(diff, diff));
    }

    __m128 sum = _mm_add_ps(sum1, sum2);
    float result = hsum_sse(sum);

    // Handle remaining elements
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

inline float ip_sse(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    std::size_t i = 0;

    // Process 16 elements per iteration (4 SIMD ops unrolled)
    for (; i + 16 <= dim; i += 16) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(va, vb));

        va = _mm_loadu_ps(a + i + 4);
        vb = _mm_loadu_ps(b + i + 4);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(va, vb));

        va = _mm_loadu_ps(a + i + 8);
        vb = _mm_loadu_ps(b + i + 8);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(va, vb));

        va = _mm_loadu_ps(a + i + 12);
        vb = _mm_loadu_ps(b + i + 12);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(va, vb));
    }

    __m128 sum = _mm_add_ps(sum1, sum2);
    float result = hsum_sse(sum);

    // Handle remaining elements
    for (; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}
#endif

// AVX (256-bit)
#if defined(__AVX__)
inline float l2_avx(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    std::size_t i = 0;

    // Process 32 elements per iteration (4 SIMD ops unrolled)
    for (; i + 32 <= dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff, diff));

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(diff, diff));

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        diff = _mm256_sub_ps(va, vb);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff, diff));

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(diff, diff));
    }

    __m256 sum = _mm256_add_ps(sum1, sum2);
    float result = hsum_avx(sum);

    // Handle remaining elements
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

inline float ip_avx(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    std::size_t i = 0;

    // Process 32 elements per iteration (4 SIMD ops unrolled)
    for (; i + 32 <= dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(va, vb));

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(va, vb));

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(va, vb));

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(va, vb));
    }

    __m256 sum = _mm256_add_ps(sum1, sum2);
    float result = hsum_avx(sum);

    // Handle remaining elements
    for (; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}
#endif

// AVX2 (256-bit with FMA)
#if defined(__AVX2__) && defined(__FMA__)
inline float l2_avx2(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism.
    // FMA (fused multiply-add) improves throughput significantly.
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    std::size_t i = 0;

    // Process 32 elements per iteration (4 SIMD ops unrolled)
    for (; i + 32 <= dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        // FMA computes: diff*diff + sum1 in single operation
        sum1 = _mm256_fmadd_ps(diff, diff, sum1);

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_fmadd_ps(diff, diff, sum2);

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        diff = _mm256_sub_ps(va, vb);
        sum1 = _mm256_fmadd_ps(diff, diff, sum1);

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_fmadd_ps(diff, diff, sum2);
    }

    __m256 sum = _mm256_add_ps(sum1, sum2);
    float result = hsum_avx(sum);

    // Handle remaining elements
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

inline float ip_avx2(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism.
    // FMA (fused multiply-add) improves throughput significantly.
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    std::size_t i = 0;

    // Process 32 elements per iteration (4 SIMD ops unrolled)
    for (; i + 32 <= dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        // FMA computes: va*vb + sum1 in single operation
        sum1 = _mm256_fmadd_ps(va, vb, sum1);

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        sum2 = _mm256_fmadd_ps(va, vb, sum2);

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        sum1 = _mm256_fmadd_ps(va, vb, sum1);

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        sum2 = _mm256_fmadd_ps(va, vb, sum2);
    }

    __m256 sum = _mm256_add_ps(sum1, sum2);
    float result = hsum_avx(sum);

    // Handle remaining elements
    for (; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}
#endif

// AVX512 (512-bit)
#if defined(__AVX512F__)
inline float l2_avx512(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism.
    // Process 64 elements per iteration for better throughput.
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    std::size_t i = 0;

    // Process 64 elements per iteration (4 SIMD ops unrolled)
    for (; i + 64 <= dim; i += 64) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum1 = _mm512_fmadd_ps(diff, diff, sum1);

        va = _mm512_loadu_ps(a + i + 16);
        vb = _mm512_loadu_ps(b + i + 16);
        diff = _mm512_sub_ps(va, vb);
        sum2 = _mm512_fmadd_ps(diff, diff, sum2);

        va = _mm512_loadu_ps(a + i + 32);
        vb = _mm512_loadu_ps(b + i + 32);
        diff = _mm512_sub_ps(va, vb);
        sum1 = _mm512_fmadd_ps(diff, diff, sum1);

        va = _mm512_loadu_ps(a + i + 48);
        vb = _mm512_loadu_ps(b + i + 48);
        diff = _mm512_sub_ps(va, vb);
        sum2 = _mm512_fmadd_ps(diff, diff, sum2);
    }

    __m512 sum = _mm512_add_ps(sum1, sum2);
    float result = hsum_avx512(sum);

    // Handle remaining elements
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

inline float ip_avx512(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism.
    // Process 64 elements per iteration for better throughput.
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    std::size_t i = 0;

    // Process 64 elements per iteration (4 SIMD ops unrolled)
    for (; i + 64 <= dim; i += 64) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum1 = _mm512_fmadd_ps(va, vb, sum1);

        va = _mm512_loadu_ps(a + i + 16);
        vb = _mm512_loadu_ps(b + i + 16);
        sum2 = _mm512_fmadd_ps(va, vb, sum2);

        va = _mm512_loadu_ps(a + i + 32);
        vb = _mm512_loadu_ps(b + i + 32);
        sum1 = _mm512_fmadd_ps(va, vb, sum1);

        va = _mm512_loadu_ps(a + i + 48);
        vb = _mm512_loadu_ps(b + i + 48);
        sum2 = _mm512_fmadd_ps(va, vb, sum2);
    }

    __m512 sum = _mm512_add_ps(sum1, sum2);
    float result = hsum_avx512(sum);

    // Handle remaining elements
    for (; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}
#endif

// NEON (ARM64 / Apple Silicon)
#if defined(__aarch64__) || defined(_M_ARM64)
inline float l2_neon(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism.
    // NEON is 128-bit (4 floats), so we process 16 floats per iteration.
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    // Process 16 elements per iteration (4 SIMD ops unrolled)
    for (; i + 16 <= dim; i += 16) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        // FMA: sum1 = sum1 + (diff * diff)
        sum1 = vfmaq_f32(sum1, diff, diff);

        va = vld1q_f32(a + i + 4);
        vb = vld1q_f32(b + i + 4);
        diff = vsubq_f32(va, vb);
        sum2 = vfmaq_f32(sum2, diff, diff);

        va = vld1q_f32(a + i + 8);
        vb = vld1q_f32(b + i + 8);
        diff = vsubq_f32(va, vb);
        sum1 = vfmaq_f32(sum1, diff, diff);

        va = vld1q_f32(a + i + 12);
        vb = vld1q_f32(b + i + 12);
        diff = vsubq_f32(va, vb);
        sum2 = vfmaq_f32(sum2, diff, diff);
    }

    float32x4_t sum = vaddq_f32(sum1, sum2);
    float result = vaddvq_f32(sum);

    // Handle remaining elements
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

inline float ip_neon(const float* a, const float* b, std::size_t dim) {
    // Use two accumulators for better instruction-level parallelism.
    // NEON is 128-bit (4 floats), so we process 16 floats per iteration.
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    // Process 16 elements per iteration (4 SIMD ops unrolled)
    for (; i + 16 <= dim; i += 16) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum1 = vfmaq_f32(sum1, va, vb);

        va = vld1q_f32(a + i + 4);
        vb = vld1q_f32(b + i + 4);
        sum2 = vfmaq_f32(sum2, va, vb);

        va = vld1q_f32(a + i + 8);
        vb = vld1q_f32(b + i + 8);
        sum1 = vfmaq_f32(sum1, va, vb);

        va = vld1q_f32(a + i + 12);
        vb = vld1q_f32(b + i + 12);
        sum2 = vfmaq_f32(sum2, va, vb);
    }

    float32x4_t sum = vaddq_f32(sum1, sum2);
    float result = vaddvq_f32(sum);

    // Handle remaining elements
    for (; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}
#endif

// ============================================================================
// Aligned-dimension optimized kernels (skip residual handling)
// ============================================================================

// SSE aligned kernels
#if defined(__SSE__)
inline float l2_sse_aligned16(const float* a, const float* b, std::size_t dim) {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    // Assume dim % 16 == 0, process all elements without residual loop
    for (std::size_t i = 0; i < dim; i += 16) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(diff, diff));

        va = _mm_loadu_ps(a + i + 4);
        vb = _mm_loadu_ps(b + i + 4);
        diff = _mm_sub_ps(va, vb);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(diff, diff));

        va = _mm_loadu_ps(a + i + 8);
        vb = _mm_loadu_ps(b + i + 8);
        diff = _mm_sub_ps(va, vb);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(diff, diff));

        va = _mm_loadu_ps(a + i + 12);
        vb = _mm_loadu_ps(b + i + 12);
        diff = _mm_sub_ps(va, vb);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(diff, diff));
    }
    __m128 sum = _mm_add_ps(sum1, sum2);
    return hsum_sse(sum);
}

inline float ip_sse_aligned16(const float* a, const float* b, std::size_t dim) {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 16) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(va, vb));

        va = _mm_loadu_ps(a + i + 4);
        vb = _mm_loadu_ps(b + i + 4);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(va, vb));

        va = _mm_loadu_ps(a + i + 8);
        vb = _mm_loadu_ps(b + i + 8);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(va, vb));

        va = _mm_loadu_ps(a + i + 12);
        vb = _mm_loadu_ps(b + i + 12);
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(va, vb));
    }
    __m128 sum = _mm_add_ps(sum1, sum2);
    return hsum_sse(sum);
}

inline float l2_sse_aligned4(const float* a, const float* b, std::size_t dim) {
    float result = 0.0f;
    for (std::size_t i = 0; i < dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 prod = _mm_mul_ps(diff, diff);
        result += hsum_sse(prod);
    }
    return result;
}

inline float ip_sse_aligned4(const float* a, const float* b, std::size_t dim) {
    float result = 0.0f;
    for (std::size_t i = 0; i < dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 prod = _mm_mul_ps(va, vb);
        result += hsum_sse(prod);
    }
    return result;
}
#endif

// AVX aligned kernels
#if defined(__AVX__)
inline float l2_avx_aligned16(const float* a, const float* b, std::size_t dim) {
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff, diff));

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(diff, diff));

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        diff = _mm256_sub_ps(va, vb);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff, diff));

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(diff, diff));
    }
    __m256 sum = _mm256_add_ps(sum1, sum2);
    return hsum_avx(sum);
}

inline float ip_avx_aligned16(const float* a, const float* b, std::size_t dim) {
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(va, vb));

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(va, vb));

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(va, vb));

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(va, vb));
    }
    __m256 sum = _mm256_add_ps(sum1, sum2);
    return hsum_avx(sum);
}

inline float l2_avx_aligned4(const float* a, const float* b, std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    return hsum_avx(sum);
}

inline float ip_avx_aligned4(const float* a, const float* b, std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    return hsum_avx(sum);
}
#endif

// AVX2 aligned kernels
#if defined(__AVX2__) && defined(__FMA__)
inline float l2_avx2_aligned16(const float* a, const float* b, std::size_t dim) {
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum1 = _mm256_fmadd_ps(diff, diff, sum1);

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_fmadd_ps(diff, diff, sum2);

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        diff = _mm256_sub_ps(va, vb);
        sum1 = _mm256_fmadd_ps(diff, diff, sum1);

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        diff = _mm256_sub_ps(va, vb);
        sum2 = _mm256_fmadd_ps(diff, diff, sum2);
    }
    __m256 sum = _mm256_add_ps(sum1, sum2);
    return hsum_avx(sum);
}

inline float ip_avx2_aligned16(const float* a, const float* b, std::size_t dim) {
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 32) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum1 = _mm256_fmadd_ps(va, vb, sum1);

        va = _mm256_loadu_ps(a + i + 8);
        vb = _mm256_loadu_ps(b + i + 8);
        sum2 = _mm256_fmadd_ps(va, vb, sum2);

        va = _mm256_loadu_ps(a + i + 16);
        vb = _mm256_loadu_ps(b + i + 16);
        sum1 = _mm256_fmadd_ps(va, vb, sum1);

        va = _mm256_loadu_ps(a + i + 24);
        vb = _mm256_loadu_ps(b + i + 24);
        sum2 = _mm256_fmadd_ps(va, vb, sum2);
    }
    __m256 sum = _mm256_add_ps(sum1, sum2);
    return hsum_avx(sum);
}

inline float l2_avx2_aligned4(const float* a, const float* b, std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    return hsum_avx(sum);
}

inline float ip_avx2_aligned4(const float* a, const float* b, std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    return hsum_avx(sum);
}
#endif

// AVX512 aligned kernels
#if defined(__AVX512F__)
inline float l2_avx512_aligned16(const float* a, const float* b, std::size_t dim) {
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 64) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum1 = _mm512_fmadd_ps(diff, diff, sum1);

        va = _mm512_loadu_ps(a + i + 16);
        vb = _mm512_loadu_ps(b + i + 16);
        diff = _mm512_sub_ps(va, vb);
        sum2 = _mm512_fmadd_ps(diff, diff, sum2);

        va = _mm512_loadu_ps(a + i + 32);
        vb = _mm512_loadu_ps(b + i + 32);
        diff = _mm512_sub_ps(va, vb);
        sum1 = _mm512_fmadd_ps(diff, diff, sum1);

        va = _mm512_loadu_ps(a + i + 48);
        vb = _mm512_loadu_ps(b + i + 48);
        diff = _mm512_sub_ps(va, vb);
        sum2 = _mm512_fmadd_ps(diff, diff, sum2);
    }
    __m512 sum = _mm512_add_ps(sum1, sum2);
    return hsum_avx512(sum);
}

inline float ip_avx512_aligned16(const float* a, const float* b, std::size_t dim) {
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    for (std::size_t i = 0; i < dim; i += 64) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum1 = _mm512_fmadd_ps(va, vb, sum1);

        va = _mm512_loadu_ps(a + i + 16);
        vb = _mm512_loadu_ps(b + i + 16);
        sum2 = _mm512_fmadd_ps(va, vb, sum2);

        va = _mm512_loadu_ps(a + i + 32);
        vb = _mm512_loadu_ps(b + i + 32);
        sum1 = _mm512_fmadd_ps(va, vb, sum1);

        va = _mm512_loadu_ps(a + i + 48);
        vb = _mm512_loadu_ps(b + i + 48);
        sum2 = _mm512_fmadd_ps(va, vb, sum2);
    }
    __m512 sum = _mm512_add_ps(sum1, sum2);
    return hsum_avx512(sum);
}
#endif

// NEON aligned kernels
#if defined(__aarch64__) || defined(_M_ARM64)
inline float l2_neon_aligned16(const float* a, const float* b, std::size_t dim) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    for (std::size_t i = 0; i < dim; i += 16) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum1 = vfmaq_f32(sum1, diff, diff);

        va = vld1q_f32(a + i + 4);
        vb = vld1q_f32(b + i + 4);
        diff = vsubq_f32(va, vb);
        sum2 = vfmaq_f32(sum2, diff, diff);

        va = vld1q_f32(a + i + 8);
        vb = vld1q_f32(b + i + 8);
        diff = vsubq_f32(va, vb);
        sum1 = vfmaq_f32(sum1, diff, diff);

        va = vld1q_f32(a + i + 12);
        vb = vld1q_f32(b + i + 12);
        diff = vsubq_f32(va, vb);
        sum2 = vfmaq_f32(sum2, diff, diff);
    }
    float32x4_t sum = vaddq_f32(sum1, sum2);
    return vaddvq_f32(sum);
}

inline float ip_neon_aligned16(const float* a, const float* b, std::size_t dim) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    for (std::size_t i = 0; i < dim; i += 16) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum1 = vfmaq_f32(sum1, va, vb);

        va = vld1q_f32(a + i + 4);
        vb = vld1q_f32(b + i + 4);
        sum2 = vfmaq_f32(sum2, va, vb);

        va = vld1q_f32(a + i + 8);
        vb = vld1q_f32(b + i + 8);
        sum1 = vfmaq_f32(sum1, va, vb);

        va = vld1q_f32(a + i + 12);
        vb = vld1q_f32(b + i + 12);
        sum2 = vfmaq_f32(sum2, va, vb);
    }
    float32x4_t sum = vaddq_f32(sum1, sum2);
    return vaddvq_f32(sum);
}

inline float l2_neon_aligned4(const float* a, const float* b, std::size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (std::size_t i = 0; i < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }
    return vaddvq_f32(sum);
}

inline float ip_neon_aligned4(const float* a, const float* b, std::size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (std::size_t i = 0; i < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    return vaddvq_f32(sum);
}
#endif

} // namespace hnsw::impl
