#pragma once

#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Platform detection: AVX-512 VPOPCNTDQ > AVX2 > NEON > scalar
//
// Hierarchy (compile-time, mutually exclusive on any given platform):
//   - AVX-512 VPOPCNTDQ: Ice Lake+ (EC2 c6i/c7i/m6i/r6i) — native popcnt
//   - AVX2: Haswell+ (EC2 c4/c5/m5) — vpshufb Mula popcount
//   - NEON: ARM (EC2 Graviton c6g/c7g, Apple Silicon) — vcntq_u8
//   - Scalar: any platform
// ---------------------------------------------------------------------------

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
#include <immintrin.h>
#define ARROW_SIMD_TIER 3  // AVX-512 VPOPCNTDQ
#elif defined(__AVX2__)
#include <immintrin.h>
#define ARROW_SIMD_TIER 2  // AVX2
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ARROW_SIMD_TIER 1  // NEON
#else
#define ARROW_SIMD_TIER 0  // scalar
#endif

namespace arrow::simd {

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------

#if ARROW_SIMD_TIER == 1  // NEON
inline uint32_t neon_reduce_u16(uint16x8_t a0, uint16x8_t a1,
                                uint16x8_t a2, uint16x8_t a3) {
    uint64x2_t sum = vpaddlq_u32(vpaddlq_u16(a0));
    sum = vaddq_u64(sum, vpaddlq_u32(vpaddlq_u16(a1)));
    sum = vaddq_u64(sum, vpaddlq_u32(vpaddlq_u16(a2)));
    sum = vaddq_u64(sum, vpaddlq_u32(vpaddlq_u16(a3)));
    return static_cast<uint32_t>(vgetq_lane_u64(sum, 0) +
                                 vgetq_lane_u64(sum, 1));
}
#endif

#if ARROW_SIMD_TIER >= 2  // AVX2 or AVX-512 (both need AVX2 helpers)
// Mula vpshufb popcount: byte-level popcount, then SAD to u64 lanes.
inline __m256i avx2_popcnt_u64(__m256i v) {
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i cnt = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, _mm256_and_si256(v, low_mask)),
        _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask)));
    return _mm256_sad_epu8(cnt, _mm256_setzero_si256());
}

inline uint32_t avx2_hsum_u64(__m256i acc) {
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi64(lo, hi);
    return static_cast<uint32_t>(
        static_cast<uint64_t>(_mm_extract_epi64(sum, 0)) +
        static_cast<uint64_t>(_mm_extract_epi64(sum, 1)));
}
#endif

#if ARROW_SIMD_TIER == 3  // AVX-512 VPOPCNTDQ
inline uint32_t avx512_hsum_u64(__m512i acc) {
    __m256i lo = _mm512_castsi512_si256(acc);
    __m256i hi = _mm512_extracti64x4_epi64(acc, 1);
    return avx2_hsum_u64(_mm256_add_epi64(lo, hi));
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Fused bitmap AND + popcount
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_and_popcount(const uint64_t* __restrict__ a,
                                    const uint64_t* __restrict__ b,
                                    uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        auto* vo = reinterpret_cast<__m512i*>(out) + i;
        __m512i r0 = _mm512_and_si512(_mm512_loadu_si512(va), _mm512_loadu_si512(vb));
        __m512i r1 = _mm512_and_si512(_mm512_loadu_si512(va+1), _mm512_loadu_si512(vb+1));
        __m512i r2 = _mm512_and_si512(_mm512_loadu_si512(va+2), _mm512_loadu_si512(vb+2));
        __m512i r3 = _mm512_and_si512(_mm512_loadu_si512(va+3), _mm512_loadu_si512(vb+3));
        _mm512_storeu_si512(vo, r0); _mm512_storeu_si512(vo+1, r1);
        _mm512_storeu_si512(vo+2, r2); _mm512_storeu_si512(vo+3, r3);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r0));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r1));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r2));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r3));
    }
    return avx512_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 2
    return avx2_harley_seal_fused(a, b, out,
        [](__m256i x, __m256i y) { return _mm256_and_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t i = 0; i < 1024; i += 8) {
        uint64x2_t r0 = vandq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        uint64x2_t r1 = vandq_u64(vld1q_u64(a+i+2), vld1q_u64(b+i+2));
        uint64x2_t r2 = vandq_u64(vld1q_u64(a+i+4), vld1q_u64(b+i+4));
        uint64x2_t r3 = vandq_u64(vld1q_u64(a+i+6), vld1q_u64(b+i+6));
        vst1q_u64(out+i, r0); vst1q_u64(out+i+2, r1);
        vst1q_u64(out+i+4, r2); vst1q_u64(out+i+6, r3);
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r0))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r1))));
        a2 = vaddq_u16(a2, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r2))));
        a3 = vaddq_u16(a3, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r3))));
    }
    return neon_reduce_u16(a0, a1, a2, a3);
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) {
        out[i] = a[i] & b[i];
        c += static_cast<uint32_t>(__builtin_popcountll(out[i]));
    }
    return c;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused bitmap OR + popcount
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_or_popcount(const uint64_t* __restrict__ a,
                                   const uint64_t* __restrict__ b,
                                   uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        auto* vo = reinterpret_cast<__m512i*>(out) + i;
        __m512i r0 = _mm512_or_si512(_mm512_loadu_si512(va), _mm512_loadu_si512(vb));
        __m512i r1 = _mm512_or_si512(_mm512_loadu_si512(va+1), _mm512_loadu_si512(vb+1));
        __m512i r2 = _mm512_or_si512(_mm512_loadu_si512(va+2), _mm512_loadu_si512(vb+2));
        __m512i r3 = _mm512_or_si512(_mm512_loadu_si512(va+3), _mm512_loadu_si512(vb+3));
        _mm512_storeu_si512(vo, r0); _mm512_storeu_si512(vo+1, r1);
        _mm512_storeu_si512(vo+2, r2); _mm512_storeu_si512(vo+3, r3);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r0));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r1));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r2));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r3));
    }
    return avx512_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 2
    return avx2_harley_seal_fused(a, b, out,
        [](__m256i x, __m256i y) { return _mm256_or_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t i = 0; i < 1024; i += 8) {
        uint64x2_t r0 = vorrq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        uint64x2_t r1 = vorrq_u64(vld1q_u64(a+i+2), vld1q_u64(b+i+2));
        uint64x2_t r2 = vorrq_u64(vld1q_u64(a+i+4), vld1q_u64(b+i+4));
        uint64x2_t r3 = vorrq_u64(vld1q_u64(a+i+6), vld1q_u64(b+i+6));
        vst1q_u64(out+i, r0); vst1q_u64(out+i+2, r1);
        vst1q_u64(out+i+4, r2); vst1q_u64(out+i+6, r3);
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r0))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r1))));
        a2 = vaddq_u16(a2, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r2))));
        a3 = vaddq_u16(a3, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r3))));
    }
    return neon_reduce_u16(a0, a1, a2, a3);
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) {
        out[i] = a[i] | b[i];
        c += static_cast<uint32_t>(__builtin_popcountll(out[i]));
    }
    return c;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused bitmap XOR + popcount
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_xor_popcount(const uint64_t* __restrict__ a,
                                    const uint64_t* __restrict__ b,
                                    uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        auto* vo = reinterpret_cast<__m512i*>(out) + i;
        __m512i r0 = _mm512_xor_si512(_mm512_loadu_si512(va), _mm512_loadu_si512(vb));
        __m512i r1 = _mm512_xor_si512(_mm512_loadu_si512(va+1), _mm512_loadu_si512(vb+1));
        __m512i r2 = _mm512_xor_si512(_mm512_loadu_si512(va+2), _mm512_loadu_si512(vb+2));
        __m512i r3 = _mm512_xor_si512(_mm512_loadu_si512(va+3), _mm512_loadu_si512(vb+3));
        _mm512_storeu_si512(vo, r0); _mm512_storeu_si512(vo+1, r1);
        _mm512_storeu_si512(vo+2, r2); _mm512_storeu_si512(vo+3, r3);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r0));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r1));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r2));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r3));
    }
    return avx512_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 2
    return avx2_harley_seal_fused(a, b, out,
        [](__m256i x, __m256i y) { return _mm256_xor_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t i = 0; i < 1024; i += 8) {
        uint64x2_t r0 = veorq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        uint64x2_t r1 = veorq_u64(vld1q_u64(a+i+2), vld1q_u64(b+i+2));
        uint64x2_t r2 = veorq_u64(vld1q_u64(a+i+4), vld1q_u64(b+i+4));
        uint64x2_t r3 = veorq_u64(vld1q_u64(a+i+6), vld1q_u64(b+i+6));
        vst1q_u64(out+i, r0); vst1q_u64(out+i+2, r1);
        vst1q_u64(out+i+4, r2); vst1q_u64(out+i+6, r3);
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r0))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r1))));
        a2 = vaddq_u16(a2, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r2))));
        a3 = vaddq_u16(a3, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r3))));
    }
    return neon_reduce_u16(a0, a1, a2, a3);
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) {
        out[i] = a[i] ^ b[i];
        c += static_cast<uint32_t>(__builtin_popcountll(out[i]));
    }
    return c;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused bitmap ANDNOT + popcount  (out[i] = a[i] & ~b[i])
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_andnot_popcount(const uint64_t* __restrict__ a,
                                       const uint64_t* __restrict__ b,
                                       uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        auto* vo = reinterpret_cast<__m512i*>(out) + i;
        // _mm512_andnot_si512(b, a) = ~b & a = a & ~b
        __m512i r0 = _mm512_andnot_si512(_mm512_loadu_si512(vb), _mm512_loadu_si512(va));
        __m512i r1 = _mm512_andnot_si512(_mm512_loadu_si512(vb+1), _mm512_loadu_si512(va+1));
        __m512i r2 = _mm512_andnot_si512(_mm512_loadu_si512(vb+2), _mm512_loadu_si512(va+2));
        __m512i r3 = _mm512_andnot_si512(_mm512_loadu_si512(vb+3), _mm512_loadu_si512(va+3));
        _mm512_storeu_si512(vo, r0); _mm512_storeu_si512(vo+1, r1);
        _mm512_storeu_si512(vo+2, r2); _mm512_storeu_si512(vo+3, r3);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r0));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r1));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r2));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r3));
    }
    return avx512_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 2
    // andnot: out = a & ~b = ~b & a. _mm256_andnot_si256(b, a) = ~b & a
    return avx2_harley_seal_fused(a, b, out,
        [](__m256i x, __m256i y) { return _mm256_andnot_si256(y, x); });
#elif ARROW_SIMD_TIER == 1
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t i = 0; i < 1024; i += 8) {
        uint64x2_t r0 = vbicq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        uint64x2_t r1 = vbicq_u64(vld1q_u64(a+i+2), vld1q_u64(b+i+2));
        uint64x2_t r2 = vbicq_u64(vld1q_u64(a+i+4), vld1q_u64(b+i+4));
        uint64x2_t r3 = vbicq_u64(vld1q_u64(a+i+6), vld1q_u64(b+i+6));
        vst1q_u64(out+i, r0); vst1q_u64(out+i+2, r1);
        vst1q_u64(out+i+4, r2); vst1q_u64(out+i+6, r3);
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r0))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r1))));
        a2 = vaddq_u16(a2, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r2))));
        a3 = vaddq_u16(a3, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r3))));
    }
    return neon_reduce_u16(a0, a1, a2, a3);
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) {
        out[i] = a[i] & ~b[i];
        c += static_cast<uint32_t>(__builtin_popcountll(out[i]));
    }
    return c;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Standalone popcount of 1024 words
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_popcount(const uint64_t* __restrict__ words) {
#if ARROW_SIMD_TIER == 3
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* v = reinterpret_cast<const __m512i*>(words) + i;
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(v)));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(v+1)));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(v+2)));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(v+3)));
    }
    return avx512_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 2
    __m256i acc = _mm256_setzero_si256();
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* v = reinterpret_cast<const __m256i*>(words) + i;
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(_mm256_loadu_si256(v)));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(_mm256_loadu_si256(v+1)));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(_mm256_loadu_si256(v+2)));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(_mm256_loadu_si256(v+3)));
    }
    return avx2_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 1
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t i = 0; i < 1024; i += 8) {
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words+i)))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words+i+2)))));
        a2 = vaddq_u16(a2, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words+i+4)))));
        a3 = vaddq_u16(a3, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words+i+6)))));
    }
    return neon_reduce_u16(a0, a1, a2, a3);
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i)
        c += static_cast<uint32_t>(__builtin_popcountll(words[i]));
    return c;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Harley-Seal CSA popcount — fewer popcnt calls via carry-save accumulation
// On AVX2: processes 16 words per CSA round. Other tiers delegate to bitmap_popcount.
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_popcount_harley_seal(const uint64_t* __restrict__ words) {
#if ARROW_SIMD_TIER == 2
    // CSA: given 3 inputs, produce (carry, sum) without popcnt
    #define CSA256(h, l, a, b, c) \
        { __m256i u = _mm256_xor_si256(a, b); \
          h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c)); \
          l = _mm256_xor_si256(u, c); }

    const __m256i* v = reinterpret_cast<const __m256i*>(words);
    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256();
    __m256i twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();

    // 1024 uint64_t = 256 __m256i. Process 16 __m256i per iteration = 16 iterations.
    for (uint32_t i = 0; i < 256; i += 16) {
        __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

        CSA256(twosA, ones, ones, _mm256_loadu_si256(v + i + 0), _mm256_loadu_si256(v + i + 1));
        CSA256(twosB, ones, ones, _mm256_loadu_si256(v + i + 2), _mm256_loadu_si256(v + i + 3));
        CSA256(foursA, twos, twos, twosA, twosB);

        CSA256(twosA, ones, ones, _mm256_loadu_si256(v + i + 4), _mm256_loadu_si256(v + i + 5));
        CSA256(twosB, ones, ones, _mm256_loadu_si256(v + i + 6), _mm256_loadu_si256(v + i + 7));
        CSA256(foursB, twos, twos, twosA, twosB);
        CSA256(eightsA, fours, fours, foursA, foursB);

        CSA256(twosA, ones, ones, _mm256_loadu_si256(v + i + 8), _mm256_loadu_si256(v + i + 9));
        CSA256(twosB, ones, ones, _mm256_loadu_si256(v + i + 10), _mm256_loadu_si256(v + i + 11));
        CSA256(foursA, twos, twos, twosA, twosB);

        CSA256(twosA, ones, ones, _mm256_loadu_si256(v + i + 12), _mm256_loadu_si256(v + i + 13));
        CSA256(twosB, ones, ones, _mm256_loadu_si256(v + i + 14), _mm256_loadu_si256(v + i + 15));
        CSA256(foursB, twos, twos, twosA, twosB);
        CSA256(eightsB, fours, fours, foursA, foursB);

        __m256i sixteens;
        CSA256(sixteens, eights, eights, eightsA, eightsB);
        total = _mm256_add_epi64(total, avx2_popcnt_u64(sixteens));
    }

    // Weight the residual accumulators
    total = _mm256_slli_epi64(total, 4);  // sixteens * 16
    total = _mm256_add_epi64(total, _mm256_slli_epi64(avx2_popcnt_u64(eights), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(avx2_popcnt_u64(fours), 2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(avx2_popcnt_u64(twos), 1));
    total = _mm256_add_epi64(total, avx2_popcnt_u64(ones));

    #undef CSA256
    return avx2_hsum_u64(total);
#else
    // AVX-512 and NEON already have efficient native popcount
    return bitmap_popcount(words);
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused Harley-Seal CSA with bitwise ops (AVX2 only)
// Interleaves the bitwise operation with carry-save popcount accumulation,
// reducing vpshufb calls from ~256 to ~20 per 8KB bitmap pair.
// ═══════════════════════════════════════════════════════════════════════════

#if ARROW_SIMD_TIER == 2
template <typename BitwiseOp>
inline uint32_t avx2_harley_seal_fused(const uint64_t* __restrict__ a,
                                       const uint64_t* __restrict__ b,
                                       uint64_t* __restrict__ out,
                                       BitwiseOp op) {
    #define FUSED_CSA256(h, l, a, b, c) \
        { __m256i u = _mm256_xor_si256(a, b); \
          h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c)); \
          l = _mm256_xor_si256(u, c); }

    const __m256i* va = reinterpret_cast<const __m256i*>(a);
    const __m256i* vb = reinterpret_cast<const __m256i*>(b);
    __m256i* vo = reinterpret_cast<__m256i*>(out);

    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256();
    __m256i twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();

    // Helper lambda: compute op, store result, return it for CSA
    auto fop = [&](uint32_t idx) -> __m256i {
        __m256i r = op(_mm256_loadu_si256(va + idx), _mm256_loadu_si256(vb + idx));
        _mm256_storeu_si256(vo + idx, r);
        return r;
    };

    // 1024 uint64_t = 256 __m256i. Process 16 per iteration = 16 iterations.
    for (uint32_t i = 0; i < 256; i += 16) {
        __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

        FUSED_CSA256(twosA, ones, ones, fop(i + 0), fop(i + 1));
        FUSED_CSA256(twosB, ones, ones, fop(i + 2), fop(i + 3));
        FUSED_CSA256(foursA, twos, twos, twosA, twosB);

        FUSED_CSA256(twosA, ones, ones, fop(i + 4), fop(i + 5));
        FUSED_CSA256(twosB, ones, ones, fop(i + 6), fop(i + 7));
        FUSED_CSA256(foursB, twos, twos, twosA, twosB);
        FUSED_CSA256(eightsA, fours, fours, foursA, foursB);

        FUSED_CSA256(twosA, ones, ones, fop(i + 8), fop(i + 9));
        FUSED_CSA256(twosB, ones, ones, fop(i + 10), fop(i + 11));
        FUSED_CSA256(foursA, twos, twos, twosA, twosB);

        FUSED_CSA256(twosA, ones, ones, fop(i + 12), fop(i + 13));
        FUSED_CSA256(twosB, ones, ones, fop(i + 14), fop(i + 15));
        FUSED_CSA256(foursB, twos, twos, twosA, twosB);
        FUSED_CSA256(eightsB, fours, fours, foursA, foursB);

        __m256i sixteens;
        FUSED_CSA256(sixteens, eights, eights, eightsA, eightsB);
        total = _mm256_add_epi64(total, avx2_popcnt_u64(sixteens));
    }

    // Weight the residual accumulators
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(avx2_popcnt_u64(eights), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(avx2_popcnt_u64(fours), 2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(avx2_popcnt_u64(twos), 1));
    total = _mm256_add_epi64(total, avx2_popcnt_u64(ones));

    #undef FUSED_CSA256
    return avx2_hsum_u64(total);
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Nocard variants — bitwise op without cardinality (deferred computation)
// Uses templates to eliminate per-op duplication.
// ═══════════════════════════════════════════════════════════════════════════

#if ARROW_SIMD_TIER == 3
template <typename Op>
inline void avx512_bitmap_op_nocard(const uint64_t* a, const uint64_t* b,
                                    uint64_t* out, Op op) {
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        auto* vo = reinterpret_cast<__m512i*>(out) + i;
        _mm512_storeu_si512(vo, op(_mm512_loadu_si512(va), _mm512_loadu_si512(vb)));
        _mm512_storeu_si512(vo+1, op(_mm512_loadu_si512(va+1), _mm512_loadu_si512(vb+1)));
        _mm512_storeu_si512(vo+2, op(_mm512_loadu_si512(va+2), _mm512_loadu_si512(vb+2)));
        _mm512_storeu_si512(vo+3, op(_mm512_loadu_si512(va+3), _mm512_loadu_si512(vb+3)));
    }
}
#endif

#if ARROW_SIMD_TIER == 2
template <typename Op>
inline void avx2_bitmap_op_nocard(const uint64_t* a, const uint64_t* b,
                                  uint64_t* out, Op op) {
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* va = reinterpret_cast<const __m256i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m256i*>(b) + i;
        auto* vo = reinterpret_cast<__m256i*>(out) + i;
        _mm256_storeu_si256(vo, op(_mm256_loadu_si256(va), _mm256_loadu_si256(vb)));
        _mm256_storeu_si256(vo+1, op(_mm256_loadu_si256(va+1), _mm256_loadu_si256(vb+1)));
        _mm256_storeu_si256(vo+2, op(_mm256_loadu_si256(va+2), _mm256_loadu_si256(vb+2)));
        _mm256_storeu_si256(vo+3, op(_mm256_loadu_si256(va+3), _mm256_loadu_si256(vb+3)));
    }
}
#endif

#if ARROW_SIMD_TIER == 1
template <typename Op>
inline void neon_bitmap_op_nocard(const uint64_t* a, const uint64_t* b,
                                  uint64_t* out, Op op) {
    for (uint32_t i = 0; i < 1024; i += 8) {
        vst1q_u64(out+i, op(vld1q_u64(a+i), vld1q_u64(b+i)));
        vst1q_u64(out+i+2, op(vld1q_u64(a+i+2), vld1q_u64(b+i+2)));
        vst1q_u64(out+i+4, op(vld1q_u64(a+i+4), vld1q_u64(b+i+4)));
        vst1q_u64(out+i+6, op(vld1q_u64(a+i+6), vld1q_u64(b+i+6)));
    }
}
#endif

inline void bitmap_and_nocard(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b, uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    avx512_bitmap_op_nocard(a, b, out, [](__m512i x, __m512i y) { return _mm512_and_si512(x, y); });
#elif ARROW_SIMD_TIER == 2
    avx2_bitmap_op_nocard(a, b, out, [](__m256i x, __m256i y) { return _mm256_and_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    neon_bitmap_op_nocard(a, b, out, [](uint64x2_t x, uint64x2_t y) { return vandq_u64(x, y); });
#else
    for (uint32_t i = 0; i < 1024; ++i) out[i] = a[i] & b[i];
#endif
}

inline void bitmap_or_nocard(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b, uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    avx512_bitmap_op_nocard(a, b, out, [](__m512i x, __m512i y) { return _mm512_or_si512(x, y); });
#elif ARROW_SIMD_TIER == 2
    avx2_bitmap_op_nocard(a, b, out, [](__m256i x, __m256i y) { return _mm256_or_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    neon_bitmap_op_nocard(a, b, out, [](uint64x2_t x, uint64x2_t y) { return vorrq_u64(x, y); });
#else
    for (uint32_t i = 0; i < 1024; ++i) out[i] = a[i] | b[i];
#endif
}

inline void bitmap_xor_nocard(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b, uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    avx512_bitmap_op_nocard(a, b, out, [](__m512i x, __m512i y) { return _mm512_xor_si512(x, y); });
#elif ARROW_SIMD_TIER == 2
    avx2_bitmap_op_nocard(a, b, out, [](__m256i x, __m256i y) { return _mm256_xor_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    neon_bitmap_op_nocard(a, b, out, [](uint64x2_t x, uint64x2_t y) { return veorq_u64(x, y); });
#else
    for (uint32_t i = 0; i < 1024; ++i) out[i] = a[i] ^ b[i];
#endif
}

inline void bitmap_andnot_nocard(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b, uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    avx512_bitmap_op_nocard(a, b, out, [](__m512i x, __m512i y) { return _mm512_andnot_si512(y, x); });
#elif ARROW_SIMD_TIER == 2
    avx2_bitmap_op_nocard(a, b, out, [](__m256i x, __m256i y) { return _mm256_andnot_si256(y, x); });
#elif ARROW_SIMD_TIER == 1
    neon_bitmap_op_nocard(a, b, out, [](uint64x2_t x, uint64x2_t y) { return vbicq_u64(x, y); });
#else
    for (uint32_t i = 0; i < 1024; ++i) out[i] = a[i] & ~b[i];
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Bitmap NOT without cardinality: out[i] = ~a[i]
// ═══════════════════════════════════════════════════════════════════════════
inline void bitmap_not_nocard(const uint64_t* __restrict__ a, uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    __m512i ones = _mm512_set1_epi64(static_cast<int64_t>(~0ULL));
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        auto* vo = reinterpret_cast<__m512i*>(out) + i;
        _mm512_storeu_si512(vo, _mm512_xor_si512(_mm512_loadu_si512(va), ones));
        _mm512_storeu_si512(vo+1, _mm512_xor_si512(_mm512_loadu_si512(va+1), ones));
        _mm512_storeu_si512(vo+2, _mm512_xor_si512(_mm512_loadu_si512(va+2), ones));
        _mm512_storeu_si512(vo+3, _mm512_xor_si512(_mm512_loadu_si512(va+3), ones));
    }
#elif ARROW_SIMD_TIER == 2
    __m256i ones = _mm256_set1_epi64x(static_cast<int64_t>(~0ULL));
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* va = reinterpret_cast<const __m256i*>(a) + i;
        auto* vo = reinterpret_cast<__m256i*>(out) + i;
        _mm256_storeu_si256(vo, _mm256_xor_si256(_mm256_loadu_si256(va), ones));
        _mm256_storeu_si256(vo+1, _mm256_xor_si256(_mm256_loadu_si256(va+1), ones));
        _mm256_storeu_si256(vo+2, _mm256_xor_si256(_mm256_loadu_si256(va+2), ones));
        _mm256_storeu_si256(vo+3, _mm256_xor_si256(_mm256_loadu_si256(va+3), ones));
    }
#elif ARROW_SIMD_TIER == 1
    for (uint32_t i = 0; i < 1024; i += 8) {
        vst1q_u64(out+i, vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(vld1q_u64(a+i)))));
        vst1q_u64(out+i+2, vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(vld1q_u64(a+i+2)))));
        vst1q_u64(out+i+4, vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(vld1q_u64(a+i+4)))));
        vst1q_u64(out+i+6, vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(vld1q_u64(a+i+6)))));
    }
#else
    for (uint32_t i = 0; i < 1024; ++i) out[i] = ~a[i];
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Bitmap equality check
// ═══════════════════════════════════════════════════════════════════════════
inline bool bitmap_equal(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b) {
#if ARROW_SIMD_TIER == 3
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        __mmask8 m0 = _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(va), _mm512_loadu_si512(vb));
        __mmask8 m1 = _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(va+1), _mm512_loadu_si512(vb+1));
        __mmask8 m2 = _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(va+2), _mm512_loadu_si512(vb+2));
        __mmask8 m3 = _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(va+3), _mm512_loadu_si512(vb+3));
        if ((m0 & m1 & m2 & m3) != 0xFF) return false;
    }
    return true;
#elif ARROW_SIMD_TIER == 2
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* va = reinterpret_cast<const __m256i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m256i*>(b) + i;
        __m256i cmp0 = _mm256_cmpeq_epi64(_mm256_loadu_si256(va), _mm256_loadu_si256(vb));
        __m256i cmp1 = _mm256_cmpeq_epi64(_mm256_loadu_si256(va+1), _mm256_loadu_si256(vb+1));
        __m256i cmp2 = _mm256_cmpeq_epi64(_mm256_loadu_si256(va+2), _mm256_loadu_si256(vb+2));
        __m256i cmp3 = _mm256_cmpeq_epi64(_mm256_loadu_si256(va+3), _mm256_loadu_si256(vb+3));
        __m256i all = _mm256_and_si256(_mm256_and_si256(cmp0, cmp1),
                                       _mm256_and_si256(cmp2, cmp3));
        if (_mm256_movemask_epi8(all) != -1) return false;
    }
    return true;
#elif ARROW_SIMD_TIER == 1
    for (uint32_t i = 0; i < 1024; i += 2) {
        uint64x2_t cmp = vceqq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        if (vminvq_u32(vreinterpretq_u32_u64(cmp)) == 0) return false;
    }
    return true;
#else
    return std::memcmp(a, b, 1024 * sizeof(uint64_t)) == 0;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Extract set bit positions from bitmap into uint16_t array.
// Returns count of values written.
//
// Uses direct ctz (count-trailing-zeros) + clear-lowest-set-bit loop.
// On ARM64: RBIT+CLZ is single-cycle; on x86-64: TZCNT is single-cycle.
// Processes 4 words at a time and skips runs of zero words quickly.
// ═══════════════════════════════════════════════════════════════════════════

inline uint32_t bitmap_to_array(const uint64_t* __restrict__ words,
                                uint16_t* __restrict__ out) {
    uint32_t pos = 0;
    for (uint32_t w = 0; w < 1024; ++w) {
        uint64_t bits = words[w];
        if (bits == 0) continue;
        uint32_t base = w << 6;
        while (bits) {
            uint32_t tz = static_cast<uint32_t>(__builtin_ctzll(bits));
            out[pos++] = static_cast<uint16_t>(base + tz);
            bits &= bits - 1;  // clear lowest set bit
        }
    }
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════
// Extract set bit positions from bitmap into uint32_t array with base offset.
// Same ctz loop as bitmap_to_array but outputs uint32_t = base + bit_position.
// ═══════════════════════════════════════════════════════════════════════════

inline uint32_t bitmap_to_uint32_array(const uint64_t* __restrict__ words,
                                        uint32_t base,
                                        uint32_t* __restrict__ out) {
    uint32_t pos = 0;
    for (uint32_t w = 0; w < 1024; ++w) {
        uint64_t bits = words[w];
        if (bits == 0) continue;
        uint32_t wbase = base + (w << 6);
        while (bits) {
            out[pos++] = wbase + static_cast<uint32_t>(__builtin_ctzll(bits));
            bits &= bits - 1;
        }
    }
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════
// Bitmap NOT + popcount: out[i] = ~a[i], return popcount
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_not_popcount(const uint64_t* __restrict__ a, uint64_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 3
    __m512i acc = _mm512_setzero_si512();
    __m512i ones = _mm512_set1_epi64(static_cast<int64_t>(~0ULL));
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        auto* vo = reinterpret_cast<__m512i*>(out) + i;
        __m512i r0 = _mm512_xor_si512(_mm512_loadu_si512(va), ones);
        __m512i r1 = _mm512_xor_si512(_mm512_loadu_si512(va+1), ones);
        __m512i r2 = _mm512_xor_si512(_mm512_loadu_si512(va+2), ones);
        __m512i r3 = _mm512_xor_si512(_mm512_loadu_si512(va+3), ones);
        _mm512_storeu_si512(vo, r0); _mm512_storeu_si512(vo+1, r1);
        _mm512_storeu_si512(vo+2, r2); _mm512_storeu_si512(vo+3, r3);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r0));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r1));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r2));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r3));
    }
    return avx512_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 2
    __m256i acc = _mm256_setzero_si256();
    __m256i ones = _mm256_set1_epi64x(static_cast<int64_t>(~0ULL));
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* va = reinterpret_cast<const __m256i*>(a) + i;
        auto* vo = reinterpret_cast<__m256i*>(out) + i;
        __m256i r0 = _mm256_xor_si256(_mm256_loadu_si256(va), ones);
        __m256i r1 = _mm256_xor_si256(_mm256_loadu_si256(va+1), ones);
        __m256i r2 = _mm256_xor_si256(_mm256_loadu_si256(va+2), ones);
        __m256i r3 = _mm256_xor_si256(_mm256_loadu_si256(va+3), ones);
        _mm256_storeu_si256(vo, r0); _mm256_storeu_si256(vo+1, r1);
        _mm256_storeu_si256(vo+2, r2); _mm256_storeu_si256(vo+3, r3);
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(r0));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(r1));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(r2));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(r3));
    }
    return avx2_hsum_u64(acc);
#elif ARROW_SIMD_TIER == 1
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    uint64x2_t ones = vdupq_n_u64(~0ULL);
    for (uint32_t i = 0; i < 1024; i += 8) {
        uint64x2_t r0 = veorq_u64(vld1q_u64(a+i), ones);
        uint64x2_t r1 = veorq_u64(vld1q_u64(a+i+2), ones);
        uint64x2_t r2 = veorq_u64(vld1q_u64(a+i+4), ones);
        uint64x2_t r3 = veorq_u64(vld1q_u64(a+i+6), ones);
        vst1q_u64(out+i, r0); vst1q_u64(out+i+2, r1);
        vst1q_u64(out+i+4, r2); vst1q_u64(out+i+6, r3);
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r0))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r1))));
        a2 = vaddq_u16(a2, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r2))));
        a3 = vaddq_u16(a3, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(r3))));
    }
    return neon_reduce_u16(a0, a1, a2, a3);
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) {
        out[i] = ~a[i];
        c += static_cast<uint32_t>(__builtin_popcountll(out[i]));
    }
    return c;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Cardinality-only (no output array) — count bits without materializing.
// Uses templates per tier to eliminate per-op duplication.
// ═══════════════════════════════════════════════════════════════════════════

#if ARROW_SIMD_TIER == 3
template <typename Op>
inline uint32_t avx512_bitmap_op_popcount_noout(const uint64_t* a, const uint64_t* b, Op op) {
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(op(_mm512_loadu_si512(va), _mm512_loadu_si512(vb))));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(op(_mm512_loadu_si512(va+1), _mm512_loadu_si512(vb+1))));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(op(_mm512_loadu_si512(va+2), _mm512_loadu_si512(vb+2))));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(op(_mm512_loadu_si512(va+3), _mm512_loadu_si512(vb+3))));
    }
    return avx512_hsum_u64(acc);
}
#endif

#if ARROW_SIMD_TIER == 2
template <typename Op>
inline uint32_t avx2_bitmap_op_popcount_noout(const uint64_t* a, const uint64_t* b, Op op) {
    __m256i acc = _mm256_setzero_si256();
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* va = reinterpret_cast<const __m256i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m256i*>(b) + i;
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(op(_mm256_loadu_si256(va), _mm256_loadu_si256(vb))));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(op(_mm256_loadu_si256(va+1), _mm256_loadu_si256(vb+1))));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(op(_mm256_loadu_si256(va+2), _mm256_loadu_si256(vb+2))));
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(op(_mm256_loadu_si256(va+3), _mm256_loadu_si256(vb+3))));
    }
    return avx2_hsum_u64(acc);
}
#endif

#if ARROW_SIMD_TIER == 1
template <typename Op>
inline uint32_t neon_bitmap_op_popcount_noout(const uint64_t* a, const uint64_t* b, Op op) {
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t i = 0; i < 1024; i += 8) {
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(op(vld1q_u64(a+i), vld1q_u64(b+i))))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(op(vld1q_u64(a+i+2), vld1q_u64(b+i+2))))));
        a2 = vaddq_u16(a2, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(op(vld1q_u64(a+i+4), vld1q_u64(b+i+4))))));
        a3 = vaddq_u16(a3, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(op(vld1q_u64(a+i+6), vld1q_u64(b+i+6))))));
    }
    return neon_reduce_u16(a0, a1, a2, a3);
}
#endif

inline uint32_t bitmap_and_popcount_noout(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b) {
#if ARROW_SIMD_TIER == 3
    return avx512_bitmap_op_popcount_noout(a, b, [](__m512i x, __m512i y) { return _mm512_and_si512(x, y); });
#elif ARROW_SIMD_TIER == 2
    return avx2_bitmap_op_popcount_noout(a, b, [](__m256i x, __m256i y) { return _mm256_and_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    return neon_bitmap_op_popcount_noout(a, b, [](uint64x2_t x, uint64x2_t y) { return vandq_u64(x, y); });
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) c += static_cast<uint32_t>(__builtin_popcountll(a[i] & b[i]));
    return c;
#endif
}

inline uint32_t bitmap_or_popcount_noout(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b) {
#if ARROW_SIMD_TIER == 3
    return avx512_bitmap_op_popcount_noout(a, b, [](__m512i x, __m512i y) { return _mm512_or_si512(x, y); });
#elif ARROW_SIMD_TIER == 2
    return avx2_bitmap_op_popcount_noout(a, b, [](__m256i x, __m256i y) { return _mm256_or_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    return neon_bitmap_op_popcount_noout(a, b, [](uint64x2_t x, uint64x2_t y) { return vorrq_u64(x, y); });
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) c += static_cast<uint32_t>(__builtin_popcountll(a[i] | b[i]));
    return c;
#endif
}

inline uint32_t bitmap_xor_popcount_noout(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b) {
#if ARROW_SIMD_TIER == 3
    return avx512_bitmap_op_popcount_noout(a, b, [](__m512i x, __m512i y) { return _mm512_xor_si512(x, y); });
#elif ARROW_SIMD_TIER == 2
    return avx2_bitmap_op_popcount_noout(a, b, [](__m256i x, __m256i y) { return _mm256_xor_si256(x, y); });
#elif ARROW_SIMD_TIER == 1
    return neon_bitmap_op_popcount_noout(a, b, [](uint64x2_t x, uint64x2_t y) { return veorq_u64(x, y); });
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) c += static_cast<uint32_t>(__builtin_popcountll(a[i] ^ b[i]));
    return c;
#endif
}

inline uint32_t bitmap_andnot_popcount_noout(const uint64_t* __restrict__ a, const uint64_t* __restrict__ b) {
#if ARROW_SIMD_TIER == 3
    return avx512_bitmap_op_popcount_noout(a, b, [](__m512i x, __m512i y) { return _mm512_andnot_si512(y, x); });
#elif ARROW_SIMD_TIER == 2
    return avx2_bitmap_op_popcount_noout(a, b, [](__m256i x, __m256i y) { return _mm256_andnot_si256(y, x); });
#elif ARROW_SIMD_TIER == 1
    return neon_bitmap_op_popcount_noout(a, b, [](uint64x2_t x, uint64x2_t y) { return vbicq_u64(x, y); });
#else
    uint32_t c = 0;
    for (uint32_t i = 0; i < 1024; ++i) c += static_cast<uint32_t>(__builtin_popcountll(a[i] & ~b[i]));
    return c;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Popcount of first n words (variable-length prefix)
// Used by containerRank for BitmapContainer.
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_popcount_n(const uint64_t* __restrict__ words, uint32_t n) {
#if ARROW_SIMD_TIER == 3
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const auto* v = reinterpret_cast<const __m512i*>(words + i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(v)));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(v + 1)));
    }
    uint32_t total = avx512_hsum_u64(acc);
    for (; i < n; ++i)
        total += static_cast<uint32_t>(__builtin_popcountll(words[i]));
    return total;
#elif ARROW_SIMD_TIER == 2
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const auto* v = reinterpret_cast<const __m256i*>(words + i);
        acc = _mm256_add_epi64(acc, avx2_popcnt_u64(_mm256_loadu_si256(v)));
    }
    uint32_t total = avx2_hsum_u64(acc);
    for (; i < n; ++i)
        total += static_cast<uint32_t>(__builtin_popcountll(words[i]));
    return total;
#elif ARROW_SIMD_TIER == 1
    uint32_t total = 0;
    uint32_t i = 0;
    // NEON: process 8 words (4 × uint64x2_t) at a time
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0;
    for (; i + 8 <= n; i += 8) {
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words + i)))));
        a0 = vaddq_u16(a0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words + i + 2)))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words + i + 4)))));
        a1 = vaddq_u16(a1, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words + i + 6)))));
    }
    // Reduce NEON accumulators
    uint64x2_t sum = vpaddlq_u32(vpaddlq_u16(a0));
    sum = vaddq_u64(sum, vpaddlq_u32(vpaddlq_u16(a1)));
    total = static_cast<uint32_t>(vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1));
    for (; i < n; ++i)
        total += static_cast<uint32_t>(__builtin_popcountll(words[i]));
    return total;
#else
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i)
        total += static_cast<uint32_t>(__builtin_popcountll(words[i]));
    return total;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch bit manipulation from sorted uint16_t arrays
// ═══════════════════════════════════════════════════════════════════════════

// Set bits from a sorted uint16_t array. Simple load-OR-store per element,
// matching CRoaring's bitset_set_list. Avoids the branch misprediction cost
// of a coalescing inner loop on sparse data.
inline void bitmap_set_list(uint64_t* __restrict__ words, const uint16_t* __restrict__ list, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t pos = list[i];
        words[pos >> 6] |= UINT64_C(1) << (pos & 63);
    }
}

inline void bitmap_clear_list(uint64_t* __restrict__ words, const uint16_t* __restrict__ list, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t pos = list[i];
        words[pos >> 6] &= ~(UINT64_C(1) << (pos & 63));
    }
}

inline void bitmap_flip_list(uint64_t* __restrict__ words, const uint16_t* __restrict__ list, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t pos = list[i];
        words[pos >> 6] ^= UINT64_C(1) << (pos & 63);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused AND + extract set bit positions (avoids intermediate bitmap)
// ═══════════════════════════════════════════════════════════════════════════
inline uint32_t bitmap_and_extract(const uint64_t* __restrict__ a,
                                   const uint64_t* __restrict__ b,
                                   uint16_t* __restrict__ out) {
    uint32_t pos = 0;
    for (uint32_t w = 0; w < 1024; ++w) {
        uint64_t bits = a[w] & b[w];
        while (bits) {
            out[pos++] = static_cast<uint16_t>((w << 6) | __builtin_ctzll(bits));
            bits &= bits - 1;
        }
    }
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════
// Bitmap intersection test — early exit on first nonzero AND
// ═══════════════════════════════════════════════════════════════════════════
inline bool bitmap_intersects_any(const uint64_t* __restrict__ a,
                                  const uint64_t* __restrict__ b) {
#if ARROW_SIMD_TIER == 3
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* va = reinterpret_cast<const __m512i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m512i*>(b) + i;
        __mmask8 m0 = _mm512_test_epi64_mask(_mm512_loadu_si512(va), _mm512_loadu_si512(vb));
        __mmask8 m1 = _mm512_test_epi64_mask(_mm512_loadu_si512(va+1), _mm512_loadu_si512(vb+1));
        __mmask8 m2 = _mm512_test_epi64_mask(_mm512_loadu_si512(va+2), _mm512_loadu_si512(vb+2));
        __mmask8 m3 = _mm512_test_epi64_mask(_mm512_loadu_si512(va+3), _mm512_loadu_si512(vb+3));
        if ((m0 | m1 | m2 | m3) != 0) return true;
    }
    return false;
#elif ARROW_SIMD_TIER == 2
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* va = reinterpret_cast<const __m256i*>(a) + i;
        const auto* vb = reinterpret_cast<const __m256i*>(b) + i;
        __m256i r0 = _mm256_and_si256(_mm256_loadu_si256(va), _mm256_loadu_si256(vb));
        __m256i r1 = _mm256_and_si256(_mm256_loadu_si256(va+1), _mm256_loadu_si256(vb+1));
        __m256i r2 = _mm256_and_si256(_mm256_loadu_si256(va+2), _mm256_loadu_si256(vb+2));
        __m256i r3 = _mm256_and_si256(_mm256_loadu_si256(va+3), _mm256_loadu_si256(vb+3));
        __m256i any = _mm256_or_si256(_mm256_or_si256(r0, r1), _mm256_or_si256(r2, r3));
        if (!_mm256_testz_si256(any, any)) return true;
    }
    return false;
#elif ARROW_SIMD_TIER == 1
    for (uint32_t i = 0; i < 1024; i += 8) {
        uint64x2_t r0 = vandq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        uint64x2_t r1 = vandq_u64(vld1q_u64(a+i+2), vld1q_u64(b+i+2));
        uint64x2_t r2 = vandq_u64(vld1q_u64(a+i+4), vld1q_u64(b+i+4));
        uint64x2_t r3 = vandq_u64(vld1q_u64(a+i+6), vld1q_u64(b+i+6));
        uint64x2_t any = vorrq_u64(vorrq_u64(r0, r1), vorrq_u64(r2, r3));
        if (vmaxvq_u32(vreinterpretq_u32_u64(any)) != 0) return true;
    }
    return false;
#else
    for (uint32_t i = 0; i < 1024; ++i)
        if (a[i] & b[i]) return true;
    return false;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Bitmap empty test — early exit on first nonzero word
// ═══════════════════════════════════════════════════════════════════════════
inline bool bitmap_is_empty(const uint64_t* __restrict__ words) {
#if ARROW_SIMD_TIER == 3
    for (uint32_t i = 0; i < 128; i += 4) {
        const auto* v = reinterpret_cast<const __m512i*>(words) + i;
        __m512i any = _mm512_or_si512(
            _mm512_or_si512(_mm512_loadu_si512(v), _mm512_loadu_si512(v+1)),
            _mm512_or_si512(_mm512_loadu_si512(v+2), _mm512_loadu_si512(v+3)));
        if (_mm512_test_epi64_mask(any, any) != 0) return false;
    }
    return true;
#elif ARROW_SIMD_TIER == 2
    for (uint32_t i = 0; i < 256; i += 4) {
        const auto* v = reinterpret_cast<const __m256i*>(words) + i;
        __m256i any = _mm256_or_si256(
            _mm256_or_si256(_mm256_loadu_si256(v), _mm256_loadu_si256(v+1)),
            _mm256_or_si256(_mm256_loadu_si256(v+2), _mm256_loadu_si256(v+3)));
        if (!_mm256_testz_si256(any, any)) return false;
    }
    return true;
#elif ARROW_SIMD_TIER == 1
    for (uint32_t i = 0; i < 1024; i += 8) {
        uint64x2_t any = vorrq_u64(
            vorrq_u64(vld1q_u64(words+i), vld1q_u64(words+i+2)),
            vorrq_u64(vld1q_u64(words+i+4), vld1q_u64(words+i+6)));
        if (vmaxvq_u32(vreinterpretq_u32_u64(any)) != 0) return false;
    }
    return true;
#else
    for (uint32_t i = 0; i < 1024; ++i)
        if (words[i]) return false;
    return true;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMD sorted array intersection (Gap 1)
// Returns count of matching elements written to `out`.
// Both inputs must be sorted. Output is sorted.
// ═══════════════════════════════════════════════════════════════════════════

#if ARROW_SIMD_TIER == 1  // NEON

// NEON: broadcast-compare intersection.
// For each element in A[0..7], broadcast it to all 8 lanes, compare against
// B[0..7] using vceqq_u16, reduce with vmaxvq. If any lane matches, emit.
inline uint32_t array_intersect_neon(const uint16_t* __restrict__ a, uint32_t na,
                                     const uint16_t* __restrict__ b, uint32_t nb,
                                     uint16_t* __restrict__ out) {
    uint32_t pos = 0;
    uint32_t ia = 0, ib = 0;

    while (ia + 8 <= na && ib + 8 <= nb) {
        // Skip blocks that can't intersect.
        if (a[ia + 7] < b[ib]) { ia += 8; continue; }
        if (b[ib + 7] < a[ia]) { ib += 8; continue; }

        uint16x8_t vb = vld1q_u16(b + ib);

        // Check each of a[ia..ia+7] against all of vb.
        for (uint32_t k = 0; k < 8 && ia + k < na; ++k) {
            uint16x8_t va = vdupq_n_u16(a[ia + k]);
            uint16x8_t cmp = vceqq_u16(va, vb);
            if (vmaxvq_u16(cmp) != 0) {
                out[pos++] = a[ia + k];
            }
        }

        // Advance the array whose maximum is smaller.
        if (a[ia + 7] <= b[ib + 7]) ia += 8;
        else ib += 8;
    }

    // Scalar tail.
    while (ia < na && ib < nb) {
        if (a[ia] < b[ib]) ++ia;
        else if (a[ia] > b[ib]) ++ib;
        else { out[pos++] = a[ia]; ++ia; ++ib; }
    }
    return pos;
}

#endif  // NEON

// Portable (all tiers): sorted array intersection with block-skipping.
// Falls through to NEON intrinsics on tier 1, scalar two-pointer otherwise.
inline uint32_t array_intersect(const uint16_t* __restrict__ a, uint32_t na,
                                const uint16_t* __restrict__ b, uint32_t nb,
                                uint16_t* __restrict__ out) {
#if ARROW_SIMD_TIER == 1
    return array_intersect_neon(a, na, b, nb, out);
#else
    // Scalar two-pointer with block-skipping.
    uint32_t pos = 0;
    uint32_t ia = 0, ib = 0;
    while (ia < na && ib < nb) {
        if (a[ia] < b[ib]) ++ia;
        else if (a[ia] > b[ib]) ++ib;
        else { out[pos++] = a[ia]; ++ia; ++ib; }
    }
    return pos;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMD sorted array union (Gap 2)
// Merge two sorted arrays, deduplicating. Output is sorted.
// ═══════════════════════════════════════════════════════════════════════════

inline uint32_t array_union(const uint16_t* __restrict__ a, uint32_t na,
                            const uint16_t* __restrict__ b, uint32_t nb,
                            uint16_t* __restrict__ out) {
    uint32_t pos = 0;
    uint32_t ia = 0, ib = 0;

    while (ia < na && ib < nb) {
        // Bulk-copy from A when next 8+ elements are all < B[ib].
        if (ia + 8 <= na && a[ia + 7] < b[ib]) {
            std::memcpy(out + pos, a + ia, 8 * sizeof(uint16_t));
            pos += 8;
            ia += 8;
            continue;
        }
        // Symmetric: bulk-copy from B.
        if (ib + 8 <= nb && b[ib + 7] < a[ia]) {
            std::memcpy(out + pos, b + ib, 8 * sizeof(uint16_t));
            pos += 8;
            ib += 8;
            continue;
        }

        if (a[ia] < b[ib]) out[pos++] = a[ia++];
        else if (a[ia] > b[ib]) out[pos++] = b[ib++];
        else { out[pos++] = a[ia++]; ++ib; }
    }
    // Copy remainder.
    if (ia < na) {
        std::memcpy(out + pos, a + ia, (na - ia) * sizeof(uint16_t));
        pos += na - ia;
    }
    if (ib < nb) {
        std::memcpy(out + pos, b + ib, (nb - ib) * sizeof(uint16_t));
        pos += nb - ib;
    }
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════
// Sorted array XOR (Gap 2): emit elements in exactly one of A or B.
// ═══════════════════════════════════════════════════════════════════════════

inline uint32_t array_xor(const uint16_t* __restrict__ a, uint32_t na,
                          const uint16_t* __restrict__ b, uint32_t nb,
                          uint16_t* __restrict__ out) {
    uint32_t pos = 0;
    uint32_t ia = 0, ib = 0;

    while (ia < na && ib < nb) {
        // Bulk-copy from A when next 8+ elements are all < B[ib].
        if (ia + 8 <= na && a[ia + 7] < b[ib]) {
            std::memcpy(out + pos, a + ia, 8 * sizeof(uint16_t));
            pos += 8;
            ia += 8;
            continue;
        }
        if (ib + 8 <= nb && b[ib + 7] < a[ia]) {
            std::memcpy(out + pos, b + ib, 8 * sizeof(uint16_t));
            pos += 8;
            ib += 8;
            continue;
        }

        if (a[ia] < b[ib]) out[pos++] = a[ia++];
        else if (a[ia] > b[ib]) out[pos++] = b[ib++];
        else { ++ia; ++ib; }  // skip common
    }
    if (ia < na) {
        std::memcpy(out + pos, a + ia, (na - ia) * sizeof(uint16_t));
        pos += na - ia;
    }
    if (ib < nb) {
        std::memcpy(out + pos, b + ib, (nb - ib) * sizeof(uint16_t));
        pos += nb - ib;
    }
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════
// Sorted array difference (Gap 2): emit elements in A but not B.
// ═══════════════════════════════════════════════════════════════════════════

inline uint32_t array_diff(const uint16_t* __restrict__ a, uint32_t na,
                           const uint16_t* __restrict__ b, uint32_t nb,
                           uint16_t* __restrict__ out) {
    uint32_t pos = 0;
    uint32_t ia = 0, ib = 0;

    while (ia < na && ib < nb) {
        // Bulk-copy from A when next 8+ elements are all < B[ib].
        if (ia + 8 <= na && a[ia + 7] < b[ib]) {
            std::memcpy(out + pos, a + ia, 8 * sizeof(uint16_t));
            pos += 8;
            ia += 8;
            continue;
        }
        // Skip B when next 8+ elements are all < A[ia].
        if (ib + 8 <= nb && b[ib + 7] < a[ia]) {
            ib += 8;
            continue;
        }

        if (a[ia] < b[ib]) out[pos++] = a[ia++];
        else if (a[ia] > b[ib]) ++ib;
        else { ++ia; ++ib; }
    }
    if (ia < na) {
        std::memcpy(out + pos, a + ia, (na - ia) * sizeof(uint16_t));
        pos += na - ia;
    }
    return pos;
}

}  // namespace arrow::simd
