#pragma once

#include <cstddef>
#include <cstdint>
#include "scalar_quantizer.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace hnsw::impl {

// ============================================================================
// Scalar fallback SQ distance kernels
// ============================================================================

/// Asymmetric L2 distance: float32 query vs uint8 quantized vector (scalar).
/// Uses query transform to avoid per-element scale+offset multiply.
inline float sq_l2_scalar(
    const float* query,
    const uint8_t* quantized,
    float scale, float offset,
    std::size_t dim) {
    const float invScale = 1.0f / scale;
    const float scale2 = scale * scale;
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float qp = (query[i] - offset) * invScale;
        float diff = qp - quantized[i];
        sum += diff * diff;
    }
    return sum * scale2;
}

/// Asymmetric IP distance: float32 query vs uint8 quantized vector (scalar).
inline float sq_ip_scalar(
    const float* query,
    const uint8_t* quantized,
    float scale, float offset,
    std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float deq = quantized[i] * scale + offset;
        sum += query[i] * deq;
    }
    return 1.0f - sum;
}

/// Batch asymmetric L2: float32 query vs N uint8 vectors (scalar).
inline void sq_l2_batch_scalar(
    const float* query,
    const uint8_t* const* targets,
    const float* scales,
    const float* offsets,
    std::size_t numTargets,
    std::size_t dim,
    float* outDistances) {
    for (std::size_t t = 0; t < numTargets; ++t) {
        outDistances[t] = sq_l2_scalar(query, targets[t], scales[t], offsets[t], dim);
    }
}

/// Batch asymmetric IP: float32 query vs N uint8 vectors (scalar).
inline void sq_ip_batch_scalar(
    const float* query,
    const uint8_t* const* targets,
    const float* scales,
    const float* offsets,
    std::size_t numTargets,
    std::size_t dim,
    float* outDistances) {
    for (std::size_t t = 0; t < numTargets; ++t) {
        outDistances[t] = sq_ip_scalar(query, targets[t], scales[t], offsets[t], dim);
    }
}

// ============================================================================
// NEON (ARM64 / Apple Silicon) SQ distance kernels
// ============================================================================

#if defined(__aarch64__) || defined(_M_ARM64)

/// Helper: widen 4 uint8 values (from low half of uint8x8) to float32x4.
inline float32x4_t widen_u8x4_to_f32(uint8x8_t raw) {
    uint16x8_t u16 = vmovl_u8(raw);              // u8x8 → u16x8
    uint32x4_t u32 = vmovl_u16(vget_low_u16(u16)); // low 4 → u32x4
    return vcvtq_f32_u32(u32);                    // u32x4 → f32x4
}

/// Helper: widen high 4 uint8 values (from high half of uint8x8) to float32x4.
inline float32x4_t widen_u8x4_hi_to_f32(uint8x8_t raw) {
    uint16x8_t u16 = vmovl_u8(raw);
    uint32x4_t u32 = vmovl_u16(vget_high_u16(u16));
    return vcvtq_f32_u32(u32);
}

/// Asymmetric L2: float32 query vs uint8 quantized vector (NEON).
/// Uses precomputed query transform to avoid per-element dequantization.
///
/// Math: L2(q, scale*x+offset) = scale²·Σ(x - q')² where q' = (q - offset)/scale
/// This lets us work in integer-friendly domain: accumulate (x-q')² in float
/// but with q' precomputed once per query, avoiding scale*x+offset per element.
inline float sq_l2_neon(
    const float* query,
    const uint8_t* quantized,
    float scale, float offset,
    std::size_t dim) {

    // Precompute transformed query: q' = (q - offset) / scale
    // Then L2 = scale² * Σ(q'_i - x_i)²
    const float invScale = 1.0f / scale;
    const float scale2 = scale * scale;

    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t vInvScale = vdupq_n_f32(invScale);
    float32x4_t vOffset = vdupq_n_f32(offset);

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        // Load 16 uint8 values
        uint8x16_t raw = vld1q_u8(quantized + i);
        uint8x8_t lo8 = vget_low_u8(raw);
        uint8x8_t hi8 = vget_high_u8(raw);

        // Widen uint8 → float32
        float32x4_t x0 = widen_u8x4_to_f32(lo8);
        float32x4_t x1 = widen_u8x4_hi_to_f32(lo8);
        float32x4_t x2 = widen_u8x4_to_f32(hi8);
        float32x4_t x3 = widen_u8x4_hi_to_f32(hi8);

        // Transform query: q' = (q - offset) * invScale
        float32x4_t q0 = vmulq_f32(vsubq_f32(vld1q_f32(query + i), vOffset), vInvScale);
        float32x4_t q1 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 4), vOffset), vInvScale);
        float32x4_t q2 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 8), vOffset), vInvScale);
        float32x4_t q3 = vmulq_f32(vsubq_f32(vld1q_f32(query + i + 12), vOffset), vInvScale);

        // Diff in quantized domain
        float32x4_t d0 = vsubq_f32(q0, x0);
        float32x4_t d1 = vsubq_f32(q1, x1);
        float32x4_t d2 = vsubq_f32(q2, x2);
        float32x4_t d3 = vsubq_f32(q3, x3);

        sum1 = vfmaq_f32(sum1, d0, d0);
        sum2 = vfmaq_f32(sum2, d1, d1);
        sum1 = vfmaq_f32(sum1, d2, d2);
        sum2 = vfmaq_f32(sum2, d3, d3);
    }

    float32x4_t sum = vaddq_f32(sum1, sum2);
    float result = vaddvq_f32(sum);

    // Residual
    for (; i < dim; ++i) {
        float qp = (query[i] - offset) * invScale;
        float diff = qp - quantized[i];
        result += diff * diff;
    }
    return result * scale2;
}

/// Asymmetric IP: float32 query vs uint8 quantized vector (NEON).
inline float sq_ip_neon(
    const float* query,
    const uint8_t* quantized,
    float scale, float offset,
    std::size_t dim) {

    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t voffset = vdupq_n_f32(offset);

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        uint8x16_t raw = vld1q_u8(quantized + i);
        uint8x8_t lo8 = vget_low_u8(raw);
        uint8x8_t hi8 = vget_high_u8(raw);

        float32x4_t f0 = vfmaq_f32(voffset, widen_u8x4_to_f32(lo8), vscale);
        float32x4_t f1 = vfmaq_f32(voffset, widen_u8x4_hi_to_f32(lo8), vscale);
        float32x4_t f2 = vfmaq_f32(voffset, widen_u8x4_to_f32(hi8), vscale);
        float32x4_t f3 = vfmaq_f32(voffset, widen_u8x4_hi_to_f32(hi8), vscale);

        sum1 = vfmaq_f32(sum1, vld1q_f32(query + i), f0);
        sum2 = vfmaq_f32(sum2, vld1q_f32(query + i + 4), f1);
        sum1 = vfmaq_f32(sum1, vld1q_f32(query + i + 8), f2);
        sum2 = vfmaq_f32(sum2, vld1q_f32(query + i + 12), f3);
    }

    float32x4_t sum = vaddq_f32(sum1, sum2);
    float result = vaddvq_f32(sum);

    for (; i < dim; ++i) {
        float deq = quantized[i] * scale + offset;
        result += query[i] * deq;
    }
    return 1.0f - result;
}

/// Batch asymmetric L2: float32 query vs N uint8 vectors (NEON).
/// Uses per-target query transform to work in quantized domain.
inline void sq_l2_batch_neon(
    const float* query,
    const uint8_t* const* targets,
    const float* scales,
    const float* offsets,
    std::size_t numTargets,
    std::size_t dim,
    float* outDistances) {

    std::size_t t = 0;
    for (; t + 4 <= numTargets; t += 4) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        const uint8_t* t0 = targets[t];
        const uint8_t* t1 = targets[t + 1];
        const uint8_t* t2 = targets[t + 2];
        const uint8_t* t3 = targets[t + 3];

        // Precompute invScale for each target
        float invScale0 = 1.0f / scales[t];
        float invScale1 = 1.0f / scales[t + 1];
        float invScale2 = 1.0f / scales[t + 2];
        float invScale3 = 1.0f / scales[t + 3];

        float32x4_t vInvS0 = vdupq_n_f32(invScale0);
        float32x4_t vOff0  = vdupq_n_f32(offsets[t]);
        float32x4_t vInvS1 = vdupq_n_f32(invScale1);
        float32x4_t vOff1  = vdupq_n_f32(offsets[t + 1]);
        float32x4_t vInvS2 = vdupq_n_f32(invScale2);
        float32x4_t vOff2  = vdupq_n_f32(offsets[t + 2]);
        float32x4_t vInvS3 = vdupq_n_f32(invScale3);
        float32x4_t vOff3  = vdupq_n_f32(offsets[t + 3]);

        std::size_t i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t q = vld1q_f32(query + i);

            // Target 0: q' = (q - offset) * invScale, diff = q' - x
            float32x4_t qp0 = vmulq_f32(vsubq_f32(q, vOff0), vInvS0);
            float32x4_t d0 = vsubq_f32(qp0, widen_u8x4_to_f32(vld1_u8(t0 + i)));
            sum0 = vfmaq_f32(sum0, d0, d0);

            float32x4_t qp1 = vmulq_f32(vsubq_f32(q, vOff1), vInvS1);
            float32x4_t d1 = vsubq_f32(qp1, widen_u8x4_to_f32(vld1_u8(t1 + i)));
            sum1 = vfmaq_f32(sum1, d1, d1);

            float32x4_t qp2 = vmulq_f32(vsubq_f32(q, vOff2), vInvS2);
            float32x4_t d2 = vsubq_f32(qp2, widen_u8x4_to_f32(vld1_u8(t2 + i)));
            sum2 = vfmaq_f32(sum2, d2, d2);

            float32x4_t qp3 = vmulq_f32(vsubq_f32(q, vOff3), vInvS3);
            float32x4_t d3 = vsubq_f32(qp3, widen_u8x4_to_f32(vld1_u8(t3 + i)));
            sum3 = vfmaq_f32(sum3, d3, d3);
        }

        outDistances[t]     = vaddvq_f32(sum0) * scales[t] * scales[t];
        outDistances[t + 1] = vaddvq_f32(sum1) * scales[t + 1] * scales[t + 1];
        outDistances[t + 2] = vaddvq_f32(sum2) * scales[t + 2] * scales[t + 2];
        outDistances[t + 3] = vaddvq_f32(sum3) * scales[t + 3] * scales[t + 3];

        // Residual dims
        for (; i < dim; ++i) {
            float qv = query[i];
            float qp0 = (qv - offsets[t]) * invScale0;
            float qp1 = (qv - offsets[t + 1]) * invScale1;
            float qp2 = (qv - offsets[t + 2]) * invScale2;
            float qp3 = (qv - offsets[t + 3]) * invScale3;
            outDistances[t]     += (qp0 - t0[i]) * (qp0 - t0[i]) * scales[t] * scales[t];
            outDistances[t + 1] += (qp1 - t1[i]) * (qp1 - t1[i]) * scales[t + 1] * scales[t + 1];
            outDistances[t + 2] += (qp2 - t2[i]) * (qp2 - t2[i]) * scales[t + 2] * scales[t + 2];
            outDistances[t + 3] += (qp3 - t3[i]) * (qp3 - t3[i]) * scales[t + 3] * scales[t + 3];
        }
    }

    for (; t < numTargets; ++t) {
        outDistances[t] = sq_l2_neon(query, targets[t], scales[t], offsets[t], dim);
    }
}

/// Batch asymmetric IP: float32 query vs N uint8 vectors (NEON).
inline void sq_ip_batch_neon(
    const float* query,
    const uint8_t* const* targets,
    const float* scales,
    const float* offsets,
    std::size_t numTargets,
    std::size_t dim,
    float* outDistances) {

    std::size_t t = 0;
    for (; t + 4 <= numTargets; t += 4) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        const uint8_t* t0 = targets[t];
        const uint8_t* t1 = targets[t + 1];
        const uint8_t* t2 = targets[t + 2];
        const uint8_t* t3 = targets[t + 3];

        float32x4_t vs0 = vdupq_n_f32(scales[t]);
        float32x4_t vo0 = vdupq_n_f32(offsets[t]);
        float32x4_t vs1 = vdupq_n_f32(scales[t + 1]);
        float32x4_t vo1 = vdupq_n_f32(offsets[t + 1]);
        float32x4_t vs2 = vdupq_n_f32(scales[t + 2]);
        float32x4_t vo2 = vdupq_n_f32(offsets[t + 2]);
        float32x4_t vs3 = vdupq_n_f32(scales[t + 3]);
        float32x4_t vo3 = vdupq_n_f32(offsets[t + 3]);

        std::size_t i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t q = vld1q_f32(query + i);

            sum0 = vfmaq_f32(sum0, q, vfmaq_f32(vo0, widen_u8x4_to_f32(vld1_u8(t0 + i)), vs0));
            sum1 = vfmaq_f32(sum1, q, vfmaq_f32(vo1, widen_u8x4_to_f32(vld1_u8(t1 + i)), vs1));
            sum2 = vfmaq_f32(sum2, q, vfmaq_f32(vo2, widen_u8x4_to_f32(vld1_u8(t2 + i)), vs2));
            sum3 = vfmaq_f32(sum3, q, vfmaq_f32(vo3, widen_u8x4_to_f32(vld1_u8(t3 + i)), vs3));
        }

        outDistances[t]     = vaddvq_f32(sum0);
        outDistances[t + 1] = vaddvq_f32(sum1);
        outDistances[t + 2] = vaddvq_f32(sum2);
        outDistances[t + 3] = vaddvq_f32(sum3);

        for (; i < dim; ++i) {
            float qv = query[i];
            outDistances[t]     += qv * (t0[i] * scales[t] + offsets[t]);
            outDistances[t + 1] += qv * (t1[i] * scales[t + 1] + offsets[t + 1]);
            outDistances[t + 2] += qv * (t2[i] * scales[t + 2] + offsets[t + 2]);
            outDistances[t + 3] += qv * (t3[i] * scales[t + 3] + offsets[t + 3]);
        }

        // Convert similarity to distance
        outDistances[t]     = 1.0f - outDistances[t];
        outDistances[t + 1] = 1.0f - outDistances[t + 1];
        outDistances[t + 2] = 1.0f - outDistances[t + 2];
        outDistances[t + 3] = 1.0f - outDistances[t + 3];
    }

    for (; t < numTargets; ++t) {
        outDistances[t] = sq_ip_neon(query, targets[t], scales[t], offsets[t], dim);
    }
}
// ============================================================================
// Integer-domain L2 distance kernels (for global quantization)
// Both query and target are uint8 — no float math in the hot loop.
// ============================================================================

/// Integer L2: uint8 query × uint8 target → Σ(q-t)² as uint32 (NEON).
/// Uses vabdl for widening absolute difference, then multiply-accumulate.
inline uint32_t sq_int_l2_neon(
    const uint8_t* query,
    const uint8_t* target,
    std::size_t dim) {

    uint32x4_t sum1 = vdupq_n_u32(0);
    uint32x4_t sum2 = vdupq_n_u32(0);

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        uint8x16_t q = vld1q_u8(query + i);
        uint8x16_t t = vld1q_u8(target + i);

        // Absolute difference, widened to uint16
        uint16x8_t diff_lo = vabdl_u8(vget_low_u8(q), vget_low_u8(t));
        uint16x8_t diff_hi = vabdl_high_u8(q, t);

        // Square and accumulate into uint32
        sum1 = vmlal_u16(sum1, vget_low_u16(diff_lo), vget_low_u16(diff_lo));
        sum2 = vmlal_u16(sum2, vget_high_u16(diff_lo), vget_high_u16(diff_lo));
        sum1 = vmlal_u16(sum1, vget_low_u16(diff_hi), vget_low_u16(diff_hi));
        sum2 = vmlal_u16(sum2, vget_high_u16(diff_hi), vget_high_u16(diff_hi));
    }

    uint32_t result = vaddvq_u32(vaddq_u32(sum1, sum2));

    // Residual
    for (; i < dim; ++i) {
        int diff = static_cast<int>(query[i]) - static_cast<int>(target[i]);
        result += static_cast<uint32_t>(diff * diff);
    }
    return result;
}

/// Integer batch L2: uint8 query × N uint8 targets → N uint32 distances (NEON).
inline void sq_int_l2_batch_neon(
    const uint8_t* query,
    const uint8_t* const* targets,
    std::size_t numTargets,
    std::size_t dim,
    uint32_t* outDistances) {

    std::size_t t = 0;
    for (; t + 4 <= numTargets; t += 4) {
        uint32x4_t s0 = vdupq_n_u32(0);
        uint32x4_t s1 = vdupq_n_u32(0);
        uint32x4_t s2 = vdupq_n_u32(0);
        uint32x4_t s3 = vdupq_n_u32(0);

        const uint8_t* t0 = targets[t];
        const uint8_t* t1 = targets[t + 1];
        const uint8_t* t2 = targets[t + 2];
        const uint8_t* t3 = targets[t + 3];

        std::size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            uint8x16_t q = vld1q_u8(query + i);

            // Target 0
            uint16x8_t d0_lo = vabdl_u8(vget_low_u8(q), vget_low_u8(vld1q_u8(t0 + i)));
            uint16x8_t d0_hi = vabdl_high_u8(q, vld1q_u8(t0 + i));
            s0 = vmlal_u16(s0, vget_low_u16(d0_lo), vget_low_u16(d0_lo));
            s0 = vmlal_u16(s0, vget_high_u16(d0_lo), vget_high_u16(d0_lo));
            s0 = vmlal_u16(s0, vget_low_u16(d0_hi), vget_low_u16(d0_hi));
            s0 = vmlal_u16(s0, vget_high_u16(d0_hi), vget_high_u16(d0_hi));

            // Target 1
            uint16x8_t d1_lo = vabdl_u8(vget_low_u8(q), vget_low_u8(vld1q_u8(t1 + i)));
            uint16x8_t d1_hi = vabdl_high_u8(q, vld1q_u8(t1 + i));
            s1 = vmlal_u16(s1, vget_low_u16(d1_lo), vget_low_u16(d1_lo));
            s1 = vmlal_u16(s1, vget_high_u16(d1_lo), vget_high_u16(d1_lo));
            s1 = vmlal_u16(s1, vget_low_u16(d1_hi), vget_low_u16(d1_hi));
            s1 = vmlal_u16(s1, vget_high_u16(d1_hi), vget_high_u16(d1_hi));

            // Target 2
            uint16x8_t d2_lo = vabdl_u8(vget_low_u8(q), vget_low_u8(vld1q_u8(t2 + i)));
            uint16x8_t d2_hi = vabdl_high_u8(q, vld1q_u8(t2 + i));
            s2 = vmlal_u16(s2, vget_low_u16(d2_lo), vget_low_u16(d2_lo));
            s2 = vmlal_u16(s2, vget_high_u16(d2_lo), vget_high_u16(d2_lo));
            s2 = vmlal_u16(s2, vget_low_u16(d2_hi), vget_low_u16(d2_hi));
            s2 = vmlal_u16(s2, vget_high_u16(d2_hi), vget_high_u16(d2_hi));

            // Target 3
            uint16x8_t d3_lo = vabdl_u8(vget_low_u8(q), vget_low_u8(vld1q_u8(t3 + i)));
            uint16x8_t d3_hi = vabdl_high_u8(q, vld1q_u8(t3 + i));
            s3 = vmlal_u16(s3, vget_low_u16(d3_lo), vget_low_u16(d3_lo));
            s3 = vmlal_u16(s3, vget_high_u16(d3_lo), vget_high_u16(d3_lo));
            s3 = vmlal_u16(s3, vget_low_u16(d3_hi), vget_low_u16(d3_hi));
            s3 = vmlal_u16(s3, vget_high_u16(d3_hi), vget_high_u16(d3_hi));
        }

        outDistances[t]     = vaddvq_u32(s0);
        outDistances[t + 1] = vaddvq_u32(s1);
        outDistances[t + 2] = vaddvq_u32(s2);
        outDistances[t + 3] = vaddvq_u32(s3);

        // Residual
        for (; i < dim; ++i) {
            int q_val = query[i];
            int d0 = q_val - t0[i]; outDistances[t]     += d0 * d0;
            int d1 = q_val - t1[i]; outDistances[t + 1] += d1 * d1;
            int d2 = q_val - t2[i]; outDistances[t + 2] += d2 * d2;
            int d3 = q_val - t3[i]; outDistances[t + 3] += d3 * d3;
        }
    }

    for (; t < numTargets; ++t) {
        outDistances[t] = sq_int_l2_neon(query, targets[t], dim);
    }
}
#endif

// Scalar fallback for integer L2
inline uint32_t sq_int_l2_scalar(
    const uint8_t* query,
    const uint8_t* target,
    std::size_t dim) {
    uint32_t sum = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        int diff = static_cast<int>(query[i]) - static_cast<int>(target[i]);
        sum += static_cast<uint32_t>(diff * diff);
    }
    return sum;
}

inline void sq_int_l2_batch_scalar(
    const uint8_t* query,
    const uint8_t* const* targets,
    std::size_t numTargets,
    std::size_t dim,
    uint32_t* outDistances) {
    for (std::size_t t = 0; t < numTargets; ++t) {
        outDistances[t] = sq_int_l2_scalar(query, targets[t], dim);
    }
}

} // namespace hnsw::impl
