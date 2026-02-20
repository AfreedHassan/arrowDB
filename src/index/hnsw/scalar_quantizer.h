#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace hnsw {

/// Per-vector quantization parameters for scalar quantization.
struct SQVectorMeta {
    float scale;    // (max - min) / 255
    float offset;   // min value
};

/// SQ distance function: float32 query × uint8 quantized vector → distance.
using psq_distfunc_t = float (*)(
    const float* query,
    const uint8_t* quantized,
    float scale, float offset,
    std::size_t dim);

/// SQ batch distance function: float32 query × N uint8 vectors → N distances.
using psq_batchdistfunc_t = void (*)(
    const float* query,
    const uint8_t* const* targets,
    const float* scales,
    const float* offsets,
    std::size_t numTargets,
    std::size_t dim,
    float* outDistances);

/// Distance type selector for SQ kernel selection.
enum class SQDistType : uint8_t { L2, IP };

/// Global quantization mode: one scale/offset for entire dataset.
/// Enables pure uint8-vs-uint8 integer distance kernels.
enum class SQMode : uint8_t {
    PerVector,  // Per-vector min/max (default, better accuracy)
    Global      // Global min/max (faster: integer-domain kernels)
};

/// Scalar quantizer: maps float32 vectors to uint8 [0,255].
class ScalarQuantizer {
    std::size_t dim_;

public:
    explicit ScalarQuantizer(std::size_t dim) : dim_(dim) {}

    /// Quantize a single float32 vector to uint8, computing per-vector params.
    void quantize(const float* input, uint8_t* output, SQVectorMeta& params) const {
        float minVal = input[0];
        float maxVal = input[0];
        for (std::size_t i = 1; i < dim_; ++i) {
            minVal = std::min(minVal, input[i]);
            maxVal = std::max(maxVal, input[i]);
        }

        float range = maxVal - minVal;
        if (range < 1e-10f) range = 1e-10f;

        params.scale = range / 255.0f;
        params.offset = minVal;
        float invScale = 255.0f / range;

        for (std::size_t i = 0; i < dim_; ++i) {
            float normalized = (input[i] - minVal) * invScale;
            int val = static_cast<int>(normalized + 0.5f);
            output[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
        }
    }

    /// Quantize using pre-computed global params.
    void quantizeWithParams(const float* input, uint8_t* output,
                            float scale, float offset) const {
        float invScale = 1.0f / scale;
        for (std::size_t i = 0; i < dim_; ++i) {
            float normalized = (input[i] - offset) * invScale;
            int val = static_cast<int>(normalized + 0.5f);
            output[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
        }
    }

    /// Dequantize a uint8 vector back to float32.
    void dequantize(const uint8_t* input, float* output, const SQVectorMeta& params) const {
        for (std::size_t i = 0; i < dim_; ++i) {
            output[i] = input[i] * params.scale + params.offset;
        }
    }

    std::size_t dim() const { return dim_; }
};

/// Integer-domain distance: uint8 query × uint8 target → uint32 distance.
/// Used with global quantization (no per-element float math).
using psq_int_distfunc_t = uint32_t (*)(
    const uint8_t* query,
    const uint8_t* target,
    std::size_t dim);

/// Integer-domain batch distance: uint8 query × N uint8 targets → N uint32 distances.
using psq_int_batchdistfunc_t = void (*)(
    const uint8_t* query,
    const uint8_t* const* targets,
    std::size_t numTargets,
    std::size_t dim,
    uint32_t* outDistances);

} // namespace hnsw
