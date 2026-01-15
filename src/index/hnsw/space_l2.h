#pragma once

#include "lib.h"
#include "backend_interface.h"
#include "backend_registry.h"

namespace hnsw {

/// L2 (Euclidean) Space - measures distance via squared L2 norm.
/// Lower values indicate closer vectors (minimize distance).
class L2Space : public SpaceInterface<float> {
private:
    const DistanceBackend<float>& backend_;
    std::size_t dim_;
    std::size_t data_size_;
    pdistfunc_t<float> pdistfunc_;  // Cached function pointer for optimal performance

public:
    /// Construct L2 space for vectors of given dimension.
    explicit L2Space(std::size_t dim)
        : backend_(selectBackend<float>()), dim_(dim),
          data_size_(dim * sizeof(float)) {
        // Select optimal kernel at construction time based on dimension alignment
        // This eliminates branch in hot path of distance calculations
        if (dim_ % 16 == 0) {
            pdistfunc_ = backend_.l2_aligned16;
        } else if (dim_ % 4 == 0) {
            pdistfunc_ = backend_.l2_aligned4;
        } else {
            pdistfunc_ = backend_.l2;  // default with residual handling
        }
    }

    /// Get size of vector data in bytes.
    size_t getDataSize() override {
        return data_size_;
    }

    /// Get distance function pointer (adapts void* to typed interface).
    pdistfunc_t<float> getDistFunc() override {
        return pdistfunc_;
    }

    /// Get distance function parameters (dimension).
    void* getDistFuncParam() override {
        return &dim_;
    }

    /// Direct typed interface (convenience method, not part of SpaceInterface).
    float distance(const float* a, const float* b) const {
        return pdistfunc_(a, b, dim_);
    }

    /// Get dimension of vectors in this space.
    std::size_t dim() const { return dim_; }

    /// Get name of selected SIMD backend (for debugging/info).
    const char* backend_name() const { return backend_.name; }

    ~L2Space() = default;
};

}  // namespace hnsw
