#pragma once

#include "lib.h"
#include "backend_interface.h"
#include "backend_registry.h"

namespace hnsw {

/// Inner Product Space - measures similarity via dot product.
/// Higher values indicate more similar vectors (maximize similarity).
class InnerProductSpace : public SpaceInterface<float> {
private:
    const DistanceBackend<float>& backend_;
    std::size_t dim_;
    std::size_t data_size_;
    pdistfunc_t<float> pdistfunc_;  // Cached function pointer for optimal performance

public:
    /// Construct Inner Product space for vectors of given dimension.
    explicit InnerProductSpace(std::size_t dim)
        : backend_(selectBackend<float>()), dim_(dim),
          data_size_(dim * sizeof(float)) {
        // Select optimal kernel at construction time based on dimension alignment
        // This eliminates branch in hot path of distance calculations
        if (dim_ % 16 == 0) {
            pdistfunc_ = backend_.ip_aligned16;
        } else if (dim_ % 4 == 0) {
            pdistfunc_ = backend_.ip_aligned4;
        } else {
            pdistfunc_ = backend_.ip;  // default with residual handling
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

    ~InnerProductSpace() = default;
};

}  // namespace hnsw
