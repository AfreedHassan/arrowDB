#pragma once

#include "lib.h"
#include "backend_interface.h"
#include "backend_registry.h"

namespace hnsw {

/// Interface for spaces that support multi-vector document storage.
/// Extends SpaceInterface to add document ID get/set operations.
/// Memory layout: [vector data][DocIdType]
template<typename DocIdType>
class MultiVectorSpaceInterface : public SpaceInterface<float> {
public:
    /// Extract document ID from a datapoint.
    [[nodiscard]] virtual DocIdType getDocId(const void* datapoint) const = 0;

    /// Set document ID in a datapoint.
    virtual void setDocId(void* datapoint, DocIdType docId) = 0;

    /// Get size of vector data only (excluding doc ID).
    [[nodiscard]] virtual std::size_t vectorSize() const = 0;

    virtual ~MultiVectorSpaceInterface() = default;
};

/// L2 (Euclidean) space with document ID storage.
/// Memory layout: [float[dim]][DocIdType]
/// Lower distance values indicate closer vectors.
template<typename DocIdType>
class MultiVectorL2Space : public MultiVectorSpaceInterface<DocIdType> {
private:
    const DistanceBackend<float>& backend_;
    std::size_t dim_;
    std::size_t vectorSize_;
    std::size_t dataSize_;
    pdistfunc_t<float> pdistfunc_;

public:
    /// Construct L2 space for vectors of given dimension.
    explicit MultiVectorL2Space(std::size_t dim)
        : backend_(selectBackend<float>())
        , dim_(dim)
        , vectorSize_(dim * sizeof(float))
        , dataSize_(vectorSize_ + sizeof(DocIdType)) {
        // Select optimal kernel based on dimension alignment
        // This eliminates branching in the hot path
        if (dim_ % 16 == 0) {
            pdistfunc_ = backend_.l2_aligned16;
        } else if (dim_ % 4 == 0) {
            pdistfunc_ = backend_.l2_aligned4;
        } else {
            pdistfunc_ = backend_.l2;
        }
    }

    /// Get total size of datapoint in bytes (vector + doc ID).
    std::size_t getDataSize() override {
        return dataSize_;
    }

    /// Get distance function pointer.
    pdistfunc_t<float> getDistFunc() override {
        return pdistfunc_;
    }

    /// Get distance function parameters (pointer to dimension).
    void* getDistFuncParam() override {
        return &dim_;
    }

    /// Get size of vector data only (excluding doc ID).
    [[nodiscard]] std::size_t vectorSize() const override {
        return vectorSize_;
    }

    /// Extract document ID from a datapoint.
    [[nodiscard]] DocIdType getDocId(const void* datapoint) const override {
        return *reinterpret_cast<const DocIdType*>(
            static_cast<const char*>(datapoint) + vectorSize_);
    }

    /// Set document ID in a datapoint.
    void setDocId(void* datapoint, DocIdType docId) override {
        *reinterpret_cast<DocIdType*>(
            static_cast<char*>(datapoint) + vectorSize_) = docId;
    }

    /// Compute L2 distance between two vectors (convenience method).
    [[nodiscard]] float distance(const float* a, const float* b) const {
        return pdistfunc_(a, b, dim_);
    }

    /// Get dimension of vectors in this space.
    [[nodiscard]] std::size_t dim() const { return dim_; }

    /// Get name of selected SIMD backend (for debugging/info).
    [[nodiscard]] const char* backendName() const { return backend_.name; }

    ~MultiVectorL2Space() = default;
};

/// Inner Product space with document ID storage.
/// Memory layout: [float[dim]][DocIdType]
/// Higher similarity values indicate more similar vectors.
template<typename DocIdType>
class MultiVectorInnerProductSpace : public MultiVectorSpaceInterface<DocIdType> {
private:
    const DistanceBackend<float>& backend_;
    std::size_t dim_;
    std::size_t vectorSize_;
    std::size_t dataSize_;
    pdistfunc_t<float> pdistfunc_;

public:
    /// Construct Inner Product space for vectors of given dimension.
    explicit MultiVectorInnerProductSpace(std::size_t dim)
        : backend_(selectBackend<float>())
        , dim_(dim)
        , vectorSize_(dim * sizeof(float))
        , dataSize_(vectorSize_ + sizeof(DocIdType)) {
        // Select optimal kernel based on dimension alignment
        // This eliminates branching in the hot path
        if (dim_ % 16 == 0) {
            pdistfunc_ = backend_.ip_aligned16;
        } else if (dim_ % 4 == 0) {
            pdistfunc_ = backend_.ip_aligned4;
        } else {
            pdistfunc_ = backend_.ip;
        }
    }

    /// Get total size of datapoint in bytes (vector + doc ID).
    std::size_t getDataSize() override {
        return dataSize_;
    }

    /// Get distance function pointer.
    pdistfunc_t<float> getDistFunc() override {
        return pdistfunc_;
    }

    /// Get distance function parameters (pointer to dimension).
    void* getDistFuncParam() override {
        return &dim_;
    }

    /// Get size of vector data only (excluding doc ID).
    [[nodiscard]] std::size_t vectorSize() const override {
        return vectorSize_;
    }

    /// Extract document ID from a datapoint.
    [[nodiscard]] DocIdType getDocId(const void* datapoint) const override {
        return *reinterpret_cast<const DocIdType*>(
            static_cast<const char*>(datapoint) + vectorSize_);
    }

    /// Set document ID in a datapoint.
    void setDocId(void* datapoint, DocIdType docId) override {
        *reinterpret_cast<DocIdType*>(
            static_cast<char*>(datapoint) + vectorSize_) = docId;
    }

    /// Compute inner product between two vectors (convenience method).
    [[nodiscard]] float distance(const float* a, const float* b) const {
        return pdistfunc_(a, b, dim_);
    }

    /// Get dimension of vectors in this space.
    [[nodiscard]] std::size_t dim() const { return dim_; }

    /// Get name of selected SIMD backend (for debugging/info).
    [[nodiscard]] const char* backendName() const { return backend_.name; }

    ~MultiVectorInnerProductSpace() = default;
};

}  // namespace hnsw
