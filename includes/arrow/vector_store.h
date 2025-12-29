#ifndef VECTOR_STORE_H
#define VECTOR_STORE_H

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace arrow {
	using VectorID = uint64_t;  ///< Type alias for vector identifiers

/**
 * @brief Storage for high-dimensional vectors with associated IDs.
 * 
 * VectorStore maintains a collection of vectors stored as a contiguous
 * array of floats, along with their associated unique identifiers.
 * Vectors are stored in a flat array format for efficient memory access.
 */
class VectorStore {
private:
    size_t dim_;
    std::vector<float> vectors_; 
    std::vector<VectorID> ids_;

public:
    /**
     * @brief Constructs a VectorStore with the specified dimension.
     * @param dimension The dimension of all vectors that will be stored.
     */
    explicit VectorStore(size_t dimension)
        : dim_(dimension) {}

    /**
     * @brief Inserts a vector and its associated ID into the store.
     * @param id The unique identifier for the vector.
     * @param vec The vector to be inserted. Must match the store's dimension.
     * @return The internal index at which the vector was inserted.
     * @throws std::invalid_argument if the dimension of vec does not match dim_.
     */
    size_t insert(VectorID id, const std::vector<float>& vec) {

        if (vec.size() != dim_) {
            throw std::invalid_argument("dimension mismatch");
        }

				const size_t index = ids_.size();
				ids_.push_back(id);

				const size_t oldSize = vectors_.size();
				vectors_.resize(oldSize + dim_);

				std::memcpy(
						vectors_.data() + oldSize,
						vec.data(),
						dim_ * sizeof(float)
						);

				return index;
    }

    /**
     * @brief Returns the number of vectors stored.
     * @return The number of vectors currently in the store.
     */
    size_t size() const {
        return ids_.size();
    }

    /**
     * @brief Gets a pointer to the vector at the specified index.
     * @param index The internal index of the vector.
     * @return A pointer to the first element of the vector at index.
     * @note The returned pointer points to dim_ consecutive floats.
     */
    const float* vecAt(size_t index) const {
        return &vectors_[index * dim_];
    }

    /**
     * @brief Gets the ID associated with the vector at the specified index.
     * @param index The internal index of the vector.
     * @return The unique identifier of the vector at index.
     */
    VectorID vecIdAt(size_t index) const {
        return ids_[index];
    }

    /**
     * @brief Returns the dimension of vectors stored in this store.
     * @return The dimension of all vectors in the store.
     */
    size_t dimension() const {
        return dim_;
    }
};

} // namespace arrow
	
#endif // VECTOR_STORE_H

