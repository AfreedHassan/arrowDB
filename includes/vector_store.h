#ifndef VECTOR_STORE_H
#define VECTOR_STORE_H

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace arrow {

class VectorStore {
private:
    size_t dim_;
    std::vector<float> vectors_; 
    std::vector<uint64_t> ids_;

public:
    explicit VectorStore(size_t dimension)
        : dim_(dimension) {}

    // internal index
    size_t insert(uint64_t id, const std::vector<float>& vec) {

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

    size_t size() const {
        return ids_.size();
    }

    const float* vecAt(size_t index) const {
        return &vectors_[index * dim_];
    }

    uint64_t vecIdAt(size_t index) const {
        return ids_[index];
    }

    size_t dimension() const {
        return dim_;
    }
};

} // namespace arrow
	
#endif // VECTOR_STORE_H
