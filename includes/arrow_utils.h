#ifndef ARROW_UTILS_H
#define ARROW_UTILS_H

#include <vector>
#include <cmath>
#include <stdexcept>

namespace arrow {

inline void validateDimension(
    const std::vector<float>& vec,
    size_t expectedDims
) {
    if (vec.size() != expectedDims) {
        throw std::invalid_argument(
            "vector dimension mismatch"
        );
    }
}

inline void normalizeL2(std::vector<float>& vec) {
    float normSq = 0.0f;

    for (float v : vec) {
        normSq += v * v;
    }

    if (normSq == 0.0f) {
        throw std::invalid_argument("zero vector cannot be normalized");
    }

    float invNorm = 1.0f / std::sqrt(normSq);
    for (float& v : vec) {
        v *= invNorm;
    }
}

inline std::vector<float> validateAndNormalize(
		const std::vector<float>& vec,
		size_t expectedDims
		) {
	validateDimension(vec, expectedDims);

	std::vector<float> out = vec;
	normalizeL2(out);
	return out;
}

} // namespace arrow

#endif // ARROW_UTILS_H	
