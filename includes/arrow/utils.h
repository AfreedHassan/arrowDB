#ifndef ARROW_UTILS_H
#define ARROW_UTILS_H

#include <vector>
#include <cmath>
#include <stdexcept>

namespace arrow {

/**
 * @brief Validates that a vector has the expected dimension.
 * @param vec The vector to validate.
 * @param expectedDims The expected dimension.
 * @throws std::invalid_argument if vec.size() != expectedDims.
 */
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

/**
 * @brief Normalizes a vector to unit length using L2 norm.
 * 
 * Modifies the input vector in-place to have unit length.
 * 
 * @param vec The vector to normalize (modified in-place).
 * @throws std::invalid_argument if vec is a zero vector.
 */
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

/**
 * @brief Validates vector dimension and normalizes it to unit length.
 * 
 * This is a convenience function that combines validateDimension and normalizeL2.
 * Returns a new normalized vector without modifying the input.
 * 
 * @param vec The vector to validate and normalize.
 * @param expectedDims The expected dimension.
 * @return A new vector that is normalized to unit length.
 * @throws std::invalid_argument if vec.size() != expectedDims or vec is zero.
 */
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

