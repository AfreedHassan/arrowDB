#ifndef TESTS_TEST_UTIL_H_
#define TESTS_TEST_UTIL_H_

#include <vector>
#include <random>
#include <cmath>

namespace arrow {
namespace testing {

// Helper: generate random normalized vector for testing
inline std::vector<float> RandomVector(size_t dim, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    for (auto& v : vec) v /= norm;
    
    return vec;
}

}  // namespace testing
}  // namespace arrow

#endif  // TESTS_TEST_UTIL_H_

