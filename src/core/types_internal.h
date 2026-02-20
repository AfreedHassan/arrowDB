#ifndef ARROW_TYPES_INTERNAL_H
#define ARROW_TYPES_INTERNAL_H

#include <cstdint>

namespace arrow {

using InternalID = uint64_t;

/// Data types for vector storage (internal use).
enum class DataType {
    Int32,
    Float32
};

} // namespace arrow

#endif // ARROW_TYPES_INTERNAL_H
