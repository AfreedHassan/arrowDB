// Copyright 2025 ArrowDB
#ifndef ARROW_OPTIONS_H
#define ARROW_OPTIONS_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "types.h"

namespace arrow {

/// Configuration for creating a new collection.
struct CollectionConfig {
    std::string name;                              ///< Collection name
    uint32_t dimensions;                           ///< Vector dimension
    DistanceMetric metric = DistanceMetric::Cosine; ///< Distance metric for similarity
};

/// Configuration for the HNSW index.
///
/// Default values optimized for 100K+ vectors based on benchmark results:
/// - M=64: Provides 91-92% recall@10 for 100K vectors
/// - ef_construction=200: Balanced build time and quality
///
/// For smaller datasets (<10K), M=32 may be sufficient and uses less memory.
struct IndexOptions {
    size_t max_elements = 1000000;   ///< Initial capacity
    size_t M = 64;                   ///< Max connections per node
    size_t ef_construction = 200;    ///< Construction beam width
    size_t ef_search = 200;          ///< Default search beam width
};

/// Client options for initializing ArrowDB.
struct ClientOptions {
    std::filesystem::path data_dir;                ///< Directory for storing collections
    IndexOptions default_index_options;            ///< Default index config for new collections
    // Future: std::string server_address;         ///< For remote mode
    // Future: size_t connection_timeout_ms;       ///< Connection timeout
};

} // namespace arrow

#endif // ARROW_OPTIONS_H
