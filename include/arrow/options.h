// Copyright 2025 ArrowDB
#ifndef ARROW_OPTIONS_H
#define ARROW_OPTIONS_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "types.h"

namespace arrow {

/// HNSW-specific index parameters.
///
/// Default values tuned from benchmarks on ≤100K vectors:
/// - M=16: Optimal for ≤100K vectors (M=64 wastes memory for this range)
/// - ef_construction=200: Balanced build time and quality
/// - ef_search=200: Provides ~91% recall@10
struct HNSWParams {
    size_t M = 16;                  ///< Max connections per node
    size_t ef_construction = 200;   ///< Construction beam width
    size_t ef_search = 200;         ///< Default search beam width
};

/// Index configuration.
struct IndexConfig {
    IndexType index_type = IndexType::HNSW;
    size_t max_elements = 1000000;   ///< Initial capacity
    Quantization quantization = Quantization::None;
    HNSWParams hnsw_params;          ///< Active when index_type == HNSW
};

/// Configuration for creating a new collection.
struct CollectionConfig {
    std::string name;                              ///< Collection name
    uint32_t dimensions = 0;                       ///< Vector dimension
    Space space = Space::Cosine;                   ///< Index space
    IndexConfig index_config;                      ///< Index configuration
    MetadataSchema schema;                                 ///< Metadata schema (empty = no validation)
};

/// Client options for initializing ArrowDB.
struct ClientOptions {
    std::filesystem::path dataDir;                ///< Directory for storing collections
    IndexConfig defaultIndexConfig;            ///< Default index config for new collections
    // Future: std::string server_address;         ///< For remote mode
    // Future: size_t connection_timeout_ms;       ///< Connection timeout
};

} // namespace arrow

#endif // ARROW_OPTIONS_H
