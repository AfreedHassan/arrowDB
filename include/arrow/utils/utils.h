#ifndef ARROW_UTILS_H
#define ARROW_UTILS_H

#include "arrow/types.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <iostream>
#include <format>

/**
 * @brief Stream output operator for std::vector.
 * @tparam T The element type of the vector.
 * @param os The output stream.
 * @param vec The vector to output.
 * @return Reference to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
	os << std::format("{}", vec) << '\n';
	return os;
}
namespace arrow::utils {
/**
 * @brief Convert MetadataValue to JSON value.
 * 
 * Handles all variant types: int64_t, double, std::string, bool
 */
inline json metadataValueToJson(const MetadataValue& value) {
    return std::visit([](auto&& arg) -> json {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            return json(arg);
        } else if constexpr (std::is_same_v<T, double>) {
            return json(arg);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return json(arg);
        } else if constexpr (std::is_same_v<T, bool>) {
            return json(arg);
        }
    }, value);
}

/**
 * @brief Convert JSON value to MetadataValue.
 * 
 * @throws std::runtime_error if JSON type is not supported
 */
inline MetadataValue jsonToMetadataValue(const json& j) {
    if (j.is_number_integer()) {
        return j.get<int64_t>();
    } else if (j.is_number_float()) {
        return j.get<double>();
    } else if (j.is_string()) {
        return j.get<std::string>();
    } else if (j.is_boolean()) {
        return j.get<bool>();
    } else {
        throw std::runtime_error("Unsupported JSON type for MetadataValue");
    }
}

/**
 * @brief Convert Metadata map to JSON object.
 * 
 * Example output:
 * {
 *   "category": "image",
 *   "score": 0.95,
 *   "tags": "dog",
 *   "active": true
 * }
 */
inline json metadataToJson(const Metadata& metadata) {
    json j = json::object();
    for (const auto& [key, value] : metadata) {
        j[key] = metadataValueToJson(value);
    }
    return j;
}

/**
 * @brief Convert JSON object to Metadata map.
 */
inline Metadata jsonToMetadata(const json& j) {
    if (!j.is_object()) {
        throw std::runtime_error("Expected JSON object for Metadata");
    }
    
    Metadata metadata;
    for (const auto& [key, value] : j.items()) {
        metadata[key] = jsonToMetadataValue(value);
    }
    return metadata;
}

/**
 * @brief Convert DistanceMetric enum to JSON string.
 */
inline json distanceMetricToJson(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::Cosine:
            return "Cosine";
        case DistanceMetric::L2:
            return "L2";
        case DistanceMetric::InnerProduct:
            return "InnerProduct";
        default:
            throw std::runtime_error("Unknown DistanceMetric");
    }
}

/**
 * @brief Convert JSON string to DistanceMetric enum.
 */
inline DistanceMetric jsonToDistanceMetric(const json& j) {
    std::string metricStr = j.get<std::string>();
    if (metricStr == "Cosine") {
        return DistanceMetric::Cosine;
    } else if (metricStr == "L2") {
        return DistanceMetric::L2;
    } else if (metricStr == "InnerProduct") {
        return DistanceMetric::InnerProduct;
    } else {
        throw std::runtime_error("Unknown DistanceMetric: " + metricStr);
    }
}

/**
 * @brief Convert DataType enum to JSON string.
 */
inline json dataTypeToJson(DataType dtype) {
    switch (dtype) {
        case DataType::Int16:
            return "Int16";
        case DataType::Float16:
            return "Float16";
        default:
            throw std::runtime_error("Unknown DataType");
    }
}

/**
 * @brief Convert JSON string to DataType enum.
 */
inline DataType jsonToDataType(const json& j) {
    std::string dtypeStr = j.get<std::string>();
    if (dtypeStr == "Int16") {
        return DataType::Int16;
    } else if (dtypeStr == "Float16") {
        return DataType::Float16;
    } else {
        throw std::runtime_error("Unknown DataType: " + dtypeStr);
    }
}

/**
 * @brief Export a map of VectorID -> Metadata to JSON file.
 * 
 * Format:
 * {
 *   "1": {
 *     "category": "image",
 *     "score": 0.95,
 *     "active": true
 *   },
 *   "2": {
 *     "category": "text",
 *     "author": "John Doe"
 *   }
 * }
 * 
 * @param metadataMap Map of vector IDs to their metadata
 * @param filepath Path to output JSON file
 * @throws std::runtime_error if file cannot be written
 */
inline void exportMetadataToJson(
    const std::unordered_map<VectorID, Metadata>& metadataMap,
    const std::string& filepath
) {
    json j = json::object();
    for (const auto& [id, metadata] : metadataMap) {
        // Convert VectorID to string for JSON key
        j[std::to_string(id)] = metadataToJson(metadata);
    }
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    file << j.dump(2);  // Pretty print with 2-space indent
    file.close();
}

/**
 * @brief Import metadata from JSON file.
 * 
 * @param filepath Path to input JSON file
 * @return Map of VectorID -> Metadata
 * @throws std::runtime_error if file cannot be read or parsed
 */
inline std::unordered_map<VectorID, Metadata> importMetadataFromJson(
    const std::string& filepath
) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    json j;
    file >> j;
    file.close();
    
    if (!j.is_object()) {
        throw std::runtime_error("Expected JSON object in metadata file");
    }
    
    std::unordered_map<VectorID, Metadata> metadataMap;
    for (const auto& [key, value] : j.items()) {
        VectorID id = std::stoull(key);  // Convert string key to VectorID
        metadataMap[id] = jsonToMetadata(value);
    }
    
    return metadataMap;
}
} // namespace arrow::utils


#endif // ARROWDB_H

#include "binary.h"
