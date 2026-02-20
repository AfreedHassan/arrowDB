#ifndef ARROW_JSON_UTILS_H
#define ARROW_JSON_UTILS_H

#include "arrow/types.h"
#include "core/types_internal.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <unordered_map>
#include <iostream>
#include <format>

/**
 * @brief Stream output operator for std::vector.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
	os << std::format("{}", vec) << '\n';
	return os;
}

namespace arrow::utils {
using json = nlohmann::json;

/**
 * @brief Convert MetadataValue to JSON value.
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
 * @brief Convert Space enum to JSON string.
 */
inline json spaceToJson(Space space) {
    switch (space) {
        case Space::Cosine:
            return "Cosine";
        case Space::L2:
            return "L2";
        case Space::InnerProduct:
            return "InnerProduct";
        default:
            throw std::runtime_error("Unknown Index Space");
    }
}

/**
 * @brief Convert JSON string to Space.
 */
inline Space jsonToSpace(const json& j) {
    std::string spaceStr = j.get<std::string>();
    if (spaceStr == "Cosine") {
        return Space::Cosine;
    } else if (spaceStr == "L2") {
        return Space::L2;
    } else if (spaceStr == "InnerProduct") {
        return Space::InnerProduct;
    } else {
        throw std::runtime_error("Unknown Index Space: " + spaceStr);
    }
}

/**
 * @brief Convert DataType enum to JSON string.
 */
inline json dataTypeToJson(DataType dtype) {
    switch (dtype) {
        case DataType::Int32:
            return "Int32";
        case DataType::Float32:
            return "Float32";
        default:
            throw std::runtime_error("Unknown DataType");
    }
}

/**
 * @brief Convert JSON string to DataType enum.
 */
inline DataType jsonToDataType(const json& j) {
    std::string dtypeStr = j.get<std::string>();
    if (dtypeStr == "Int32") {
        return DataType::Int32;
    } else if (dtypeStr == "Float32") {
        return DataType::Float32;
    } else {
        throw std::runtime_error("Unknown DataType: " + dtypeStr);
    }
}

/// Export metadata to JSON file.
/// @return true on success, false on I/O error.
inline bool exportMetadataToJson(
    const std::unordered_map<InternalID, Metadata>& metadataMap,
    const std::string& filepath
) {
    json j = json::object();
    for (const auto& [id, metadata] : metadataMap) {
        j[std::to_string(id)] = metadataToJson(metadata);
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    file << j.dump(2);
    file.close();
    return !file.fail();
}

/**
 * @brief Import metadata from JSON file.
 */
inline std::unordered_map<InternalID, Metadata> importMetadataFromJson(
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

    std::unordered_map<InternalID, Metadata> metadataMap;
    for (const auto& [key, value] : j.items()) {
        InternalID id = std::stoull(key);
        metadataMap[id] = jsonToMetadata(value);
    }

    return metadataMap;
}
} // namespace arrow::utils

#endif // ARROW_JSON_UTILS_H
