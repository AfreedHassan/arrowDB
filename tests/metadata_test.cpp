#include <gtest/gtest.h>
#include "arrow/utils/utils.h"
#include "arrow/types.h"
#include <unordered_map>
#include <filesystem>
#include <fstream>

using namespace arrow;

// ============================================================================
// UNIT TESTS - Test JSON conversion functions without file I/O
// ============================================================================

class MetadataUnit : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test metadata
        Metadata vec1Meta;
        vec1Meta["category"] = std::string("image");
        vec1Meta["tags"] = std::string("dog,pet,animal");
        vec1Meta["score"] = 0.95;
        vec1Meta["active"] = true;
        metadataMap_[1] = vec1Meta;
        
        Metadata vec2Meta;
        vec2Meta["category"] = std::string("text");
        vec2Meta["author"] = std::string("John Doe");
        vec2Meta["word_count"] = static_cast<int64_t>(1250);
        metadataMap_[2] = vec2Meta;
        
        Metadata vec3Meta;
        vec3Meta["category"] = std::string("audio");
        vec3Meta["duration"] = 180.5;
        vec3Meta["format"] = std::string("mp3");
        vec3Meta["active"] = false;
        metadataMap_[3] = vec3Meta;
    }
    
    std::unordered_map<VectorID, Metadata> metadataMap_;
};

// Test MetadataValue to JSON conversion for all variant types
TEST_F(MetadataUnit, MetadataValueToJsonInt64) {
    MetadataValue value = static_cast<int64_t>(42);
    utils::json j = utils::metadataValueToJson(value);
    EXPECT_TRUE(j.is_number_integer());
    EXPECT_EQ(j.get<int64_t>(), 42);
}

TEST_F(MetadataUnit, MetadataValueToJsonDouble) {
    MetadataValue value = 3.14159;
    utils::json j = utils::metadataValueToJson(value);
    EXPECT_TRUE(j.is_number_float());
    EXPECT_DOUBLE_EQ(j.get<double>(), 3.14159);
}

TEST_F(MetadataUnit, MetadataValueToJsonString) {
    MetadataValue value = std::string("test_string");
    utils::json j = utils::metadataValueToJson(value);
    EXPECT_TRUE(j.is_string());
    EXPECT_EQ(j.get<std::string>(), "test_string");
}

TEST_F(MetadataUnit, MetadataValueToJsonBool) {
    MetadataValue valueTrue = true;
    MetadataValue valueFalse = false;
    
    utils::json jTrue = utils::metadataValueToJson(valueTrue);
    utils::json jFalse = utils::metadataValueToJson(valueFalse);
    
    EXPECT_TRUE(jTrue.is_boolean());
    EXPECT_TRUE(jFalse.is_boolean());
    EXPECT_EQ(jTrue.get<bool>(), true);
    EXPECT_EQ(jFalse.get<bool>(), false);
}

// Test JSON to MetadataValue conversion
TEST_F(MetadataUnit, JsonToMetadataValueInt64) {
    utils::json j = 42;
    MetadataValue value = utils::jsonToMetadataValue(j);
    EXPECT_EQ(std::get<int64_t>(value), 42);
}

TEST_F(MetadataUnit, JsonToMetadataValueDouble) {
    utils::json j = 3.14159;
    MetadataValue value = utils::jsonToMetadataValue(j);
    EXPECT_DOUBLE_EQ(std::get<double>(value), 3.14159);
}

TEST_F(MetadataUnit, JsonToMetadataValueString) {
    utils::json j = "test_string";
    MetadataValue value = utils::jsonToMetadataValue(j);
    EXPECT_EQ(std::get<std::string>(value), "test_string");
}

TEST_F(MetadataUnit, JsonToMetadataValueBool) {
    utils::json jTrue = true;
    utils::json jFalse = false;
    
    MetadataValue valueTrue = utils::jsonToMetadataValue(jTrue);
    MetadataValue valueFalse = utils::jsonToMetadataValue(jFalse);
    
    EXPECT_EQ(std::get<bool>(valueTrue), true);
    EXPECT_EQ(std::get<bool>(valueFalse), false);
}

TEST_F(MetadataUnit, JsonToMetadataValueInvalidType) {
    utils::json j = utils::json::array(); // Array is not supported
    EXPECT_THROW({
        utils::jsonToMetadataValue(j);
    }, std::runtime_error);
}

// Test Metadata map to JSON conversion
TEST_F(MetadataUnit, MetadataToJson) {
    Metadata meta;
    meta["category"] = std::string("image");
    meta["score"] = 0.95;
    meta["active"] = true;
    meta["count"] = static_cast<int64_t>(42);
    
    utils::json j = utils::metadataToJson(meta);
    
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j.size(), 4);
    EXPECT_EQ(j["category"], "image");
    EXPECT_DOUBLE_EQ(j["score"].get<double>(), 0.95);
    EXPECT_EQ(j["active"], true);
    EXPECT_EQ(j["count"].get<int64_t>(), 42);
}

TEST_F(MetadataUnit, MetadataToJsonEmpty) {
    Metadata emptyMeta;
    utils::json j = utils::metadataToJson(emptyMeta);
    
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j.size(), 0);
}

// Test JSON to Metadata map conversion
TEST_F(MetadataUnit, JsonToMetadata) {
    utils::json j = utils::json::object();
    j["category"] = "image";
    j["score"] = 0.95;
    j["active"] = true;
    j["count"] = 42;
    
    Metadata meta = utils::jsonToMetadata(j);
    
    EXPECT_EQ(meta.size(), 4);
    EXPECT_EQ(std::get<std::string>(meta.at("category")), "image");
    EXPECT_DOUBLE_EQ(std::get<double>(meta.at("score")), 0.95);
    EXPECT_EQ(std::get<bool>(meta.at("active")), true);
    EXPECT_EQ(std::get<int64_t>(meta.at("count")), 42);
}

TEST_F(MetadataUnit, JsonToMetadataEmpty) {
    utils::json j = utils::json::object();
    Metadata meta = utils::jsonToMetadata(j);
    
    EXPECT_EQ(meta.size(), 0);
}

TEST_F(MetadataUnit, JsonToMetadataInvalidType) {
    utils::json j = "not an object"; // Should be an object
    EXPECT_THROW({
        utils::jsonToMetadata(j);
    }, std::runtime_error);
}

// Test round-trip conversion (Metadata -> JSON -> Metadata)
TEST_F(MetadataUnit, MetadataRoundTrip) {
    Metadata original;
    original["category"] = std::string("image");
    original["score"] = 0.95;
    original["active"] = true;
    original["count"] = static_cast<int64_t>(42);
    original["duration"] = 180.5;
    
    // Convert to JSON and back
    utils::json j = utils::metadataToJson(original);
    Metadata converted = utils::jsonToMetadata(j);
    
    // Verify all fields match
    EXPECT_EQ(converted.size(), original.size());
    EXPECT_EQ(std::get<std::string>(converted.at("category")), 
              std::get<std::string>(original.at("category")));
    EXPECT_DOUBLE_EQ(std::get<double>(converted.at("score")), 
                     std::get<double>(original.at("score")));
    EXPECT_EQ(std::get<bool>(converted.at("active")), 
              std::get<bool>(original.at("active")));
    EXPECT_EQ(std::get<int64_t>(converted.at("count")), 
              std::get<int64_t>(original.at("count")));
    EXPECT_DOUBLE_EQ(std::get<double>(converted.at("duration")), 
                     std::get<double>(original.at("duration")));
}

// Test DistanceMetric enum conversion
TEST_F(MetadataUnit, DistanceMetricToJson) {
    EXPECT_EQ(utils::distanceMetricToJson(DistanceMetric::Cosine), "Cosine");
    EXPECT_EQ(utils::distanceMetricToJson(DistanceMetric::L2), "L2");
    EXPECT_EQ(utils::distanceMetricToJson(DistanceMetric::InnerProduct), "InnerProduct");
}

TEST_F(MetadataUnit, JsonToDistanceMetric) {
    EXPECT_EQ(utils::jsonToDistanceMetric("Cosine"), DistanceMetric::Cosine);
    EXPECT_EQ(utils::jsonToDistanceMetric("L2"), DistanceMetric::L2);
    EXPECT_EQ(utils::jsonToDistanceMetric("InnerProduct"), DistanceMetric::InnerProduct);
}

TEST_F(MetadataUnit, JsonToDistanceMetricInvalid) {
    utils::json j = "InvalidMetric";
    EXPECT_THROW({
        utils::jsonToDistanceMetric(j);
    }, std::runtime_error);
}

// Test DataType enum conversion
TEST_F(MetadataUnit, DataTypeToJson) {
    EXPECT_EQ(utils::dataTypeToJson(DataType::Int16), "Int16");
    EXPECT_EQ(utils::dataTypeToJson(DataType::Float16), "Float16");
}

TEST_F(MetadataUnit, JsonToDataType) {
    EXPECT_EQ(utils::jsonToDataType("Int16"), DataType::Int16);
    EXPECT_EQ(utils::jsonToDataType("Float16"), DataType::Float16);
}

TEST_F(MetadataUnit, JsonToDataTypeInvalid) {
    utils::json j = "InvalidType";
    EXPECT_THROW({
        utils::jsonToDataType(j);
    }, std::runtime_error);
}

// Test VectorID -> Metadata map conversion (without file I/O)
TEST_F(MetadataUnit, MetadataMapToJson) {
    // Convert metadata map to JSON manually (simulating what exportMetadataToJson does)
    utils::json j = utils::json::object();
    for (const auto& [id, metadata] : metadataMap_) {
        j[std::to_string(id)] = utils::metadataToJson(metadata);
    }
    
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j.size(), 3);
    EXPECT_TRUE(j.contains("1"));
    EXPECT_TRUE(j.contains("2"));
    EXPECT_TRUE(j.contains("3"));
    
    // Verify content
    EXPECT_EQ(j["1"]["category"], "image");
    EXPECT_DOUBLE_EQ(j["1"]["score"].get<double>(), 0.95);
    EXPECT_EQ(j["2"]["author"], "John Doe");
    EXPECT_EQ(j["3"]["format"], "mp3");
}

// ============================================================================
// INTEGRATION TESTS - Test file I/O operations
// ============================================================================

class MetadataIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test metadata
        Metadata vec1Meta;
        vec1Meta["category"] = std::string("image");
        vec1Meta["tags"] = std::string("dog,pet,animal");
        vec1Meta["score"] = 0.95;
        vec1Meta["active"] = true;
        metadataMap_[1] = vec1Meta;
        
        Metadata vec2Meta;
        vec2Meta["category"] = std::string("text");
        vec2Meta["author"] = std::string("John Doe");
        vec2Meta["word_count"] = static_cast<int64_t>(1250);
        metadataMap_[2] = vec2Meta;
        
        Metadata vec3Meta;
        vec3Meta["category"] = std::string("audio");
        vec3Meta["duration"] = 180.5;
        vec3Meta["format"] = std::string("mp3");
        vec3Meta["active"] = false;
        metadataMap_[3] = vec3Meta;
        
        testFilePath_ = "test_metadata.json";
    }
    
    void TearDown() override {
        // Clean up test file if it exists
        if (std::filesystem::exists(testFilePath_)) {
            std::filesystem::remove(testFilePath_);
        }
    }
    
    std::unordered_map<VectorID, Metadata> metadataMap_;
    std::string testFilePath_;
};

TEST_F(MetadataIntegrationTest, ExportMetadataToJson) {
    // Export metadata to JSON file
    EXPECT_NO_THROW({
        utils::exportMetadataToJson(metadataMap_, testFilePath_);
    });
    
    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(testFilePath_));
    
    // Verify file is not empty
    std::ifstream file(testFilePath_);
    EXPECT_TRUE(file.good());
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
}

TEST_F(MetadataIntegrationTest, ExportMetadataFileFormat) {
    utils::exportMetadataToJson(metadataMap_, testFilePath_);
    
    // Read and parse the JSON file
    std::ifstream file(testFilePath_);
    utils::json j;
    file >> j;
    
    // Verify structure: should be an object with vector IDs as keys
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j.size(), 3);
    
    // Verify vector IDs are strings
    EXPECT_TRUE(j.contains("1"));
    EXPECT_TRUE(j.contains("2"));
    EXPECT_TRUE(j.contains("3"));
}

TEST_F(MetadataIntegrationTest, ImportMetadataFromJson) {
    // First export metadata
    utils::exportMetadataToJson(metadataMap_, testFilePath_);
    
    // Then import it back
    std::unordered_map<VectorID, Metadata> imported;
    EXPECT_NO_THROW({
        imported = utils::importMetadataFromJson(testFilePath_);
    });
    
    // Verify imported size matches
    EXPECT_EQ(imported.size(), metadataMap_.size());
    EXPECT_EQ(imported.size(), 3);
}

TEST_F(MetadataIntegrationTest, ImportMetadataContent) {
    // Export and import metadata
    utils::exportMetadataToJson(metadataMap_, testFilePath_);
    auto imported = utils::importMetadataFromJson(testFilePath_);
    
    // Verify vector 1 metadata
    EXPECT_TRUE(imported.contains(1));
    const auto& vec1Meta = imported[1];
    EXPECT_EQ(std::get<std::string>(vec1Meta.at("category")), "image");
    EXPECT_EQ(std::get<std::string>(vec1Meta.at("tags")), "dog,pet,animal");
    EXPECT_DOUBLE_EQ(std::get<double>(vec1Meta.at("score")), 0.95);
    EXPECT_EQ(std::get<bool>(vec1Meta.at("active")), true);
    
    // Verify vector 2 metadata
    EXPECT_TRUE(imported.contains(2));
    const auto& vec2Meta = imported[2];
    EXPECT_EQ(std::get<std::string>(vec2Meta.at("category")), "text");
    EXPECT_EQ(std::get<std::string>(vec2Meta.at("author")), "John Doe");
    EXPECT_EQ(std::get<int64_t>(vec2Meta.at("word_count")), 1250);
    
    // Verify vector 3 metadata
    EXPECT_TRUE(imported.contains(3));
    const auto& vec3Meta = imported[3];
    EXPECT_EQ(std::get<std::string>(vec3Meta.at("category")), "audio");
    EXPECT_DOUBLE_EQ(std::get<double>(vec3Meta.at("duration")), 180.5);
    EXPECT_EQ(std::get<std::string>(vec3Meta.at("format")), "mp3");
    EXPECT_EQ(std::get<bool>(vec3Meta.at("active")), false);
}

TEST_F(MetadataIntegrationTest, RoundTripMetadata) {
    // Export original metadata
    utils::exportMetadataToJson(metadataMap_, testFilePath_);
    
    // Import it back
    auto imported = utils::importMetadataFromJson(testFilePath_);
    
    // Export imported metadata to a new file
    std::string roundTripPath = "test_metadata_roundtrip.json";
    utils::exportMetadataToJson(imported, roundTripPath);
    
    // Compare the two JSON files
    std::ifstream original(testFilePath_);
    std::ifstream roundTrip(roundTripPath);
    
    utils::json jOriginal, jRoundTrip;
    original >> jOriginal;
    roundTrip >> jRoundTrip;
    
    // The JSON objects should be equivalent (may differ in key order)
    EXPECT_EQ(jOriginal.size(), jRoundTrip.size());
    for (const auto& [key, value] : jOriginal.items()) {
        EXPECT_TRUE(jRoundTrip.contains(key));
        EXPECT_EQ(jRoundTrip[key], value);
    }
    
    // Clean up round trip file
    if (std::filesystem::exists(roundTripPath)) {
        std::filesystem::remove(roundTripPath);
    }
}

TEST_F(MetadataIntegrationTest, ExportEmptyMetadata) {
    std::unordered_map<VectorID, Metadata> emptyMap;
    EXPECT_NO_THROW({
        utils::exportMetadataToJson(emptyMap, testFilePath_);
    });
    
    // File should exist but contain empty object
    std::ifstream file(testFilePath_);
    utils::json j;
    file >> j;
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j.size(), 0);
}

TEST_F(MetadataIntegrationTest, ImportNonExistentFile) {
    EXPECT_THROW({
        utils::importMetadataFromJson("nonexistent_file.json");
    }, std::runtime_error);
}

TEST_F(MetadataIntegrationTest, ImportInvalidJson) {
    // Create a file with invalid JSON
    std::ofstream file(testFilePath_);
    file << "{ invalid json }";
    file.close();
    
    EXPECT_THROW({
        utils::importMetadataFromJson(testFilePath_);
    }, std::exception);
}

TEST_F(MetadataIntegrationTest, ImportNonObjectJson) {
    // Create a file with valid JSON but not an object
    std::ofstream file(testFilePath_);
    file << "[1, 2, 3]";  // Array instead of object
    file.close();
    
    EXPECT_THROW({
        utils::importMetadataFromJson(testFilePath_);
    }, std::runtime_error);
}
