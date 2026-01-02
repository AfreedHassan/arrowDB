#include <gtest/gtest.h>
#include "arrow/wal.h"
#include "arrow/utils/binary.h"
#include "test_util.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <random>

using namespace arrow;
using arrow::testing::RandomVector;

// ============================================================================
// WAL Test Fixture
// ============================================================================

class WALTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        testDir = std::filesystem::temp_directory_path() / "arrow_wal_test";
        std::filesystem::create_directories(testDir);
        
        // Initialize random number generator with fixed seed for reproducibility
        gen.seed(42);
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }
    
    std::filesystem::path testDir;
    std::mt19937 gen;  // Random number generator for RandomVector
    
    std::string GetTestPath(const std::string& dirname) {
        return (testDir / dirname).string();
    }
    
    // Helper to create test entry
    // If embedding is not provided, generates a random normalized vector
    WAL::Entry CreateTestEntry(WAL::EntryType type = WAL::EntryType::INSERT, 
                              VectorID id = 1, 
                              uint32_t dim = 3, 
                              const std::vector<float>& embedding = {}) {
        std::vector<float> vec = embedding.empty() ? RandomVector(dim, gen) : embedding;
        return WAL::Entry(type, id, dim, vec);
    }
};

// ============================================================================
// UNIT TESTS - Test components without file I/O
// ============================================================================

// Header tests
TEST_F(WALTest, HeaderDefaults) {
    WAL::Header header;
    EXPECT_EQ(header.magic, 0x01);
    EXPECT_EQ(header.version, 1);
    EXPECT_EQ(header.endianness, 0);  // little-endian default
    EXPECT_EQ(header.crc32, 0);
}

TEST_F(WALTest, HeaderEndiannessDetection) {
    // Test that endianness detection works
    uint8_t endianness = WAL::Header::detectEndianness();
    // Should be 0 (little-endian) or 1 (big-endian)
    EXPECT_TRUE(endianness == 0 || endianness == 1);
}

TEST_F(WALTest, HeaderEndiannessCompatibility) {
    WAL::Header header;
    header.endianness = WAL::Header::detectEndianness();
    EXPECT_TRUE(header.isEndiannessCompatible());
    
    // Test incompatible endianness
    header.endianness = 1 - WAL::Header::detectEndianness();  // Flip it
    EXPECT_FALSE(header.isEndiannessCompatible());
}

TEST_F(WALTest, HeaderWriteReadRoundTrip) {
    WAL::Header original;
    original.magic = 0x01;
    original.version = 2;
    original.endianness = 1;
    original.crc32 = 0xDEADBEEF;
    
    // Write to buffer
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    original.write(writer);
    
    // Read back
    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Header read;
    read.read(reader);
    
    EXPECT_EQ(read.magic, original.magic);
    EXPECT_EQ(read.version, original.version);
    EXPECT_EQ(read.endianness, original.endianness);
    EXPECT_EQ(read.crc32, original.crc32);
}

TEST_F(WALTest, HeaderReadFailure) {
    std::stringstream buffer;
    BinaryReader reader(buffer);
    WAL::Header header;
    
    // Empty buffer should set magic to 0
    header.read(reader);
    EXPECT_EQ(header.magic, 0);
}

// Entry tests
TEST_F(WALTest, EntryConstructor) {
    WAL::Entry entry(WAL::EntryType::INSERT, 42, 3, {1.0f, 2.0f, 3.0f});
    EXPECT_EQ(entry.type, WAL::EntryType::INSERT);
    EXPECT_EQ(entry.id, 42);
    EXPECT_EQ(entry.dimension, 3);
    EXPECT_EQ(entry.embedding, std::vector<float>({1.0f, 2.0f, 3.0f}));
}

TEST_F(WALTest, EntryTypeToString) {
    WAL::Entry insertEntry(WAL::EntryType::INSERT, 1, 1, {1.0f});
    WAL::Entry deleteEntry(WAL::EntryType::DELETE, 1, 1, {1.0f});
    WAL::Entry updateEntry(WAL::EntryType::UPDATE, 1, 1, {1.0f});
    
    EXPECT_EQ(insertEntry.typeToString(), "INSERT");
    EXPECT_EQ(deleteEntry.typeToString(), "DELETE");
    EXPECT_EQ(updateEntry.typeToString(), "UPDATE");
}

TEST_F(WALTest, EntryToJson) {
    WAL::Entry entry(WAL::EntryType::INSERT, 42, 2, {1.5f, 2.5f});
    utils::json j = entry.toJson();
    
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j["type"], "INSERT");
    EXPECT_EQ(j["id"], 42);
    EXPECT_EQ(j["dimension"], 2);
    EXPECT_EQ(j["embedding"], std::vector<float>({1.5f, 2.5f}));
}

TEST_F(WALTest, EntryCrc32Computation) {
    WAL::Entry entry1(WAL::EntryType::INSERT, 1, 2, {1.0f, 2.0f});
    WAL::Entry entry2(WAL::EntryType::INSERT, 1, 2, {1.0f, 2.0f});
    WAL::Entry entry3(WAL::EntryType::DELETE, 1, 2, {1.0f, 2.0f});
    
    uint32_t crc1 = entry1.computeCrc32();
    uint32_t crc2 = entry2.computeCrc32();
    uint32_t crc3 = entry3.computeCrc32();
    
    // Identical entries should have same CRC
    EXPECT_EQ(crc1, crc2);
    // Different type should have different CRC
    EXPECT_NE(crc1, crc3);
}

TEST_F(WALTest, EntryWriteReadRoundTrip) {
    WAL::Entry original(WAL::EntryType::UPDATE, 123, 4, {1.1f, 2.2f, 3.3f, 4.4f});
    
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    
    WAL::Status writeStatus = original.write(writer);
    EXPECT_EQ(writeStatus, WAL::Status::SUCCESS);
    
    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Entry read(reader);  // Constructor calls read() internally
    
    // Entry constructor already reads, so we don't need to call read() again
    EXPECT_EQ(read.type, original.type);
    EXPECT_EQ(read.id, original.id);
    EXPECT_EQ(read.dimension, original.dimension);
    EXPECT_EQ(read.embedding, original.embedding);
}

TEST_F(WALTest, EntryReadWithCrcMismatch) {
    WAL::Entry original(WAL::EntryType::INSERT, 1, 2, {1.0f, 2.0f});
    
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    original.write(writer);
    
    // Corrupt the CRC32 value in the buffer
    buffer.seekp(-4, std::ios::end);  // Seek to CRC32 position
    uint32_t badCrc = 0xFFFFFFFF;
    buffer.write(reinterpret_cast<char*>(&badCrc), sizeof(badCrc));
    
    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Entry read(reader);
    
    WAL::Status readStatus = read.read(reader);
    EXPECT_EQ(readStatus, WAL::Status::FAILURE);
}

TEST_F(WALTest, EntryDimensionMismatch) {
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    
    // Write entry with dimension 2 but embedding size 3
    writer.write(WAL::EntryType::INSERT);
    writer.write(static_cast<VectorID>(1));
    writer.write(static_cast<uint32_t>(2));  // dimension = 2
    writer.write(std::vector<float>({1.0f, 2.0f, 3.0f}));  // but 3 elements
    uint32_t crc = 0;  // dummy CRC
    writer.write(crc);
    
    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Entry entry(reader);
    
    WAL::Status readStatus = entry.read(reader);
    EXPECT_EQ(readStatus, WAL::Status::FAILURE);
}

// ============================================================================
// INTEGRATION TESTS - Test file I/O operations
// ============================================================================

TEST_F(WALTest, WALLogCreatesDirectory) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("test_wal_dir");
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    
    EXPECT_NO_THROW(wal.log(entry, walPath, true));
    EXPECT_TRUE(std::filesystem::exists(walPath));
    EXPECT_TRUE(std::filesystem::is_directory(walPath));
    EXPECT_TRUE(std::filesystem::exists(dbPath));
}

TEST_F(WALTest, WALLogResetMode) {
    WAL wal;
    WAL::Entry entry1 = CreateTestEntry(WAL::EntryType::INSERT, 1, 3, {1.0f, 2.0f, 3.0f});
    WAL::Entry entry2 = CreateTestEntry(WAL::EntryType::DELETE, 2, 3, {4.0f, 5.0f, 6.0f});
    
    std::string walPath = GetTestPath("reset_test");
    
    // Log first entry in reset mode
    EXPECT_EQ(wal.log(entry1, walPath, true), WAL::Status::SUCCESS);
    
    // Log second entry in reset mode (should overwrite)
    EXPECT_EQ(wal.log(entry2, walPath, true), WAL::Status::SUCCESS);
    
    // Read all entries - should only have the second one
    auto result = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(result));
    auto& entries = std::get<std::vector<WAL::EntryPtr>>(result);
    EXPECT_EQ(entries.size(), 1);
    EXPECT_EQ((*entries[0]).id, 2);
}

TEST_F(WALTest, WALLogAppendMode) {
    WAL wal;
    WAL::Entry entry1 = CreateTestEntry(WAL::EntryType::INSERT, 1, 3, {1.0f, 2.0f, 3.0f});
    WAL::Entry entry2 = CreateTestEntry(WAL::EntryType::UPDATE, 2, 3, {4.0f, 5.0f, 6.0f});
    
    std::string walPath = GetTestPath("append_test");
    
    // Log first entry in reset mode
    EXPECT_EQ(wal.log(entry1, walPath, true), WAL::Status::SUCCESS);
    
    // Log second entry in append mode
    EXPECT_EQ(wal.log(entry2, walPath, false), WAL::Status::SUCCESS);
    
    // Read all entries - should have both
    auto result = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(result));
    auto& entries = std::get<std::vector<WAL::EntryPtr>>(result);
    EXPECT_EQ(entries.size(), 2);
    EXPECT_EQ((*entries[0]).id, 1);
    EXPECT_EQ((*entries[1]).id, 2);
}

TEST_F(WALTest, WALReadFirstEntry) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry(WAL::EntryType::INSERT, 42, 2, {3.14f, 2.71f});
    
    std::string walPath = GetTestPath("read_entry_test");
    wal.log(entry, walPath, true);
    
    auto readResult = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::EntryPtr>(readResult));

    WAL::EntryPtr& readEntry = std::get<WAL::EntryPtr>(readResult);
    EXPECT_EQ((*readEntry).type, entry.type);
    EXPECT_EQ((*readEntry).id, entry.id);
    EXPECT_EQ((*readEntry).dimension, entry.dimension);
    EXPECT_EQ((*readEntry).embedding, entry.embedding);
}

TEST_F(WALTest, WALReadAllEntries) {
    WAL wal;
    std::vector<WAL::Entry> testEntries;
    testEntries.push_back(CreateTestEntry(WAL::EntryType::INSERT, 1, 2, {1.0f, 2.0f}));
    testEntries.push_back(CreateTestEntry(WAL::EntryType::UPDATE, 2, 2, {3.0f, 4.0f}));
    testEntries.push_back(CreateTestEntry(WAL::EntryType::DELETE, 3, 2, {5.0f, 6.0f}));
    
    std::string walPath = GetTestPath("read_all_test");
    
    // Log entries in append mode
    wal.log(testEntries[0], walPath, true);
    wal.log(testEntries[1], walPath, false);
    wal.log(testEntries[2], walPath, false);
    
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(readResult));

    auto& entries = std::get<std::vector<WAL::EntryPtr>>(readResult);
    EXPECT_EQ(entries.size(), 3);
    
    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ((*entries[i]).id, testEntries[i].id);
        EXPECT_EQ((*entries[i]).type, testEntries[i].type);
        EXPECT_EQ((*entries[i]).embedding, testEntries[i].embedding);
    }
}

TEST_F(WALTest, WALReadAllEmptyFile) {
    WAL wal;
    
    std::string walPath = GetTestPath("empty_read_test");
    std::filesystem::create_directories(walPath);
    
    // Create empty file
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();
    
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(readResult));

    auto& entries = std::get<std::vector<WAL::EntryPtr>>(readResult);
    EXPECT_EQ(entries.size(), 0);
}

TEST_F(WALTest, WALReadAllCorruptedEntry) {
    WAL wal;
    WAL::Entry goodEntry = CreateTestEntry();
    
    std::string walPath = GetTestPath("corrupted_test");
    wal.log(goodEntry, walPath, true);
    
    // Manually corrupt the file by truncating it
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary | std::ios::trunc);
    file.write("corrupted", 9);  // Write some garbage
    file.close();
    
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadAllWithFileCrcMismatch) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("crc_mismatch_test");
    wal.log(entry, walPath, true);
    
    // Corrupt the file CRC32
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::fstream file(dbPath, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(-4, std::ios::end);  // Seek to file CRC32
    uint32_t badCrc = 0xFFFFFFFF;
    file.write(reinterpret_cast<char*>(&badCrc), sizeof(badCrc));
    file.close();
    
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALLogCreatesParentDirectories) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("nonexistent_dir/subdir");
    
    // Should succeed - log() creates directories (including parent directories) automatically
    EXPECT_EQ(wal.log(entry, walPath, true), WAL::Status::SUCCESS);
    
    // Verify directory was created
    EXPECT_TRUE(std::filesystem::exists(walPath));
    EXPECT_TRUE(std::filesystem::is_directory(walPath));
    
    // Verify WAL file was created
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    EXPECT_TRUE(std::filesystem::exists(dbPath));
}

TEST_F(WALTest, WALReadFromNonexistentDirectory) {
    WAL wal;
    
    auto readResult = wal.read("/nonexistent/directory");
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
    
    auto readAllResult = wal.readAll("/nonexistent/directory");
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readAllResult));
    EXPECT_EQ(std::get<WAL::Status>(readAllResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALRoundTripMultipleEntries) {
    WAL wal;
    const size_t numEntries = 10;
    std::vector<WAL::Entry> originalEntries;
    
    for (size_t i = 0; i < numEntries; ++i) {
        // Use RandomVector for more realistic test data
        originalEntries.push_back(CreateTestEntry(
            WAL::EntryType::INSERT, 
            static_cast<VectorID>(i), 
            3
        ));
    }
    
    std::string walPath = GetTestPath("roundtrip_test");
    
    // Log all entries
    wal.log(originalEntries[0], walPath, true);
    for (size_t i = 1; i < numEntries; ++i) {
        wal.log(originalEntries[i], walPath, false);
    }
    
    // Read all back
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(readResult));

    auto& readEntries = std::get<std::vector<WAL::EntryPtr>>(readResult);
    EXPECT_EQ(readEntries.size(), numEntries);
    
    for (size_t i = 0; i < numEntries; ++i) {
        EXPECT_EQ((*readEntries[i]).id, originalEntries[i].id);
        EXPECT_EQ((*readEntries[i]).type, originalEntries[i].type);
        EXPECT_EQ((*readEntries[i]).dimension, originalEntries[i].dimension);
        EXPECT_EQ((*readEntries[i]).embedding, originalEntries[i].embedding);
    }
}

TEST_F(WALTest, WALEmptyEmbedding) {
    WAL wal;
    // Create entry with empty embedding - this should work for dimension 0
    WAL::Entry entry(WAL::EntryType::DELETE, 1, 0, {});
    
    std::string walPath = GetTestPath("empty_embedding_test");
    EXPECT_EQ(wal.log(entry, walPath, true), WAL::Status::SUCCESS);
    
    auto readResult = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::EntryPtr>(readResult));

    WAL::EntryPtr& readEntry = std::get<WAL::EntryPtr>(readResult);
    EXPECT_EQ((*readEntry).dimension, 0);
    EXPECT_TRUE((*readEntry).embedding.empty());
}

// ============================================================================
// readHeader() TESTS
// ============================================================================

TEST_F(WALTest, WALReadHeaderSuccess) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("read_header_test");
    wal.log(entry, walPath, true);
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Header>(headerResult));
    
    const WAL::Header& header = std::get<WAL::Header>(headerResult);
    EXPECT_EQ(header.magic, 0x01);
    EXPECT_EQ(header.version, 1);
    EXPECT_TRUE(header.isEndiannessCompatible());
}

TEST_F(WALTest, WALReadHeaderEmptyFile) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_header_empty");
    std::filesystem::create_directories(walPath);
    
    // Create empty file
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Header>(headerResult));
    
    const WAL::Header& header = std::get<WAL::Header>(headerResult);
    EXPECT_EQ(header.magic, 0x01);
    EXPECT_EQ(header.version, 1);
    EXPECT_EQ(header.crc32, 0);
    EXPECT_TRUE(header.isEndiannessCompatible());
}

TEST_F(WALTest, WALReadHeaderNonexistentDirectory) {
    WAL wal;
    
    auto headerResult = wal.readHeader("/nonexistent/directory");
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(headerResult));
    EXPECT_EQ(std::get<WAL::Status>(headerResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadHeaderNonexistentFile) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_header_nonexistent_file");
    std::filesystem::create_directories(walPath);
    // Don't create db.wal file
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(headerResult));
    EXPECT_EQ(std::get<WAL::Status>(headerResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadHeaderCorruptedMagic) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("read_header_corrupted_magic");
    wal.log(entry, walPath, true);
    
    // Corrupt the magic number
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::fstream file(dbPath, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(0, std::ios::beg);
    uint32_t badMagic = 0xDEADBEEF;
    file.write(reinterpret_cast<char*>(&badMagic), sizeof(badMagic));
    file.close();
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(headerResult));
    EXPECT_EQ(std::get<WAL::Status>(headerResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadHeaderTooSmall) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_header_too_small");
    std::filesystem::create_directories(walPath);
    
    // Create file with only 4 bytes (less than header size)
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    uint32_t dummy = 0x12345678;
    file.write(reinterpret_cast<char*>(&dummy), sizeof(dummy));
    file.close();
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(headerResult));
    EXPECT_EQ(std::get<WAL::Status>(headerResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadHeaderRoundTrip) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("read_header_roundtrip");
    
    // Log entry (creates header)
    wal.log(entry, walPath, true);
    
    // Read header back
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Header>(headerResult));
    
    const WAL::Header& header = std::get<WAL::Header>(headerResult);
    EXPECT_EQ(header.magic, 0x01);
    EXPECT_EQ(header.version, 1);
    EXPECT_TRUE(header.isEndiannessCompatible());
}

TEST_F(WALTest, WALReadHeaderWithStream) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("read_header_stream");
    wal.log(entry, walPath, true);
    
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ifstream file(dbPath, std::ios::binary);
    
    auto headerResult = wal.readHeader(file);
    EXPECT_TRUE(std::holds_alternative<WAL::Header>(headerResult));
    
    const WAL::Header& header = std::get<WAL::Header>(headerResult);
    EXPECT_EQ(header.magic, 0x01);
    EXPECT_EQ(header.version, 1);
}

// ============================================================================
// ADDITIONAL EDGE CASE TESTS
// ============================================================================

TEST_F(WALTest, WALReadEmptyFile) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_empty_file");
    std::filesystem::create_directories(walPath);
    
    // Create empty file
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();
    
    auto readResult = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadFileWithOnlyHeaderAndCrc32) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("read_only_header");
    wal.log(entry, walPath, true);
    
    // Manually truncate file to only contain header + file CRC32
    // This simulates a file that was written but entry write failed
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    
    // Calculate size: HEADERSIZE + FILECRC32SIZE
    size_t minSize = WAL::HEADERSIZE + WAL::FILECRC32SIZE;
    
    // Resize file to minimum size (header + file CRC32, no entries)
    std::filesystem::resize_file(dbPath, minSize);
    
    // read() should fail because there are no entries (only header + CRC32)
    auto readResult = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALAppendToFileWithoutCrc32) {
    WAL wal;
    
    std::string walPath = GetTestPath("append_no_crc32");
    std::filesystem::create_directories(walPath);
    
    // Create a file with only header (no CRC32 at end)
    // This simulates an incomplete/corrupted WAL file
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    
    // Write only header (HEADERSIZE bytes, but no file CRC32)
    WAL::Header header;
    header.endianness = WAL::Header::detectEndianness();
    BinaryWriter writer(file);
    header.write(writer);
    file.close();
    
    // Verify file size is only HEADERSIZE (no CRC32)
    size_t fileSize = std::filesystem::file_size(dbPath);
    EXPECT_EQ(fileSize, WAL::HEADERSIZE);
    EXPECT_LT(fileSize, WAL::HEADERSIZE + WAL::FILECRC32SIZE);
    
    // Try to append - should fail because file doesn't have CRC32
    // The append logic checks if fileSize >= FILECRC32SIZE, but since HEADERSIZE (11) >= FILECRC32SIZE (4),
    // it will incorrectly try to seek. However, the actual check should verify the file has
    // at least HEADERSIZE + FILECRC32SIZE bytes. The current implementation will fail when
    // trying to read existing entries or when seeking incorrectly.
    WAL::Entry entry = CreateTestEntry();
    EXPECT_EQ(wal.log(entry, walPath, false), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadFileTooSmallForHeader) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_too_small");
    std::filesystem::create_directories(walPath);
    
    // Create file smaller than header size
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.write("abc", 3);  // Only 3 bytes
    file.close();
    
    auto readResult = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
    
    auto readAllResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readAllResult));
    EXPECT_EQ(std::get<WAL::Status>(readAllResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALPrintMethod) {
    WAL wal;
    WAL::Entry entry1 = CreateTestEntry(WAL::EntryType::INSERT, 1, 2, {1.0f, 2.0f});
    WAL::Entry entry2 = CreateTestEntry(WAL::EntryType::UPDATE, 2, 2, {3.0f, 4.0f});
    
    // Use default WAL path (created by constructor)
    // The WAL constructor creates a "wal" directory
    wal.log(entry1, "", true);  // Use default path
    wal.log(entry2, "", false);
    
    // Test that print() doesn't crash and produces output
    // We can't easily test the exact output, but we can verify it doesn't throw
    EXPECT_NO_THROW(wal.print());
}
