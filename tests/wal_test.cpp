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
        testDir = std::filesystem::temp_directory_path() / "arrow_wal_test";
        std::filesystem::create_directories(testDir);
        
        gen.seed(42);
    }
    
    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }
    
    std::filesystem::path testDir;
    std::mt19937 gen;
    
    std::string GetTestPath(const std::string& dirname) {
        return (testDir / dirname).string();
    }
    
    WAL::Entry CreateTestEntry(WAL::OperationType type = WAL::OperationType::INSERT,
                               VectorID id = 1,
                               uint32_t dim = 3,
                               uint64_t lsn = 1,
                               uint64_t txid = 1,
                               const std::vector<float>& embedding = {}) {
        std::vector<float> vec = embedding.empty() ? RandomVector(dim, gen) : embedding;
        return WAL::Entry(type, lsn, txid, id, dim, vec);
    }
};

// ============================================================================
// UNIT TESTS - Test components without file I/O
// ============================================================================

TEST_F(WALTest, HeaderDefaults) {
    WAL::Header header;
    header.computeHeaderCrc32();
    EXPECT_EQ(sizeof(WAL::Header), 24);
    EXPECT_EQ(sizeof(header), 24);
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
    EXPECT_EQ(header.flags, 0);
    EXPECT_EQ(header.creationTime, 0);
    uint32_t EXPECTEDCRC = 1956998465;
    EXPECT_EQ(header.headerCrc32, EXPECTEDCRC);
    std::cout << "Header crc32: " << header.headerCrc32 << "\n";
    EXPECT_EQ(header._padding, 0);
}

// OLD HEADER TESTS
/*
TEST_F(WALTest, HeaderEndiannessDetection) {
    // Test that endianness detection works
    WAL_ wal;
    uint8_t endianness = WAL::Header.endianness;
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
*/

TEST_F(WALTest, HeaderWriteReadRoundTrip) {
    WAL::Header original;
    original.magic = 0x41574C01;
    original.version = 2;
    original.flags = 0x1234;
    original.creationTime = 1234567890;
    original._padding = 0;

    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    original.write(writer);

    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Header read;
    read.read(reader);

    EXPECT_EQ(read.magic, original.magic);
    EXPECT_EQ(read.version, original.version);
    EXPECT_EQ(read.flags, original.flags);
    EXPECT_EQ(read.creationTime, original.creationTime);
    EXPECT_EQ(read._padding, original._padding);
}

TEST_F(WALTest, HeaderReadFailure) {
    std::stringstream buffer;
    BinaryReader reader(buffer);
    WAL::Header header;
    
    header.read(reader);
    EXPECT_EQ(header.magic, 0);
}

TEST_F(WALTest, EntryConstructor) {
    WAL::Entry entry(WAL::OperationType::INSERT, 1, 42, 42, 3, {1.0f, 2.0f, 3.0f});
    EXPECT_EQ(entry.type, WAL::OperationType::INSERT);
    EXPECT_EQ(entry.lsn, 1);
    EXPECT_EQ(entry.txid, 42);
    EXPECT_EQ(entry.vectorId, 42);
    EXPECT_EQ(entry.dimension, 3);
    EXPECT_EQ(entry.embedding, std::vector<float>({1.0f, 2.0f, 3.0f}));
}

TEST_F(WALTest, OperationTypeToString) {
    WAL::Entry commitEntry(WAL::OperationType::COMMIT_TXN, 1, 1, 1, 1, {1.0f});
    WAL::Entry abortEntry(WAL::OperationType::ABORT_TXN, 1, 1, 1, 1, {1.0f});
    WAL::Entry insertEntry(WAL::OperationType::INSERT, 1, 1, 1, 1, {1.0f});
    WAL::Entry deleteEntry(WAL::OperationType::DELETE, 1, 1, 1, 1, {1.0f});
    WAL::Entry updateEntry(WAL::OperationType::UPDATE, 1, 1, 1, 1, {1.0f});
    WAL::Entry batchInsertEntry(WAL::OperationType::BATCH_INSERT, 1, 1, 1, 1, {1.0f});
    
    EXPECT_EQ(commitEntry.typeToString(), "COMMIT_TXN");
    EXPECT_EQ(abortEntry.typeToString(), "ABORT_TXN");
    EXPECT_EQ(insertEntry.typeToString(), "INSERT");
    EXPECT_EQ(deleteEntry.typeToString(), "DELETE");
    EXPECT_EQ(updateEntry.typeToString(), "UPDATE");
    EXPECT_EQ(batchInsertEntry.typeToString(), "BATCH_INSERT");
}

TEST_F(WALTest, EntryToJson) {
    WAL::Entry entry(WAL::OperationType::INSERT, 5, 10, 42, 2, {1.5f, 2.5f});
    utils::json j = entry.toJson();
    
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j["type"], "INSERT");
    EXPECT_EQ(j["lsn"], 5);
    EXPECT_EQ(j["txid"], 10);
    EXPECT_EQ(j["vectorId"], 42);
    EXPECT_EQ(j["dimension"], 2);
    EXPECT_EQ(j["embedding"], std::vector<float>({1.5f, 2.5f}));
}

TEST_F(WALTest, EntryCrcComputation) {
    WAL::Entry entry1(WAL::OperationType::INSERT, 1, 1, 1, 2, {1.0f, 2.0f});
    WAL::Entry entry2(WAL::OperationType::INSERT, 1, 1, 1, 2, {1.0f, 2.0f});
    WAL::Entry entry3(WAL::OperationType::DELETE, 1, 1, 1, 2, {1.0f, 2.0f});
    
    uint32_t headerCrc1 = entry1.computeHeaderCrc();
    uint32_t headerCrc2 = entry2.computeHeaderCrc();
    uint32_t headerCrc3 = entry3.computeHeaderCrc();
    
    EXPECT_EQ(headerCrc1, headerCrc2);
    EXPECT_NE(headerCrc1, headerCrc3);
    
    uint32_t payloadCrc1 = entry1.computePayloadCrc();
    uint32_t payloadCrc2 = entry2.computePayloadCrc();
    
    EXPECT_EQ(payloadCrc1, payloadCrc2);
}

TEST_F(WALTest, EntryWriteReadRoundTrip) {
    WAL::Entry original(WAL::OperationType::UPDATE, 123, 456, 123, 4, {1.1f, 2.2f, 3.3f, 4.4f});
    
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    
    WAL::Status writeStatus = original.write(writer);
    EXPECT_EQ(writeStatus, WAL::Status::SUCCESS);
    
    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Entry read(reader);
    
    EXPECT_EQ(read.type, original.type);
    EXPECT_EQ(read.lsn, original.lsn);
    EXPECT_EQ(read.txid, original.txid);
    EXPECT_EQ(read.vectorId, original.vectorId);
    EXPECT_EQ(read.dimension, original.dimension);
    EXPECT_EQ(read.embedding, original.embedding);
}

TEST_F(WALTest, EntryReadWithCrcMismatch) {
    WAL::Entry original(WAL::OperationType::INSERT, 1, 1, 1, 2, {1.0f, 2.0f});
    
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    original.write(writer);
    
    buffer.seekp(-4, std::ios::end);
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
    
    writer.write(static_cast<uint32_t>(0));
    writer.write(WAL::OperationType::INSERT);
    writer.write(static_cast<uint16_t>(1));
    writer.write(static_cast<uint64_t>(1));
    writer.write(static_cast<uint64_t>(1));
    writer.write(static_cast<uint32_t>(0));
    writer.write(static_cast<uint32_t>(1));
    writer.write(static_cast<VectorID>(1));
    writer.write(static_cast<uint32_t>(2));
    writer.write(static_cast<uint8_t>(0));
    writer.write(std::vector<float>({1.0f, 2.0f, 3.0f}));
    writer.write(static_cast<uint32_t>(0));
    
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
    WAL::Entry entry1 = CreateTestEntry(WAL::OperationType::INSERT, 1, 3, 1, 1, {1.0f, 2.0f, 3.0f});
    WAL::Entry entry2 = CreateTestEntry(WAL::OperationType::DELETE, 2, 3, 2, 1, {4.0f, 5.0f, 6.0f});
    
    std::string walPath = GetTestPath("reset_test");
    
    EXPECT_EQ(wal.log(entry1, walPath, true), WAL::Status::SUCCESS);
    EXPECT_EQ(wal.log(entry2, walPath, true), WAL::Status::SUCCESS);
    
    auto result = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(result));
    auto& entries = std::get<std::vector<WAL::EntryPtr>>(result);
    EXPECT_EQ(entries.size(), 1);
    EXPECT_EQ((*entries[0]).vectorId, 2);
}

TEST_F(WALTest, WALLogAppendMode) {
    WAL wal;
    WAL::Entry entry1 = CreateTestEntry(WAL::OperationType::INSERT, 1, 3, 1, 1, {1.0f, 2.0f, 3.0f});
    WAL::Entry entry2 = CreateTestEntry(WAL::OperationType::UPDATE, 2, 3, 2, 2, {4.0f, 5.0f, 6.0f});
    
    std::string walPath = GetTestPath("append_test");
    
    EXPECT_EQ(wal.log(entry1, walPath, true), WAL::Status::SUCCESS);
    EXPECT_EQ(wal.log(entry2, walPath, false), WAL::Status::SUCCESS);
    
    auto result = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(result));
    auto& entries = std::get<std::vector<WAL::EntryPtr>>(result);
    EXPECT_EQ(entries.size(), 2);
    EXPECT_EQ((*entries[0]).vectorId, 1);
    EXPECT_EQ((*entries[1]).vectorId, 2);
}

TEST_F(WALTest, WALReadFirstEntry) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry(WAL::OperationType::INSERT, 42, 2, 5, 10, {3.14f, 2.71f});
    
    std::string walPath = GetTestPath("read_entry_test");
    wal.log(entry, walPath, true);
    
    auto readResult = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::EntryPtr>(readResult));

    WAL::EntryPtr& readEntry = std::get<WAL::EntryPtr>(readResult);
    EXPECT_EQ((*readEntry).type, entry.type);
    EXPECT_EQ((*readEntry).vectorId, entry.vectorId);
    EXPECT_EQ((*readEntry).dimension, entry.dimension);
    EXPECT_EQ((*readEntry).embedding, entry.embedding);
}

TEST_F(WALTest, WALReadAllEntries) {
    WAL wal;
    std::vector<WAL::Entry> testEntries;
    testEntries.push_back(CreateTestEntry(WAL::OperationType::INSERT, 1, 2, 1, 1, {1.0f, 2.0f}));
    testEntries.push_back(CreateTestEntry(WAL::OperationType::UPDATE, 2, 2, 2, 2, {3.0f, 4.0f}));
    testEntries.push_back(CreateTestEntry(WAL::OperationType::DELETE, 3, 2, 3, 3, {5.0f, 6.0f}));
    
    std::string walPath = GetTestPath("read_all_test");
    
    wal.log(testEntries[0], walPath, true);
    wal.log(testEntries[1], walPath, false);
    wal.log(testEntries[2], walPath, false);
    
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(readResult));

    auto& entries = std::get<std::vector<WAL::EntryPtr>>(readResult);
    EXPECT_EQ(entries.size(), 3);
    
    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ((*entries[i]).vectorId, testEntries[i].vectorId);
        EXPECT_EQ((*entries[i]).type, testEntries[i].type);
        EXPECT_EQ((*entries[i]).embedding, testEntries[i].embedding);
    }
}

TEST_F(WALTest, WALReadAllEmptyFile) {
    WAL wal;
    
    std::string walPath = GetTestPath("empty_read_test");
    std::filesystem::create_directories(walPath);
    
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
    
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary | std::ios::trunc);
    file.write("corrupted", 9);
    file.close();
    
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
}

/* OLD TEST
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
*/

TEST_F(WALTest, WALLogCreatesParentDirectories) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("nonexistent_dir/subdir");
    
    EXPECT_EQ(wal.log(entry, walPath, true), WAL::Status::SUCCESS);
    
    EXPECT_TRUE(std::filesystem::exists(walPath));
    EXPECT_TRUE(std::filesystem::is_directory(walPath));
    
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
        originalEntries.push_back(CreateTestEntry(
            WAL::OperationType::INSERT, 
            static_cast<VectorID>(i), 
            3,
            static_cast<uint64_t>(i + 1),
            static_cast<uint64_t>(i + 1)
        ));
    }
    
    std::string walPath = GetTestPath("roundtrip_test");
    
    wal.log(originalEntries[0], walPath, true);
    for (size_t i = 1; i < numEntries; ++i) {
        wal.log(originalEntries[i], walPath, false);
    }
    
    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(readResult));

    auto& readEntries = std::get<std::vector<WAL::EntryPtr>>(readResult);
    EXPECT_EQ(readEntries.size(), numEntries);
    
    for (size_t i = 0; i < numEntries; ++i) {
        EXPECT_EQ((*readEntries[i]).vectorId, originalEntries[i].vectorId);
        EXPECT_EQ((*readEntries[i]).type, originalEntries[i].type);
        EXPECT_EQ((*readEntries[i]).dimension, originalEntries[i].dimension);
        EXPECT_EQ((*readEntries[i]).embedding, originalEntries[i].embedding);
    }
}

TEST_F(WALTest, WALEmptyEmbedding) {
    WAL wal;
    WAL::Entry entry(WAL::OperationType::DELETE, 1, 1, 1, 0, {});
    
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
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
}

TEST_F(WALTest, WALReadHeaderEmptyFile) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_header_empty");
    std::filesystem::create_directories(walPath);
    
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Header>(headerResult));
    
    const WAL::Header& header = std::get<WAL::Header>(headerResult);
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
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
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(headerResult));
    EXPECT_EQ(std::get<WAL::Status>(headerResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadHeaderCorruptedMagic) {
    WAL wal;
    WAL::Entry entry = CreateTestEntry();
    
    std::string walPath = GetTestPath("read_header_corrupted_magic");
    wal.log(entry, walPath, true);
    
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
    
    wal.log(entry, walPath, true);
    
    auto headerResult = wal.readHeader(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Header>(headerResult));
    
    const WAL::Header& header = std::get<WAL::Header>(headerResult);
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
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
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
}

// ============================================================================
// ADDITIONAL EDGE CASE TESTS
// ============================================================================

TEST_F(WALTest, WALReadEmptyFile) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_empty_file");
    std::filesystem::create_directories(walPath);
    
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();
    
    auto readResult = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::Status>(readResult));
    EXPECT_EQ(std::get<WAL::Status>(readResult), WAL::Status::FAILURE);
}

TEST_F(WALTest, WALReadFileTooSmallForHeader) {
    WAL wal;
    
    std::string walPath = GetTestPath("read_too_small");
    std::filesystem::create_directories(walPath);
    
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.write("abc", 3);
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
    WAL::Entry entry1 = CreateTestEntry(WAL::OperationType::INSERT, 1, 2, 1, 1, {1.0f, 2.0f});
    WAL::Entry entry2 = CreateTestEntry(WAL::OperationType::UPDATE, 2, 2, 2, 2, {3.0f, 4.0f});
    
    wal.log(entry1, "", true);
    wal.log(entry2, "", false);
    
    EXPECT_NO_THROW(wal.print());
}

TEST_F(WALTest, WALTransactionTypes) {
    WAL wal;
    
    WAL::Entry commitEntry(WAL::OperationType::COMMIT_TXN, 1, 1, 0, 0, {});
    WAL::Entry abortEntry(WAL::OperationType::ABORT_TXN, 2, 1, 0, 0, {});
    
    std::string walPath = GetTestPath("txn_types_test");
    
    EXPECT_EQ(wal.log(commitEntry, walPath, true), WAL::Status::SUCCESS);
    EXPECT_EQ(wal.log(abortEntry, walPath, false), WAL::Status::SUCCESS);
    
    auto result = wal.readAll(walPath);
    EXPECT_TRUE(std::holds_alternative<std::vector<WAL::EntryPtr>>(result));
    auto& entries = std::get<std::vector<WAL::EntryPtr>>(result);
    EXPECT_EQ(entries.size(), 2);
    EXPECT_EQ((*entries[0]).type, WAL::OperationType::COMMIT_TXN);
    EXPECT_EQ((*entries[1]).type, WAL::OperationType::ABORT_TXN);
}

TEST_F(WALTest, WALBatchInsert) {
    WAL wal;
    WAL::Entry batchEntry(WAL::OperationType::BATCH_INSERT, 1, 1, 0, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    
    std::string walPath = GetTestPath("batch_insert_test");
    EXPECT_EQ(wal.log(batchEntry, walPath, true), WAL::Status::SUCCESS);
    
    auto result = wal.read(walPath);
    EXPECT_TRUE(std::holds_alternative<WAL::EntryPtr>(result));
    auto& entry = *std::get<WAL::EntryPtr>(result);
    EXPECT_EQ(entry.type, WAL::OperationType::BATCH_INSERT);
    EXPECT_EQ(entry.dimension, 4);
}

TEST_F(WALTest, EntryWithAllFields) {
    WAL::Entry entry(WAL::OperationType::INSERT, 100, 200, 42, 5, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    
    EXPECT_EQ(entry.type, WAL::OperationType::INSERT);
    EXPECT_EQ(entry.lsn, 100);
    EXPECT_EQ(entry.txid, 200);
    EXPECT_EQ(entry.vectorId, 42);
    EXPECT_EQ(entry.dimension, 5);
    EXPECT_EQ(entry.version, 1);
    EXPECT_EQ(entry.padding, 0);
    
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    EXPECT_EQ(entry.write(writer), WAL::Status::SUCCESS);
    
    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Entry readEntry(reader);
    
    EXPECT_EQ(readEntry.type, entry.type);
    EXPECT_EQ(readEntry.lsn, entry.lsn);
    EXPECT_EQ(readEntry.txid, entry.txid);
    EXPECT_EQ(readEntry.vectorId, entry.vectorId);
    EXPECT_EQ(readEntry.dimension, entry.dimension);
    EXPECT_EQ(readEntry.embedding, entry.embedding);
}

TEST_F(WALTest, HeaderComputeCrc) {
    WAL::Header header;
    header.magic = 0x41574C01;
    header.version = 1;
    header.flags = 0;
    header.creationTime = 1234567890;
    header._padding = 0;
    
    header.computeHeaderCrc32();
    EXPECT_NE(header.headerCrc32, 0);
    
    std::stringstream buffer(std::ios::binary | std::ios::in | std::ios::out);
    BinaryWriter writer(buffer);
    header.write(writer);
    
    buffer.seekg(0);
    BinaryReader reader(buffer);
    WAL::Header readHeader;
    readHeader.read(reader);
    
    EXPECT_EQ(readHeader.headerCrc32, header.headerCrc32);
}
