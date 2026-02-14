#include <gtest/gtest.h>
#include "internal/wal.h"
#include "internal/binary.h"
#include "arrow/utils/status.h"
#include "arrow/collection.h"
#include "test_util.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>

using namespace arrow;
using arrow::testing::RandomVector;

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

    std::filesystem::path GetTestPath(const std::string& dirname) {
        return testDir / dirname;
    }

    wal::Entry CreateTestEntry(wal::OperationType type = wal::OperationType::INSERT,
                               const std::string& id = "1",
                               uint32_t dim = 3,
                               uint64_t lsn = 1,
                               uint64_t txid = 1,
                               const std::vector<float>& embedding = {}) {
        std::vector<float> vec = embedding.empty() ? RandomVector(dim, gen) : embedding;
        wal::Entry entry{
            .type = type,
            .version = 1,
            .lsn = lsn,
            .txid = txid,
            .headerCRC = 0,
            .payloadLength = 0,
            .dimension = dim,
            .padding = 0,
            .embedding = vec,
            .payloadCRC = 0
        };
        entry.setVectorID(id);
        entry.payloadLength = entry.computePayloadLength();
        entry.headerCRC = entry.computeHeaderCrc();
        entry.payloadCRC = entry.computePayloadCrc();
        return entry;
    }
};

// =========================================================================
// Pure unit tests (Header/Entry structs, no WAL object needed)
// =========================================================================

TEST_F(WALTest, HeaderDefaults) {
    wal::Header header;
    header.headerCrc32 = header.computeCrc32();
    EXPECT_EQ(sizeof(wal::Header), 24);
    EXPECT_EQ(sizeof(header), 24);
    EXPECT_EQ(header.magic, wal::kWalMagic);
    EXPECT_EQ(header.version, 1);
    EXPECT_EQ(header.flags, 0);
    EXPECT_EQ(header.creationTime, 0);
    uint32_t EXPECTEDCRC = 1956998465;
    EXPECT_EQ(header.headerCrc32, EXPECTEDCRC);
    EXPECT_EQ(header.padding, 0);
}

TEST_F(WALTest, HeaderWriteReadRoundTrip) {
    wal::Header original;
    original.magic = wal::kWalMagic;
    original.version = 2;
    original.flags = 0x1234;
    original.creationTime = 1234567890;
    original.padding = 0;
    original.headerCrc32 = original.computeCrc32();

    auto path = GetTestPath("header_roundtrip.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));
        wal::WriteHeader(original, writer);
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto result = wal::ParseHeader(reader);
        ASSERT_TRUE(result.ok()) << result.status().message();
        const auto& read = result.value();

        EXPECT_EQ(read.magic, original.magic);
        EXPECT_EQ(read.version, original.version);
        EXPECT_EQ(read.flags, original.flags);
        EXPECT_EQ(read.creationTime, original.creationTime);
        EXPECT_EQ(read.headerCrc32, original.headerCrc32);
        EXPECT_EQ(read.padding, original.padding);
    }
}

TEST_F(WALTest, HeaderReadFailure) {
    auto path = GetTestPath("header_empty.bin");
    {
        std::ofstream file(path, std::ios::binary);
        file.close();
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto result = wal::ParseHeader(reader);
        EXPECT_FALSE(result.ok());
    }
}

TEST_F(WALTest, EntryConstructor) {
    std::vector<float> embedding = {1.0f, 2.0f, 3.0f};
    wal::Entry entry{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 42,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 3,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry.setVectorID("42");
    entry.payloadLength = entry.computePayloadLength();
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    EXPECT_EQ(entry.type, wal::OperationType::INSERT);
    EXPECT_EQ(entry.lsn, 1);
    EXPECT_EQ(entry.txid, 42);
    EXPECT_EQ(entry.getVectorID(), "42");
    EXPECT_EQ(entry.dimension, 3);
    EXPECT_EQ(entry.embedding, std::vector<float>({1.0f, 2.0f, 3.0f}));
}

TEST_F(WALTest, EntryToJson) {
    std::vector<float> embedding = {1.5f, 2.5f};
    wal::Entry entry{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 5,
        .txid = 10,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry.setVectorID("42");
    entry.payloadLength = entry.computePayloadLength();
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    utils::json j = entry.toJson();

    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j["type"], "INSERT");
    EXPECT_EQ(j["lsn"], 5);
    EXPECT_EQ(j["txid"], 10);
    EXPECT_EQ(j["vectorId"], "42");
    EXPECT_EQ(j["dimension"], 2);
    EXPECT_EQ(j["embedding"], std::vector<float>({1.5f, 2.5f}));
}

TEST_F(WALTest, EntryCrcComputation) {
    std::vector<float> embedding = {1.0f, 2.0f};
    wal::Entry entry1{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry1.setVectorID("1");
    wal::Entry entry2{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry2.setVectorID("1");
    wal::Entry entry3{
        .type = wal::OperationType::DELETE,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry3.setVectorID("1");
    entry1.headerCRC = entry1.computeHeaderCrc();
    entry1.payloadCRC = entry1.computePayloadCrc();
    entry2.headerCRC = entry2.computeHeaderCrc();
    entry2.payloadCRC = entry2.computePayloadCrc();
    entry3.headerCRC = entry3.computeHeaderCrc();
    entry3.payloadCRC = entry3.computePayloadCrc();

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
    std::vector<float> embedding = {1.1f, 2.2f, 3.3f, 4.4f};
    wal::Entry original{
        .type = wal::OperationType::UPDATE,
        .version = 1,
        .lsn = 123,
        .txid = 456,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 4,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    original.setVectorID("123");
    original.payloadLength = original.computePayloadLength();
    original.headerCRC = original.computeHeaderCrc();
    original.payloadCRC = original.computePayloadCrc();

    auto path = GetTestPath("entry_roundtrip.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));
        utils::Status writeStatus = wal::WriteEntry(original, writer);
        EXPECT_TRUE(writeStatus.ok());
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto readResult = wal::ParseEntry(reader);
        EXPECT_TRUE(readResult.ok());
        wal::Entry read = readResult.value();

        EXPECT_EQ(read.type, original.type);
        EXPECT_EQ(read.lsn, original.lsn);
        EXPECT_EQ(read.txid, original.txid);
        EXPECT_EQ(read.getVectorID(), original.getVectorID());
        EXPECT_EQ(read.dimension, original.dimension);
        EXPECT_EQ(read.embedding, original.embedding);
    }
}

TEST_F(WALTest, EntryReadWithCrcMismatch) {
    std::vector<float> embedding = {1.0f, 2.0f};
    wal::Entry original{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    original.setVectorID("1");
    original.payloadLength = original.computePayloadLength();
    original.headerCRC = original.computeHeaderCrc();
    original.payloadCRC = original.computePayloadCrc();

    auto path = GetTestPath("entry_crc_mismatch.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));
        wal::WriteEntry(original, writer);
    }

    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(-4, std::ios::end);
        uint32_t badCrc = 0xFFFFFFFF;
        file.write(reinterpret_cast<char*>(&badCrc), sizeof(badCrc));
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto readResult = wal::ParseEntry(reader);
        EXPECT_FALSE(readResult.ok());
    }
}

TEST_F(WALTest, EntryDimensionMismatch) {
    auto path = GetTestPath("entry_dimension_mismatch.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

        writer.write(wal::OperationType::INSERT);
        writer.write(static_cast<uint16_t>(1));
        writer.write(static_cast<uint64_t>(1));
        writer.write(static_cast<uint64_t>(1));
        writer.write(static_cast<uint32_t>(0));
        // payloadLength: 128 bytes vectorID + 4 bytes dimension + 1 byte padding + 12 bytes embedding
        writer.write(static_cast<uint32_t>(128 + 4 + 1 + 3 * sizeof(float)));
        // Write 128 bytes of vectorID (all zeros)
        char vectorID[128] = {};
        writer.write(vectorID);
        writer.write(static_cast<uint32_t>(2));
        writer.write(static_cast<uint8_t>(0));
        writer.write(std::vector<float>({1.0f, 2.0f, 3.0f}));
        writer.write(static_cast<uint32_t>(0));
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto parseResult = wal::ParseEntry(reader);
        EXPECT_FALSE(parseResult.ok());
    }
}

TEST_F(WALTest, EntryWithAllFields) {
    wal::Entry entry{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 100,
        .txid = 200,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 5,
        .padding = 0,
        .embedding = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
        .payloadCRC = 0
    };
    entry.setVectorID("42");
    entry.payloadLength = entry.computePayloadLength();
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();

    EXPECT_EQ(entry.type, wal::OperationType::INSERT);
    EXPECT_EQ(entry.lsn, 100);
    EXPECT_EQ(entry.txid, 200);
    EXPECT_EQ(entry.getVectorID(), "42");
    EXPECT_EQ(entry.dimension, 5);
    EXPECT_EQ(entry.version, 1);
    EXPECT_EQ(entry.padding, 0);

    auto path = GetTestPath("entry_all_fields.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));
        EXPECT_TRUE(wal::WriteEntry(entry, writer).ok());
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto readEntryResult = wal::ParseEntry(reader);
        EXPECT_TRUE(readEntryResult.ok());
        auto& readEntry = readEntryResult.value();

        EXPECT_EQ(readEntry.type, entry.type);
        EXPECT_EQ(readEntry.lsn, entry.lsn);
        EXPECT_EQ(readEntry.txid, entry.txid);
        EXPECT_EQ(readEntry.getVectorID(), entry.getVectorID());
        EXPECT_EQ(readEntry.dimension, entry.dimension);
        EXPECT_EQ(readEntry.embedding, entry.embedding);
    }
}

TEST_F(WALTest, HeaderComputeCrc) {
    wal::Header header;
    header.magic = wal::kWalMagic;
    header.version = 1;
    header.flags = 0;
    header.creationTime = 1234567890;
    header.padding = 0;

    header.headerCrc32 = header.computeCrc32();
    EXPECT_NE(header.headerCrc32, 0u);

    auto path = GetTestPath("header_compute_crc.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));
        wal::WriteHeader(header, writer);
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto resHeaderResult = wal::ParseHeader(reader);
        ASSERT_TRUE(resHeaderResult.ok()) << resHeaderResult.status().message();
        EXPECT_EQ(resHeaderResult.value().headerCrc32, header.headerCrc32);
    }
}

TEST_F(WALTest, EntryRejectsLongVectorID) {
    wal::Entry entry;
    // Create a string that's exactly 128 bytes (1 byte too long)
    std::string longId(128, 'x');
    auto status = entry.setVectorID(longId);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kInvalidArgument);

    // Create a string that's 127 bytes (should be accepted)
    std::string maxId(127, 'y');
    status = entry.setVectorID(maxId);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(entry.getVectorID(), maxId);

    // Empty string should also work
    status = entry.setVectorID("");
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(entry.getVectorID(), "");
}

// =========================================================================
// WAL object tests (ported to WAL::open + log/readAll/truncate/recover API)
// =========================================================================

TEST_F(WALTest, WALLogCreatesDirectory) {
    auto walPath = GetTestPath("test_wal_dir");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok()) << walResult.status().message();
    auto wal = std::move(walResult.value());

    wal::Entry entry = CreateTestEntry();
    auto result = wal.log(entry);
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(std::filesystem::exists(walPath));
    EXPECT_TRUE(std::filesystem::is_directory(walPath));
    EXPECT_TRUE(std::filesystem::exists(walPath / "db.wal"));
}

TEST_F(WALTest, WALTruncateResetsEntries) {
    auto walPath = GetTestPath("truncate_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1, {1.0f, 2.0f, 3.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::DELETE, "2", 3, 2, 1, {4.0f, 5.0f, 6.0f});

    EXPECT_TRUE(wal.log(entry1).ok());

    // Truncate and write only entry2
    EXPECT_TRUE(wal.truncate().ok());
    EXPECT_TRUE(wal.log(entry2).ok());

    auto result = wal.readAll();
    ASSERT_TRUE(result.ok()) << result.status().message();
    auto& contents = result.value();
    EXPECT_EQ(contents.entries.size(), 1u);
    EXPECT_EQ(contents.entries[0].getVectorID(), "2");
}

TEST_F(WALTest, WALLogAppendMode) {
    auto walPath = GetTestPath("append_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1, {1.0f, 2.0f, 3.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::UPDATE, "2", 3, 2, 2, {4.0f, 5.0f, 6.0f});

    EXPECT_TRUE(wal.log(entry1).ok());
    EXPECT_TRUE(wal.log(entry2).ok());

    auto result = wal.readAll();
    ASSERT_TRUE(result.ok()) << result.status().message();
    auto& contents = result.value();
    EXPECT_EQ(contents.entries.size(), 2u);
    EXPECT_EQ(contents.entries[0].getVectorID(), "1");
    EXPECT_EQ(contents.entries[1].getVectorID(), "2");
}

TEST_F(WALTest, WALReadFirstEntry) {
    auto walPath = GetTestPath("read_entry_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry entry = CreateTestEntry(wal::OperationType::INSERT, "42", 2, 5, 10, {3.14f, 2.71f});
    EXPECT_TRUE(wal.log(entry).ok());

    auto readResult = wal.readAll();
    ASSERT_TRUE(readResult.ok());

    auto& entries = readResult.value().entries;
    EXPECT_EQ(entries.size(), 1u);
    const wal::Entry& readEntry = entries[0];
    EXPECT_EQ(readEntry.type, entry.type);
    EXPECT_EQ(readEntry.getVectorID(), entry.getVectorID());
    EXPECT_EQ(readEntry.dimension, entry.dimension);
    EXPECT_EQ(readEntry.embedding, entry.embedding);
}

TEST_F(WALTest, WALReadAllEntries) {
    auto walPath = GetTestPath("read_all_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    std::vector<wal::Entry> testEntries;
    testEntries.push_back(CreateTestEntry(wal::OperationType::INSERT, "1", 2, 1, 1, {1.0f, 2.0f}));
    testEntries.push_back(CreateTestEntry(wal::OperationType::UPDATE, "2", 2, 2, 2, {3.0f, 4.0f}));
    testEntries.push_back(CreateTestEntry(wal::OperationType::DELETE, "3", 2, 3, 3, {5.0f, 6.0f}));

    for (auto& e : testEntries) {
        EXPECT_TRUE(wal.log(e).ok());
    }

    auto readResult = wal.readAll();
    ASSERT_TRUE(readResult.ok());

    auto& entries = readResult.value().entries;
    EXPECT_EQ(entries.size(), 3u);

    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ(entries[i].getVectorID(), testEntries[i].getVectorID());
        EXPECT_EQ(entries[i].type, testEntries[i].type);
        EXPECT_EQ(entries[i].embedding, testEntries[i].embedding);
    }
}

TEST_F(WALTest, WALReadAllEmptyFile) {
    auto walPath = GetTestPath("empty_read_test");
    std::filesystem::create_directories(walPath);

    auto dbPath = walPath / "db.wal";
    std::ofstream file(dbPath, std::ios::binary);
    file.close();

    // Use the free function ReadAll directly on the empty file
    auto readResult = wal::ReadAll(dbPath);
    EXPECT_FALSE(readResult.ok());
    EXPECT_EQ(readResult.status().code(), arrow::utils::StatusCode::kEof);
}

TEST_F(WALTest, WALReadAllCorruptedEntry) {
    auto walPath = GetTestPath("corrupted_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry goodEntry = CreateTestEntry();
    EXPECT_TRUE(wal.log(goodEntry).ok());

    // Overwrite the WAL file with garbage
    auto dbPath = walPath / "db.wal";
    std::ofstream file(dbPath, std::ios::binary | std::ios::trunc);
    file.write("corrupted", 9);
    file.close();

    auto readResult = wal.readAll();
    EXPECT_FALSE(readResult.ok());
}

TEST_F(WALTest, WALLogCreatesParentDirectories) {
    auto walPath = GetTestPath("nonexistent_dir/subdir");

    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok()) << walResult.status().message();
    auto wal = std::move(walResult.value());

    wal::Entry entry = CreateTestEntry();
    EXPECT_TRUE(wal.log(entry).ok());

    EXPECT_TRUE(std::filesystem::exists(walPath));
    EXPECT_TRUE(std::filesystem::is_directory(walPath));
    EXPECT_TRUE(std::filesystem::exists(walPath / "db.wal"));
}

TEST_F(WALTest, WALReadFromNonexistentDirectory) {
    // Use the free function ReadAll on a nonexistent path
    auto readResult = wal::ReadAll("/nonexistent/directory/db.wal");
    EXPECT_FALSE(readResult.ok());
}

TEST_F(WALTest, WALRoundTripMultipleEntries) {
    auto walPath = GetTestPath("roundtrip_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    const size_t numEntries = 10;
    std::vector<wal::Entry> originalEntries;

    for (size_t i = 0; i < numEntries; ++i) {
        originalEntries.push_back(CreateTestEntry(wal::OperationType::INSERT, std::to_string(i),
            3,
            static_cast<uint64_t>(i + 1),
            static_cast<uint64_t>(i + 1)
        ));
    }

    for (auto& e : originalEntries) {
        EXPECT_TRUE(wal.log(e).ok());
    }

    auto readResult = wal.readAll();
    ASSERT_TRUE(readResult.ok());

    auto& readEntries = readResult.value().entries;
    EXPECT_EQ(readEntries.size(), numEntries);

    for (size_t i = 0; i < numEntries; ++i) {
        EXPECT_EQ(readEntries[i].getVectorID(), originalEntries[i].getVectorID());
        EXPECT_EQ(readEntries[i].type, originalEntries[i].type);
        EXPECT_EQ(readEntries[i].dimension, originalEntries[i].dimension);
        EXPECT_EQ(readEntries[i].embedding, originalEntries[i].embedding);
    }
}

TEST_F(WALTest, WALEmptyEmbedding) {
    auto walPath = GetTestPath("empty_embedding_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry entry{
        .type = wal::OperationType::DELETE,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 0,
        .padding = 0,
        .embedding = {},
        .payloadCRC = 0
    };
    entry.setVectorID("1");
    entry.payloadLength = entry.computePayloadLength();
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();

    EXPECT_TRUE(wal.log(entry).ok());

    auto readResult = wal.readAll();
    ASSERT_TRUE(readResult.ok());

    auto& entries = readResult.value().entries;
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].dimension, 0u);
    EXPECT_TRUE(entries[0].embedding.empty());
}

TEST_F(WALTest, WALLoadHeaderSuccess) {
    auto walPath = GetTestPath("read_header_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry entry = CreateTestEntry();
    auto res = wal.log(entry);
    ASSERT_TRUE(res.ok()) << res.message();

    auto headerResult = wal::LoadHeader(walPath);
    ASSERT_TRUE(headerResult.ok()) << headerResult.status().message();

    const wal::Header& header = headerResult.value();
    EXPECT_EQ(header.magic, wal::kWalMagic);
    EXPECT_EQ(header.version, 1);
}

TEST_F(WALTest, WALLoadHeaderEmptyFile) {
    auto walPath = GetTestPath("read_header_empty");
    std::filesystem::create_directories(walPath);

    auto dbPath = walPath / "db.wal";
    std::ofstream file(dbPath, std::ios::binary);
    file.close();

    auto headerResult = wal::LoadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
    EXPECT_EQ(headerResult.status().code(), arrow::utils::StatusCode::kBadHeader);
}

TEST_F(WALTest, WALLoadHeaderNonexistentDirectory) {
    auto headerResult = wal::LoadHeader("/nonexistent/directory");
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALLoadHeaderNonexistentFile) {
    auto walPath = GetTestPath("read_header_nonexistent_file");
    std::filesystem::create_directories(walPath);

    auto headerResult = wal::LoadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALLoadHeaderCorruptedMagic) {
    auto walPath = GetTestPath("read_header_corrupted_magic");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    {
        auto wal = std::move(walResult.value());
        wal::Entry entry = CreateTestEntry();
        EXPECT_TRUE(wal.log(entry).ok());
    }

    auto dbPath = walPath / "db.wal";
    std::fstream file(dbPath, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(0, std::ios::beg);
    uint32_t badMagic = 0xDEADBEEF;
    file.write(reinterpret_cast<char*>(&badMagic), sizeof(badMagic));
    file.close();

    auto headerResult = wal::LoadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALLoadHeaderTooSmall) {
    auto walPath = GetTestPath("read_header_too_small");
    std::filesystem::create_directories(walPath);

    auto dbPath = walPath / "db.wal";
    std::ofstream file(dbPath, std::ios::binary);
    uint32_t dummy = 0x12345678;
    file.write(reinterpret_cast<char*>(&dummy), sizeof(dummy));
    file.close();

    auto headerResult = wal::LoadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALReadEmptyFile) {
    auto walPath = GetTestPath("read_empty_file");
    std::filesystem::create_directories(walPath);

    auto dbPath = walPath / "db.wal";
    std::ofstream file(dbPath, std::ios::binary);
    file.close();

    auto readResult = wal::ReadAll(dbPath);
    EXPECT_FALSE(readResult.ok());
}

TEST_F(WALTest, WALReadFileTooSmallForHeader) {
    auto walPath = GetTestPath("read_too_small");
    std::filesystem::create_directories(walPath);

    auto dbPath = walPath / "db.wal";
    std::ofstream file(dbPath, std::ios::binary);
    file.write("abc", 3);
    file.close();

    auto readResult = wal::ReadAll(dbPath);
    EXPECT_FALSE(readResult.ok());
}

TEST_F(WALTest, WALPrintMethod) {
    auto walPath = GetTestPath("print_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 2, 1, 1, {1.0f, 2.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::UPDATE, "2", 2, 2, 2, {3.0f, 4.0f});

    EXPECT_TRUE(wal.log(entry1).ok());
    EXPECT_TRUE(wal.log(entry2).ok());

    EXPECT_NO_THROW(wal.print());
}

TEST_F(WALTest, WALTransactionTypes) {
    auto walPath = GetTestPath("txn_types_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry commitEntry{
        .type = wal::OperationType::COMMIT_TXN,
        .version = 1,
        .lsn = 1,
        .txid = 0,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 0,
        .padding = 0,
        .embedding = {},
        .payloadCRC = 0
    };
    commitEntry.setVectorID("");
    commitEntry.payloadLength = commitEntry.computePayloadLength();
    commitEntry.headerCRC = commitEntry.computeHeaderCrc();
    commitEntry.payloadCRC = commitEntry.computePayloadCrc();

    wal::Entry abortEntry{
        .type = wal::OperationType::ABORT_TXN,
        .version = 1,
        .lsn = 2,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 0,
        .padding = 0,
        .embedding = {},
        .payloadCRC = 0
    };
    abortEntry.setVectorID("");
    abortEntry.payloadLength = abortEntry.computePayloadLength();
    abortEntry.headerCRC = abortEntry.computeHeaderCrc();
    abortEntry.payloadCRC = abortEntry.computePayloadCrc();

    EXPECT_TRUE(wal.log(commitEntry).ok());
    EXPECT_TRUE(wal.log(abortEntry).ok());

    auto result = wal.readAll();
    ASSERT_TRUE(result.ok());
    auto& entries = result.value().entries;
    EXPECT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].type, wal::OperationType::COMMIT_TXN);
    EXPECT_EQ(entries[1].type, wal::OperationType::ABORT_TXN);
}

TEST_F(WALTest, WALBatchInsert) {
    auto walPath = GetTestPath("batch_insert_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    wal::Entry batchEntry{
        .type = wal::OperationType::BATCH_INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .dimension = 4,
        .padding = 0,
        .embedding = {1.0f, 2.0f, 3.0f, 4.0f},
        .payloadCRC = 0
    };
    batchEntry.setVectorID("0");
    batchEntry.payloadLength = batchEntry.computePayloadLength();
    batchEntry.headerCRC = batchEntry.computeHeaderCrc();
    batchEntry.payloadCRC = batchEntry.computePayloadCrc();

    EXPECT_TRUE(wal.log(batchEntry).ok());

    auto result = wal.readAll();
    ASSERT_TRUE(result.ok());
    auto& entries = result.value().entries;
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].type, wal::OperationType::BATCH_INSERT);
    EXPECT_EQ(entries[0].dimension, 4u);
}

TEST_F(WALTest, BatchLogMultipleEntries) {
    auto walPath = GetTestPath("batch_log_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    std::vector<wal::Entry> entries;
    for (uint64_t i = 0; i < 10; ++i) {
        wal::Entry entry{
            .type = wal::OperationType::INSERT,
            .version = 1,
            .lsn = i + 1,
            .txid = i + 1,
            .headerCRC = 0,
            .payloadLength = 0,
            .dimension = 128,
            .padding = 0,
            .embedding = std::vector<float>(128, 1.0f),
            .payloadCRC = 0
        };
        entry.setVectorID(std::to_string(i));
        entry.payloadLength = entry.computePayloadLength();
        entry.headerCRC = entry.computeHeaderCrc();
        entry.payloadCRC = entry.computePayloadCrc();
        entries.push_back(entry);
    }

    wal::Status status = wal.logBatch(entries);
    ASSERT_TRUE(status.ok()) << status.message();

    auto readResult = wal.readAll();
    ASSERT_TRUE(readResult.ok()) << readResult.status().message();
    const auto& readEntries = readResult.value().entries;
    EXPECT_EQ(readEntries.size(), 10u);

    for (size_t i = 0; i < readEntries.size(); ++i) {
        EXPECT_EQ(readEntries[i].getVectorID(), std::to_string(i));
        EXPECT_EQ(readEntries[i].lsn, i + 1);
        EXPECT_EQ(readEntries[i].txid, i + 1);
        EXPECT_EQ(readEntries[i].dimension, 128u);
    }
}

// =========================================================================
// Recovery tests
// =========================================================================

TEST_F(WALTest, CrashRecovery) {
    auto walPath = GetTestPath("crash_recovery");

    {
        auto walResult = wal::WAL::open(walPath);
        ASSERT_TRUE(walResult.ok());
        auto wal = std::move(walResult.value());

        wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1);
        wal::Entry entry2 = CreateTestEntry(wal::OperationType::INSERT, "2", 3, 2, 2);
        wal::Entry entry3 = CreateTestEntry(wal::OperationType::INSERT, "3", 3, 3, 3);

        EXPECT_TRUE(wal.log(entry1).ok());
        EXPECT_TRUE(wal.log(entry2).ok());
        EXPECT_TRUE(wal.log(entry3).ok());

        // Verify all 3 entries exist
        auto entriesBefore = wal.readAll();
        ASSERT_TRUE(entriesBefore.ok()) << entriesBefore.status().message();
        EXPECT_EQ(entriesBefore.value().entries.size(), 3u);
    }

    // Simulate crash by appending partial/corrupt data
    auto dbPath = walPath / "db.wal";
    {
        std::ofstream corruptFile(dbPath, std::ios::binary | std::ios::app);
        uint8_t partialData[10] = {0x03, 0x00, 0x01, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00};
        corruptFile.write(reinterpret_cast<char*>(partialData), 10);
    }

    // Reopen and recover
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    auto recoverResult = wal.recover();
    ASSERT_TRUE(recoverResult.ok()) << recoverResult.status().message();
    EXPECT_TRUE(recoverResult.value().truncationPerformed);

    // Verify only 3 valid entries remain
    auto entriesAfter = wal.readAll();
    ASSERT_TRUE(entriesAfter.ok());
    EXPECT_EQ(entriesAfter.value().entries.size(), 3u);
}

TEST_F(WALTest, PartialWriteRecovery) {
    auto walPath = GetTestPath("partial_write");

    // Create WAL and write one entry
    {
        auto walResult = wal::WAL::open(walPath);
        ASSERT_TRUE(walResult.ok());
        auto wal = std::move(walResult.value());

        wal::Entry entry = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1);
        EXPECT_TRUE(wal.log(entry).ok());
    }

    // Append partial entry (header only, no payload)
    {
        auto dbPath = walPath / "db.wal";
        std::ofstream ofs(dbPath, std::ios::binary | std::ios::app);
        uint8_t partialHeader[25] = {};
        partialHeader[0] = 0x03;  // INSERT type
        partialHeader[1] = 0x00;
        ofs.write(reinterpret_cast<char*>(partialHeader), 25);
    }

    // Reopen and recover
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    auto recoverResult = wal.recover();
    ASSERT_TRUE(recoverResult.ok());

    auto entries = wal.readAll();
    ASSERT_TRUE(entries.ok());
    EXPECT_EQ(entries.value().entries.size(), 1u);
    EXPECT_EQ(entries.value().entries[0].getVectorID(), "1");
}

TEST_F(WALTest, IdempotentReplay) {
    auto colPath = GetTestPath("idempotent_replay");

    // Create collection, insert data, and save (this also creates the WAL)
    {
        CollectionConfig config{.name = "test", .dimensions = 3, .space = Space::Cosine};
        Collection col(config, colPath);

        EXPECT_TRUE(col.insert("1", {1.0f, 2.0f, 3.0f}).ok());
        EXPECT_TRUE(col.insert("2", {4.0f, 5.0f, 6.0f}).ok());

        // Save creates checkpoint
        EXPECT_TRUE(col.save(colPath.string()).ok());
    }
    // Collection destructor releases file lock

    // Simulate dirty shutdown by setting cleanShutdown to false
    auto metaPath = colPath / "meta.json";
    {
        std::ifstream metaFile(metaPath);
        nlohmann::json meta;
        metaFile >> meta;
        metaFile.close();
        meta["recovery"]["cleanShutdown"] = false;
        std::ofstream outMetaFile(metaPath);
        outMetaFile << meta.dump(2);
    }

    // Load collection - should recover from WAL
    auto loadResult = Collection::load(colPath.string());
    ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
    Collection col2 = std::move(loadResult.value());

    // Verify collection has correct size (should not have duplicates)
    EXPECT_EQ(col2.size(), 2u);
}
