#include <gtest/gtest.h>
#include "arrow/wal.h"
#include "arrow/utils/binary.h"
#include "arrow/utils/status.h"
#include "test_util.h"
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

    std::string GetTestPath(const std::string& dirname) {
        return (testDir / dirname).string();
    }

    wal::Entry CreateTestEntry(wal::OperationType type = wal::OperationType::INSERT,
                               VectorID id = 1,
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
            .payloadLength = static_cast<uint32_t>(vec.size() * sizeof(float)),
            .vectorID = id,
            .dimension = dim,
            .padding = 0,
            .embedding = vec,
            .payloadCRC = 0
        };
        entry.headerCRC = entry.computeHeaderCrc();
        entry.payloadCRC = entry.computePayloadCrc();
        return entry;
    }
};

TEST_F(WALTest, HeaderDefaults) {
    wal::Header header;
    header.headerCrc32 = header.computeCrc32();
    EXPECT_EQ(sizeof(wal::Header), 24);
    EXPECT_EQ(sizeof(header), 24);
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
    EXPECT_EQ(header.flags, 0);
    EXPECT_EQ(header.creationTime, 0);
    uint32_t EXPECTEDCRC = 1956998465;
    EXPECT_EQ(header.headerCrc32, EXPECTEDCRC);
    EXPECT_EQ(header.padding, 0);
}

TEST_F(WALTest, HeaderWriteReadRoundTrip) {
    wal::Header original;
    original.magic = 0x41574C01;
    original.version = 2;
    original.flags = 0x1234;
    original.creationTime = 1234567890;
    original.padding = 0;

    std::string path = GetTestPath("header_roundtrip.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));
        wal::WriteHeader(original, writer);
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto result = wal::ParseHeader(reader);
        if (!result.ok()) {
            std::cout << "ParseHeader failed: " << result.status().message() << "\n";
        }
        ASSERT_TRUE(result.ok());
        const auto& read = result.value();

        EXPECT_EQ(read.magic, original.magic);
        EXPECT_EQ(read.version, original.version);
        EXPECT_EQ(read.flags, original.flags);
        EXPECT_EQ(read.creationTime, original.creationTime);
        EXPECT_EQ(read.padding, original.padding);
    }
}

TEST_F(WALTest, HeaderReadFailure) {
    std::string path = GetTestPath("header_empty.bin");
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .vectorID = 42,
        .dimension = 3,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    EXPECT_EQ(entry.type, wal::OperationType::INSERT);
    EXPECT_EQ(entry.lsn, 1);
    EXPECT_EQ(entry.txid, 42);
    EXPECT_EQ(entry.vectorID, 42);
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .vectorID = 42,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
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
    std::vector<float> embedding = {1.0f, 2.0f};
    wal::Entry entry1{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .vectorID = 1,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    wal::Entry entry2{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .vectorID = 1,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    wal::Entry entry3{
        .type = wal::OperationType::DELETE,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .vectorID = 1,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .vectorID = 123,
        .dimension = 4,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    original.headerCRC = original.computeHeaderCrc();
    original.payloadCRC = original.computePayloadCrc();

    std::string path = GetTestPath("entry_roundtrip.bin");
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
        EXPECT_EQ(read.vectorID, original.vectorID);
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .vectorID = 1,
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    original.headerCRC = original.computeHeaderCrc();
    original.payloadCRC = original.computePayloadCrc();

    std::string path = GetTestPath("entry_crc_mismatch.bin");
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
    std::string path = GetTestPath("entry_dimension_mismatch.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

        writer.write(wal::OperationType::INSERT);
        writer.write(static_cast<uint16_t>(1));
        writer.write(static_cast<uint64_t>(1));
        writer.write(static_cast<uint64_t>(1));
        writer.write(static_cast<uint32_t>(0));
        writer.write(static_cast<uint32_t>(3 * sizeof(float)));
        writer.write(static_cast<VectorID>(1));
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

TEST_F(WALTest, WALLogCreatesDirectory) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry();

    std::string walPath = GetTestPath("test_wal_dir");
    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    auto result = wal.log(entry, walPath, true);
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(std::filesystem::exists(walPath));
    EXPECT_TRUE(std::filesystem::is_directory(walPath));
    EXPECT_TRUE(std::filesystem::exists(dbPath));
}

TEST_F(WALTest, WALLogResetMode) {
    wal::WAL wal(testDir);
    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, 1, 3, 1, 1, {1.0f, 2.0f, 3.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::DELETE, 2, 3, 2, 1, {4.0f, 5.0f, 6.0f});

    std::string walPath = GetTestPath("reset_test");

    EXPECT_TRUE(wal.log(entry1, walPath, true).ok());
    EXPECT_TRUE(wal.log(entry2, walPath, true).ok());

    auto result = wal.readAll(walPath);
    EXPECT_TRUE(result.ok());
    auto& entries = result.value();
    EXPECT_EQ(entries.size(), 1);
    EXPECT_EQ((entries[0]).vectorID, 2);
}

TEST_F(WALTest, WALLogAppendMode) {
    wal::WAL wal(testDir);
    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, 1, 3, 1, 1, {1.0f, 2.0f, 3.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::UPDATE, 2, 3, 2, 2, {4.0f, 5.0f, 6.0f});

    std::string walPath = GetTestPath("append_test");

    EXPECT_TRUE(wal.log(entry1, walPath, true).ok());
    EXPECT_TRUE(wal.log(entry2, walPath, false).ok());

    auto result = wal.readAll(walPath);
    if (!result.ok()) {
        std::cerr << "readAll failed: " << result.status().message() << "\n";
    }
    EXPECT_TRUE(result.ok());
    auto& entries = result.value();
    EXPECT_EQ(entries.size(), 2);
    EXPECT_EQ((entries[0]).vectorID, 1);
    EXPECT_EQ((entries[1]).vectorID, 2);
}

TEST_F(WALTest, WALReadFirstEntry) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry(wal::OperationType::INSERT, 42, 2, 5, 10, {3.14f, 2.71f});

    std::string walPath = GetTestPath("read_entry_test");
    wal.log(entry, walPath, true);

    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(readResult.ok());

    auto& entries = readResult.value();
    EXPECT_EQ(entries.size(), 1);
    const wal::Entry& readEntry = entries[0];
    EXPECT_EQ(readEntry.type, entry.type);
    EXPECT_EQ(readEntry.vectorID, entry.vectorID);
    EXPECT_EQ(readEntry.dimension, entry.dimension);
    EXPECT_EQ(readEntry.embedding, entry.embedding);
}

TEST_F(WALTest, WALReadAllEntries) {
    wal::WAL wal(testDir);
    std::vector<wal::Entry> testEntries;
    testEntries.push_back(CreateTestEntry(wal::OperationType::INSERT, 1, 2, 1, 1, {1.0f, 2.0f}));
    testEntries.push_back(CreateTestEntry(wal::OperationType::UPDATE, 2, 2, 2, 2, {3.0f, 4.0f}));
    testEntries.push_back(CreateTestEntry(wal::OperationType::DELETE, 3, 2, 3, 3, {5.0f, 6.0f}));

    std::string walPath = GetTestPath("read_all_test");

    wal.log(testEntries[0], walPath, true);
    wal.log(testEntries[1], walPath, false);
    wal.log(testEntries[2], walPath, false);

    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(readResult.ok());

    auto& entries = readResult.value();
    EXPECT_EQ(entries.size(), 3);

    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ(entries[i].vectorID, testEntries[i].vectorID);
        EXPECT_EQ(entries[i].type, testEntries[i].type);
        EXPECT_EQ(entries[i].embedding, testEntries[i].embedding);
    }
}

TEST_F(WALTest, WALReadAllEmptyFile) {
    wal::WAL wal(testDir);

    std::string walPath = GetTestPath("empty_read_test");
    std::filesystem::create_directories(walPath);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();

    auto readResult = wal.readAll(walPath);
    EXPECT_FALSE(readResult.ok());
    EXPECT_EQ(readResult.status().code(), arrow::utils::StatusCode::kEof);
}

TEST_F(WALTest, WALReadAllCorruptedEntry) {
    wal::WAL wal(testDir);
    wal::Entry goodEntry = CreateTestEntry();

    std::string walPath = GetTestPath("corrupted_test");
    wal.log(goodEntry, walPath, true);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary | std::ios::trunc);
    file.write("corrupted", 9);
    file.close();

    auto readResult = wal.readAll(walPath);
    EXPECT_FALSE(readResult.ok());
}

TEST_F(WALTest, WALLogCreatesParentDirectories) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry();

    std::string walPath = GetTestPath("nonexistent_dir/subdir");

    EXPECT_TRUE(wal.log(entry, walPath, true).ok());

    EXPECT_TRUE(std::filesystem::exists(walPath));
    EXPECT_TRUE(std::filesystem::is_directory(walPath));

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    EXPECT_TRUE(std::filesystem::exists(dbPath));
}

TEST_F(WALTest, WALReadFromNonexistentDirectory) {
    wal::WAL wal(testDir);

    auto readResult = wal.readAll("/nonexistent/directory");
    EXPECT_FALSE(readResult.ok());
}

TEST_F(WALTest, WALRoundTripMultipleEntries) {
    wal::WAL wal(testDir);
    const size_t numEntries = 10;
    std::vector<wal::Entry> originalEntries;

    for (size_t i = 0; i < numEntries; ++i) {
        originalEntries.push_back(CreateTestEntry(
            wal::OperationType::INSERT,
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
    EXPECT_TRUE(readResult.ok());

    auto& readEntries = readResult.value();
    EXPECT_EQ(readEntries.size(), numEntries);

    for (size_t i = 0; i < numEntries; ++i) {
        EXPECT_EQ(readEntries[i].vectorID, originalEntries[i].vectorID);
        EXPECT_EQ(readEntries[i].type, originalEntries[i].type);
        EXPECT_EQ(readEntries[i].dimension, originalEntries[i].dimension);
        EXPECT_EQ(readEntries[i].embedding, originalEntries[i].embedding);
    }
}

TEST_F(WALTest, WALEmptyEmbedding) {
    wal::WAL wal(testDir);
    wal::Entry entry{
        .type = wal::OperationType::DELETE,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = 0,
        .vectorID = 1,
        .dimension = 0,
        .padding = 0,
        .embedding = {},
        .payloadCRC = 0
    };

    std::string walPath = GetTestPath("empty_embedding_test");
    EXPECT_TRUE(wal.log(entry, walPath, true).ok());

    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(readResult.ok());

    auto& entries = readResult.value();
    const wal::Entry& readEntry = entries[0];
    EXPECT_EQ(readEntry.dimension, 0);
    EXPECT_TRUE(readEntry.embedding.empty());
}

TEST_F(WALTest, WALReadHeaderSuccess) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry();

    std::string walPath = GetTestPath("read_header_test");
    auto res = wal.log(entry, walPath, true);
    if (!res.ok()) {
        std::cout << "log failed: " << res.message() << "\n";
    }

    auto headerResult = wal.loadHeader(walPath);
    if (!headerResult.ok()) {
        std::cout << "loadHeader failed: " << headerResult.status().message() << "\n";
    }
    EXPECT_TRUE(headerResult.ok());

    const wal::Header& header = headerResult.value();
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
}

TEST_F(WALTest, WALReadHeaderEmptyFile) {
    wal::WAL wal(testDir);

    std::string walPath = GetTestPath("read_header_empty");
    std::filesystem::create_directories(walPath);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();

    auto headerResult = wal.loadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
    EXPECT_EQ(headerResult.status().code(), arrow::utils::StatusCode::kBadHeader);
}

TEST_F(WALTest, WALReadHeaderNonexistentDirectory) {
    wal::WAL wal(testDir);

    auto headerResult = wal.loadHeader("/nonexistent/directory");
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALReadHeaderNonexistentFile) {
    wal::WAL wal(testDir);

    std::string walPath = GetTestPath("read_header_nonexistent_file");
    std::filesystem::create_directories(walPath);

    auto headerResult = wal.loadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALReadHeaderCorruptedMagic) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry();

    std::string walPath = GetTestPath("read_header_corrupted_magic");
    wal.log(entry, walPath, true);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::fstream file(dbPath, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(0, std::ios::beg);
    uint32_t badMagic = 0xDEADBEEF;
    file.write(reinterpret_cast<char*>(&badMagic), sizeof(badMagic));
    file.close();

    auto headerResult = wal.loadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALReadHeaderTooSmall) {
    wal::WAL wal(testDir);

    std::string walPath = GetTestPath("read_header_too_small");
    std::filesystem::create_directories(walPath);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    uint32_t dummy = 0x12345678;
    file.write(reinterpret_cast<char*>(&dummy), sizeof(dummy));
    file.close();

    auto headerResult = wal.loadHeader(walPath);
    EXPECT_FALSE(headerResult.ok());
}

TEST_F(WALTest, WALReadHeaderRoundTrip) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry();

    std::string walPath = GetTestPath("read_header_roundtrip");

    wal::Header header{.magic = 0x41574C01, .version = 1, .flags = 0, .creationTime = 1234567890, .headerCrc32 = 0, .padding = 0};
    wal.writeHeader(header, walPath);
    wal.log(entry, walPath, true);

    auto headerResult = wal.loadHeader(walPath);
    if (!headerResult.ok()) {
        std::cout << "loadHeader failed: " << headerResult.status().message() << "\n";
    }
    EXPECT_TRUE(headerResult.ok());

    const wal::Header& res = headerResult.value();
    EXPECT_EQ(res.magic, 0x41574C01);
    EXPECT_EQ(res.version, 1);
}

TEST_F(WALTest, DISABLED_WALReadHeaderWithStream) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry();

    std::string walPath = GetTestPath("read_header_stream");
    wal.log(entry, walPath, true);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ifstream file(dbPath, std::ios::binary);

    auto headerResult = wal::LoadHeader(walPath);
    EXPECT_TRUE(headerResult.ok());

    const wal::Header& header = headerResult.value();
    EXPECT_EQ(header.magic, 0x41574C01);
    EXPECT_EQ(header.version, 1);
}

TEST_F(WALTest, WALReadEmptyFile) {
    wal::WAL wal(testDir);

    std::string walPath = GetTestPath("read_empty_file");
    std::filesystem::create_directories(walPath);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.close();

    auto readResult = wal.readAll(walPath);
    EXPECT_FALSE(readResult.ok());
}

TEST_F(WALTest, WALReadFileTooSmallForHeader) {
    wal::WAL wal(testDir);

    std::string walPath = GetTestPath("read_too_small");
    std::filesystem::create_directories(walPath);

    std::string dbPath = (std::filesystem::path(walPath) / "db.wal").string();
    std::ofstream file(dbPath, std::ios::binary);
    file.write("abc", 3);
    file.close();

    auto readResult = wal.readAll(walPath);
    EXPECT_FALSE(readResult.ok());

    auto readAllResult = wal.readAll(walPath);
    EXPECT_FALSE(readAllResult.ok());
}

TEST_F(WALTest, WALPrintMethod) {
    wal::WAL wal(testDir);
    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, 1, 2, 1, 1, {1.0f, 2.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::UPDATE, 2, 2, 2, 2, {3.0f, 4.0f});

    wal.log(entry1, "", true);
    wal.log(entry2, "", false);

    EXPECT_NO_THROW(wal.print());
}

TEST_F(WALTest, WALTransactionTypes) {
    wal::WAL wal(testDir);

    wal::Entry commitEntry(wal::OperationType::COMMIT_TXN, 1, 1, 0, 0, {});
    wal::Entry abortEntry(wal::OperationType::ABORT_TXN, 2, 1, 0, 0, {});

    std::string walPath = GetTestPath("txn_types_test");

    EXPECT_TRUE(wal.log(commitEntry, walPath, true).ok());
    EXPECT_TRUE(wal.log(abortEntry, walPath, false).ok());

    auto result = wal.readAll(walPath);
    EXPECT_TRUE(result.ok());
    auto& entries = result.value();
    EXPECT_EQ(entries.size(), 2);
    EXPECT_EQ(entries[0].type, wal::OperationType::COMMIT_TXN);
    EXPECT_EQ(entries[1].type, wal::OperationType::ABORT_TXN);
}

TEST_F(WALTest, WALBatchInsert) {
    wal::WAL wal(testDir);
    wal::Entry batchEntry{
        .type = wal::OperationType::BATCH_INSERT,
        .version = 1,
        .lsn = 1,
        .txid = 1,
        .headerCRC = 0,
        .payloadLength = static_cast<uint32_t>(4 * sizeof(float)),
        .vectorID = 0,
        .dimension = 4,
        .padding = 0,
        .embedding = {1.0f, 2.0f, 3.0f, 4.0f},
        .payloadCRC = 0
    };

    std::string walPath = GetTestPath("batch_insert_test");
    EXPECT_TRUE(wal.log(batchEntry, walPath, true).ok());

    auto result = wal.readAll(walPath);
    EXPECT_TRUE(result.ok());
    auto& entries = result.value();
    const wal::Entry& entry = entries[0];
    EXPECT_EQ(entry.type, wal::OperationType::BATCH_INSERT);
    EXPECT_EQ(entry.dimension, 4);
}

TEST_F(WALTest, EntryWithAllFields) {
    wal::Entry entry{
        .type = wal::OperationType::INSERT,
        .version = 1,
        .lsn = 100,
        .txid = 200,
        .headerCRC = 0,
        .payloadLength = static_cast<uint32_t>(5 * sizeof(float)),
        .vectorID = 42,
        .dimension = 5,
        .padding = 0,
        .embedding = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
        .payloadCRC = 0
    };

    EXPECT_EQ(entry.type, wal::OperationType::INSERT);
    EXPECT_EQ(entry.lsn, 100);
    EXPECT_EQ(entry.txid, 200);
    EXPECT_EQ(entry.vectorID, 42);
    EXPECT_EQ(entry.dimension, 5);
    EXPECT_EQ(entry.version, 1);
    EXPECT_EQ(entry.padding, 0);

    std::string path = GetTestPath("entry_all_fields.bin");
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
        EXPECT_EQ(readEntry.vectorID, entry.vectorID);
        EXPECT_EQ(readEntry.dimension, entry.dimension);
        EXPECT_EQ(readEntry.embedding, entry.embedding);
    }
}

TEST_F(WALTest, HeaderComputeCrc) {
    wal::Header header;
    header.magic = 0x41574C01;
    header.version = 1;
    header.flags = 0;
    header.creationTime = 1234567890;
    header.padding = 0;

    header.headerCrc32 = header.computeCrc32();
    EXPECT_NE(header.headerCrc32, 0);

    std::string path = GetTestPath("header_compute_crc.bin");
    {
        std::ofstream file(path, std::ios::binary);
        BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));
        wal::WriteHeader(header, writer);
    }

    {
        std::ifstream file(path, std::ios::binary);
        BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));
        auto resHeaderResult = wal::ParseHeader(reader);
        if (!resHeaderResult.ok()) {
            std::cerr << resHeaderResult.status().message() << "\n";
        }
        EXPECT_EQ(resHeaderResult.value().headerCrc32, header.headerCrc32);
    }
}
