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

    std::string GetTestPath(const std::string& dirname) {
        return (testDir / dirname).string();
    }

    wal::Entry CreateTestEntry(wal::OperationType type = wal::OperationType::INSERT, VectorID id = "1", uint32_t dim = 3, uint64_t lsn = 1, uint64_t txid = 1, const std::vector<float>& embedding = {}) {
        std::vector<float> vec = embedding.empty() ? RandomVector(dim, gen) : embedding;
        wal::Entry entry{
            .type = type,
            .version = 1,
            .lsn = lsn,
            .txid = txid,
            .headerCRC = 0,
            .payloadLength = static_cast<uint32_t>(vec.size() * sizeof(float)),
            .dimension = dim,
            .padding = 0,
            .embedding = vec,
            .payloadCRC = 0
        };
        entry.setVectorID(id);
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
    original.headerCrc32 = original.computeCrc32();  // Compute CRC before writing

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
            std::cout << "ParseHeader failed: " << result.error().message() << "\n";
        }
        ASSERT_TRUE(result.ok());
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
        .dimension = 3,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry.setVectorID("42");
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    entry.setVectorID("42");
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .dimension = 4,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    original.setVectorID("123");
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
        .payloadLength = static_cast<uint32_t>(embedding.size() * sizeof(float)),
        .dimension = 2,
        .padding = 0,
        .embedding = embedding,
        .payloadCRC = 0
    };
    original.setVectorID("1");
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
    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1, {1.0f, 2.0f, 3.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::DELETE, "2", 3, 2, 1, {4.0f, 5.0f, 6.0f});

    std::string walPath = GetTestPath("reset_test");

    EXPECT_TRUE(wal.log(entry1, walPath, true).ok());
    EXPECT_TRUE(wal.log(entry2, walPath, true).ok());

    auto result = wal.readAll(walPath);
    EXPECT_TRUE(result.ok());
    auto& entries = result.value();
    EXPECT_EQ(entries.size(), 1);
    EXPECT_EQ((entries[0]).getVectorID(), "2");
}

TEST_F(WALTest, WALLogAppendMode) {
    wal::WAL wal(testDir);
    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1, {1.0f, 2.0f, 3.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::UPDATE, "2", 3, 2, 2, {4.0f, 5.0f, 6.0f});

    std::string walPath = GetTestPath("append_test");

    EXPECT_TRUE(wal.log(entry1, walPath, true).ok());
    EXPECT_TRUE(wal.log(entry2, walPath, false).ok());

    auto result = wal.readAll(walPath);
    if (!result.ok()) {
        std::cerr << "readAll failed: " << result.error().message() << "\n";
    }
    EXPECT_TRUE(result.ok());
    auto& entries = result.value();
    EXPECT_EQ(entries.size(), 2);
    EXPECT_EQ((entries[0]).getVectorID(), "1");
    EXPECT_EQ((entries[1]).getVectorID(), "2");
}

TEST_F(WALTest, WALReadFirstEntry) {
    wal::WAL wal(testDir);
    wal::Entry entry = CreateTestEntry(wal::OperationType::INSERT, "42", 2, 5, 10, {3.14f, 2.71f});

    std::string walPath = GetTestPath("read_entry_test");
    wal.log(entry, walPath, true);

    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(readResult.ok());

    auto& entries = readResult.value();
    EXPECT_EQ(entries.size(), 1);
    const wal::Entry& readEntry = entries[0];
    EXPECT_EQ(readEntry.type, entry.type);
    EXPECT_EQ(readEntry.getVectorID(), entry.getVectorID());
    EXPECT_EQ(readEntry.dimension, entry.dimension);
    EXPECT_EQ(readEntry.embedding, entry.embedding);
}

TEST_F(WALTest, WALReadAllEntries) {
    wal::WAL wal(testDir);
    std::vector<wal::Entry> testEntries;
    testEntries.push_back(CreateTestEntry(wal::OperationType::INSERT, "1", 2, 1, 1, {1.0f, 2.0f}));
    testEntries.push_back(CreateTestEntry(wal::OperationType::UPDATE, "2", 2, 2, 2, {3.0f, 4.0f}));
    testEntries.push_back(CreateTestEntry(wal::OperationType::DELETE, "3", 2, 3, 3, {5.0f, 6.0f}));

    std::string walPath = GetTestPath("read_all_test");

    wal.log(testEntries[0], walPath, true);
    wal.log(testEntries[1], walPath, false);
    wal.log(testEntries[2], walPath, false);

    auto readResult = wal.readAll(walPath);
    EXPECT_TRUE(readResult.ok());

    auto& entries = readResult.value();
    EXPECT_EQ(entries.size(), 3);

    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ(entries[i].getVectorID(), testEntries[i].getVectorID());
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
    EXPECT_EQ(readResult.error().code(), arrow::utils::StatusCode::kEof);
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
        originalEntries.push_back(CreateTestEntry(wal::OperationType::INSERT, std::to_string(i),
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
        EXPECT_EQ(readEntries[i].getVectorID(), originalEntries[i].getVectorID());
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
        .dimension = 0,
        .padding = 0,
        .embedding = {},
        .payloadCRC = 0
    };
    entry.setVectorID("1");

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
        std::cout << "loadHeader failed: " << headerResult.error().message() << "\n";
    }
    EXPECT_TRUE(headerResult.ok());

    const wal::Header& header = headerResult.value();
    EXPECT_EQ(header.magic, wal::kWalMagic);
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
    EXPECT_EQ(headerResult.error().code(), arrow::utils::StatusCode::kBadHeader);
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

    wal::Header header{.magic = wal::kWalMagic, .version = 1, .flags = 0, .creationTime = 1234567890, .headerCrc32 = 0, .padding = 0};
    header.headerCrc32 = header.computeCrc32();  // Compute CRC before writing
    wal.writeHeader(header, walPath);
    wal.log(entry, walPath, true);

    auto headerResult = wal.loadHeader(walPath);
    if (!headerResult.ok()) {
        std::cout << "loadHeader failed: " << headerResult.error().message() << "\n";
    }
    EXPECT_TRUE(headerResult.ok());

    const wal::Header& res = headerResult.value();
    EXPECT_EQ(res.magic, wal::kWalMagic);
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
    EXPECT_EQ(header.magic, wal::kWalMagic);
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
    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 2, 1, 1, {1.0f, 2.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::UPDATE, "2", 2, 2, 2, {3.0f, 4.0f});

    wal.log(entry1, "", true);
    wal.log(entry2, "", false);

    EXPECT_NO_THROW(wal.print());
}

TEST_F(WALTest, WALTransactionTypes) {
    wal::WAL wal(testDir);

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
        .dimension = 4,
        .padding = 0,
        .embedding = {1.0f, 2.0f, 3.0f, 4.0f},
        .payloadCRC = 0
    };
    batchEntry.setVectorID("0");

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
        .dimension = 5,
        .padding = 0,
        .embedding = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
        .payloadCRC = 0
    };
    entry.setVectorID("42");

    EXPECT_EQ(entry.type, wal::OperationType::INSERT);
    EXPECT_EQ(entry.lsn, 100);
    EXPECT_EQ(entry.txid, 200);
    EXPECT_EQ(entry.getVectorID(), "42");
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
            std::cerr << resHeaderResult.error().message() << "\n";
        }
        EXPECT_EQ(resHeaderResult.value().headerCrc32, header.headerCrc32);
    }
}

TEST_F(WALTest, BatchLogMultipleEntries) {
  wal::WAL wal(testDir);

  // Create entries to batch log
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
    entry.headerCRC = entry.computeHeaderCrc();
    entry.payloadCRC = entry.computePayloadCrc();
    entry.payloadLength = entry.computePayloadLength();
    entries.push_back(entry);
  }

  // Batch log all entries (this initializes the WAL with header automatically)
  wal::Status status = wal.logBatch(entries);
  ASSERT_TRUE(status.ok()) << status.message();

  // Read all entries back
  auto readResult = wal.readAll();
  ASSERT_TRUE(readResult.ok()) << readResult.error().message();
  const auto& readEntries = readResult.value();
  EXPECT_EQ(readEntries.size(), 10);

  // Verify all entries were read correctly
  for (size_t i = 0; i < readEntries.size(); ++i) {
    EXPECT_EQ(readEntries[i].getVectorID(), std::to_string(i));
    EXPECT_EQ(readEntries[i].lsn, i + 1);
    EXPECT_EQ(readEntries[i].txid, i + 1);
    EXPECT_EQ(readEntries[i].dimension, 128);
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

TEST_F(WALTest, CrashRecovery) {
  std::string path = GetTestPath("crash_recovery");
  std::filesystem::create_directories(path);
  
  wal::WAL wal(path);
  
  // Write 3 valid entries (first one with reset=true to write header)
  wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1);
  wal::Entry entry2 = CreateTestEntry(wal::OperationType::INSERT, "2", 3, 2, 2);
  wal::Entry entry3 = CreateTestEntry(wal::OperationType::INSERT, "3", 3, 3, 3);
  
  EXPECT_TRUE(wal.log(entry1, path, true).ok());
  EXPECT_TRUE(wal.log(entry2, path, false).ok());
  EXPECT_TRUE(wal.log(entry3, path, false).ok());

  // Verify all 3 entries exist
  auto entriesBefore = wal.readAll();
  if (!entriesBefore.ok()) {
    std::cout << "readAll error: " << entriesBefore.error().message() << "\n";
  }
  ASSERT_TRUE(entriesBefore.ok());
  EXPECT_EQ(entriesBefore.value().size(), 3);
  
  // Simulate crash by corrupting/truncating file mid-entry
  // Append partial data (half an entry)
  std::ofstream corruptFile(path + "/db.wal", std::ios::binary | std::ios::app);
  uint8_t partialData[10] = {0x03, 0x00, 0x01, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00};
  corruptFile.write(reinterpret_cast<char*>(partialData), 10);
  corruptFile.close();
  
  // Call recover
  wal::Status recoverStatus = wal.recover();
  EXPECT_TRUE(recoverStatus.ok());
  
  // Verify only 3 valid entries remain
  auto entriesAfter = wal.readAll();
  ASSERT_TRUE(entriesAfter.ok());
  EXPECT_EQ(entriesAfter.value().size(), 3);
}

TEST_F(WALTest, PartialWriteRecovery) {
  std::string path = GetTestPath("partial_write");
  std::filesystem::create_directories(path);

  // Create WAL and write entry
  {
    wal::WAL wal(path);
    wal::Entry entry = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1);
    EXPECT_TRUE(wal.log(entry, path, true).ok());
  }
  
  // Append partial entry (header only, no payload)
  {
    std::ofstream ofs(path + "/db.wal", std::ios::binary | std::ios::app);
    uint8_t partialHeader[25] = {};  // Entry header without payload
    partialHeader[0] = 0x03;  // INSERT type
    partialHeader[1] = 0x00;
    ofs.write(reinterpret_cast<char*>(partialHeader), 25);
  }
  
  // Recover should truncate the partial entry
  wal::WAL wal(path);
  wal::Status recoverStatus = wal.recover();
  EXPECT_TRUE(recoverStatus.ok());
  
  auto entries = wal.readAll();
  ASSERT_TRUE(entries.ok());
  EXPECT_EQ(entries.value().size(), 1);
  EXPECT_EQ(entries.value()[0].getVectorID(), "1");
}

TEST_F(WALTest, IdempotentReplay) {
  std::string path = GetTestPath("idempotent_replay");
  std::filesystem::create_directories(path);

  // Create WAL and write entries first
  {
    wal::WAL wal(path);
    wal::Entry entry1 = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1, 1, {1.0f, 2.0f, 3.0f});
    wal::Entry entry2 = CreateTestEntry(wal::OperationType::INSERT, "2", 3, 2, 2, {4.0f, 5.0f, 6.0f});
    wal::Entry entry3 = CreateTestEntry(wal::OperationType::DELETE, "1", 3, 3, 3);

    EXPECT_TRUE(wal.log(entry1, path, true).ok());
    EXPECT_TRUE(wal.log(entry2, path, false).ok());
    EXPECT_TRUE(wal.log(entry3, path, false).ok());
  }

  // Create collection and insert the same data
  CollectionConfig config{.name = "test", .dimensions = 3, .space = Space::Cosine};
  Collection col(config, path);

  EXPECT_TRUE(col.insert("1", {1.0f, 2.0f, 3.0f}).ok());
  EXPECT_TRUE(col.insert("2", {4.0f, 5.0f, 6.0f}).ok());
  
  // Save creates checkpoint
  EXPECT_TRUE(col.save(path).ok());
  
  // Simulate dirty shutdown by setting cleanShutdown to false
  std::ifstream metaFile(path + "/meta.json");
  nlohmann::json meta;
  metaFile >> meta;
  metaFile.close();
  meta["recovery"]["cleanShutdown"] = false;
  std::ofstream outMetaFile(path + "/meta.json");
  outMetaFile << meta.dump(2);
  outMetaFile.close();
  
  // Load collection - should recover from WAL
  auto loadResult = Collection::load(path);
  ASSERT_TRUE(loadResult.ok());
  Collection col2 = std::move(loadResult.value());
  
  // Verify collection has correct size (should not have duplicates)
  EXPECT_EQ(col2.size(), 2);
}
