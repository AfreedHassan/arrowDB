#include "common.h"

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

TEST_F(WALTest, EntryToJsonAllTypes) {
    std::vector<std::pair<wal::OperationType, std::string>> typeExpected = {
        {wal::OperationType::INSERT, "INSERT"},
        {wal::OperationType::DELETE, "DELETE"},
        {wal::OperationType::UPDATE, "UPDATE"},
        {wal::OperationType::COMMIT_TXN, "COMMIT_TXN"},
        {wal::OperationType::ABORT_TXN, "ABORT_TXN"},
        {wal::OperationType::BATCH_INSERT, "BATCH_INSERT"},
    };

    for (auto& [opType, expectedStr] : typeExpected) {
        wal::Entry entry{
            .type = opType,
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
        entry.setVectorID("");
        auto j = entry.toJson();
        EXPECT_EQ(j["type"], expectedStr) << "Failed for type value " << static_cast<int>(opType);
    }
}

TEST_F(WALTest, EntryToJsonInvalidType) {
    wal::Entry entry{};
    entry.type = static_cast<wal::OperationType>(255);
    entry.setVectorID("");
    auto j = entry.toJson();
    EXPECT_EQ(j["type"], "INVALID");
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
        writer.write(static_cast<uint32_t>(128 + 4 + 1 + 3 * sizeof(float)));
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

TEST_F(WALTest, EntryRejectsLongVectorID) {
    wal::Entry entry;
    std::string longId(128, 'x');
    auto status = entry.setVectorID(longId);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kInvalidArgument);

    std::string maxId(127, 'y');
    status = entry.setVectorID(maxId);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(entry.getVectorID(), maxId);

    status = entry.setVectorID("");
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(entry.getVectorID(), "");
}

TEST_F(WALTest, IsEntryValidBadHeaderCrc) {
    auto entry = CreateTestEntry();
    entry.headerCRC = 0xDEADBEEF;
    auto status = wal::IsEntryValid(entry);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kChecksumMismatch);
}

TEST_F(WALTest, IsEntryValidBadPayloadCrc) {
    auto entry = CreateTestEntry();
    entry.payloadCRC = 0xDEADBEEF;
    auto status = wal::IsEntryValid(entry);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kChecksumMismatch);
}

TEST_F(WALTest, IsEntryValidDimensionMismatch) {
    auto entry = CreateTestEntry(wal::OperationType::INSERT, "1", 3);
    entry.dimension = 999;
    auto status = wal::IsEntryValid(entry);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kBadRecord);
}
