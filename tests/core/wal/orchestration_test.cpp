#include "common.h"

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

TEST_F(WALTest, WALLogDeferredAndSync) {
    auto walPath = GetTestPath("deferred_sync_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    for (int i = 0; i < 3; ++i) {
        auto entry = CreateTestEntry(wal::OperationType::INSERT,
            std::to_string(i), 3, i + 1, i + 1);
        EXPECT_TRUE(wal.logDeferred(entry).ok());
    }

    EXPECT_TRUE(wal.sync().ok());

    auto readResult = wal.readAll();
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 3u);
}

TEST_F(WALTest, WALLogNullWriter) {
    auto walPath = GetTestPath("null_writer_test");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    // Truncate closes writer, then truncate reopens — but if we manually
    // move the wal, the writer is null. Test via moved-from.
    wal::WAL movedWal = std::move(wal);

    // The moved-from WAL should have null writer — log should fail
    auto entry = CreateTestEntry();
    auto status = wal.log(entry);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kIoError);
}

TEST_F(WALTest, WALMoveAssignment) {
    auto walPath1 = GetTestPath("move_assign_1");
    auto walResult1 = wal::WAL::open(walPath1);
    ASSERT_TRUE(walResult1.ok());
    auto wal1 = std::move(walResult1.value());

    auto walPath2 = GetTestPath("move_assign_2");
    auto walResult2 = wal::WAL::open(walPath2);
    ASSERT_TRUE(walResult2.ok());
    auto wal2 = std::move(walResult2.value());

    // Move-assign wal1 into wal2
    wal2 = std::move(wal1);

    // wal2 should now be functional (points to walPath1)
    auto entry = CreateTestEntry();
    auto status = wal2.log(entry);
    EXPECT_TRUE(status.ok());

    auto readResult = wal2.readAll();
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 1u);
}

TEST_F(WALTest, WALTruncateNoFile) {
    auto walPath = GetTestPath("truncate_no_file");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    // Truncate immediately (file exists from open)
    auto status = wal.truncate();
    EXPECT_TRUE(status.ok());

    // Can still write after truncate
    auto entry = CreateTestEntry();
    EXPECT_TRUE(wal.log(entry).ok());

    auto readResult = wal.readAll();
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 1u);
}
