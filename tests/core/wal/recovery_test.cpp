#include "common.h"
#include <nlohmann/json.hpp>

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
        auto createResult = Collection::create(config, colPath);
        ASSERT_TRUE(createResult.ok()) << createResult.status().message();
        Collection col = std::move(createResult.value());

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

TEST_F(WALTest, RecoverCorruptHeader) {
    auto walPath = GetTestPath("recover_corrupt_header");

    // Create valid WAL first
    {
        auto walResult = wal::WAL::open(walPath);
        ASSERT_TRUE(walResult.ok());
        auto wal = std::move(walResult.value());
        auto entry = CreateTestEntry();
        EXPECT_TRUE(wal.log(entry).ok());
    }

    // Corrupt the header
    auto dbPath = walPath / "db.wal";
    {
        std::ofstream ofs(dbPath, std::ios::binary | std::ios::trunc);
        ofs.write("GARBAGE_HEADER_DATA_1234", 24);
    }

    // Reopen and recover
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    auto recoverResult = wal.recover();
    ASSERT_TRUE(recoverResult.ok());
    EXPECT_TRUE(recoverResult.value().truncationPerformed);
    EXPECT_GT(recoverResult.value().discardedBytes, 0u);

    // After recovery, WAL should be usable
    auto entry = CreateTestEntry();
    EXPECT_TRUE(wal.log(entry).ok());
}

TEST_F(WALTest, RecoverEmptyFile) {
    auto walPath = GetTestPath("recover_empty");
    std::filesystem::create_directories(walPath);

    // Create empty WAL file
    auto dbPath = walPath / "db.wal";
    std::ofstream(dbPath, std::ios::binary).close();

    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    auto recoverResult = wal.recover();
    ASSERT_TRUE(recoverResult.ok());
    EXPECT_EQ(recoverResult.value().validEntries, 0u);
}

TEST_F(WALTest, RecoverNonexistentFile) {
    auto walPath = GetTestPath("recover_nonexistent");
    std::filesystem::create_directories(walPath);
    // Don't create db.wal

    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    // The open created the file. Remove it to simulate nonexistent.
    std::filesystem::remove(walPath / "db.wal");

    auto recoverResult = wal.recover();
    ASSERT_TRUE(recoverResult.ok());
    EXPECT_EQ(recoverResult.value().validEntries, 0u);
    EXPECT_FALSE(recoverResult.value().truncationPerformed);
}

TEST_F(WALTest, RecoverCleanFile) {
    auto walPath = GetTestPath("recover_clean");
    auto walResult = wal::WAL::open(walPath);
    ASSERT_TRUE(walResult.ok());
    auto wal = std::move(walResult.value());

    // Write 3 valid entries
    for (int i = 0; i < 3; ++i) {
        auto entry = CreateTestEntry(wal::OperationType::INSERT,
            std::to_string(i), 3, i + 1, i + 1);
        EXPECT_TRUE(wal.log(entry).ok());
    }

    auto recoverResult = wal.recover();
    ASSERT_TRUE(recoverResult.ok());
    EXPECT_EQ(recoverResult.value().validEntries, 3u);
    EXPECT_FALSE(recoverResult.value().truncationPerformed);
}
