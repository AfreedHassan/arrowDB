#include "common.h"

TEST_F(WALWriterTest, OpenCreatesDir) {
    std::string dir = (testDir / "wal" / "subdir").string();
    auto result = wal::WAL::open(dir);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(std::filesystem::exists(testDir / "wal" / "subdir"));
}

TEST_F(WALWriterTest, OpenNewFile) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());

    auto& writer = result.value();
    auto entry = CreateTestEntry();
    auto status = writer.append(entry);
    EXPECT_TRUE(status.ok());

    writer.close();
}

TEST_F(WALWriterTest, OpenExistingFile) {
    std::string path = GetTestPath("test.wal");
    {
        auto result = wal::WALWriter::open(path);
        ASSERT_TRUE(result.ok());

        auto& writer = result.value();
        auto entry = CreateTestEntry(wal::OperationType::INSERT, "1", 3, 1);
        writer.append(entry);
        writer.close();
    }

    {
        auto result = wal::WALWriter::open(path);
        ASSERT_TRUE(result.ok());

        auto& writer = result.value();
        auto entry = CreateTestEntry(wal::OperationType::INSERT, "2", 3, 2);
        auto status = writer.append(entry);
        EXPECT_TRUE(status.ok());
        writer.close();
    }

    auto readResult = wal::ReadAll(path);
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 2);
}

TEST_F(WALWriterTest, AppendAndRead) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());
    auto& writer = result.value();

    auto entry = CreateTestEntry();
    auto status = writer.append(entry);
    ASSERT_TRUE(status.ok());
    writer.close();

    auto readResult = wal::ReadAll(path);
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 1);
}

TEST_F(WALWriterTest, AppendDeferredNoSync) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());
    auto& writer = result.value();

    for (int i = 0; i < 5; ++i) {
        auto entry = CreateTestEntry(wal::OperationType::INSERT, std::to_string(i), 3, i + 1);
        writer.appendDeferred(entry);
    }
    writer.close();

    auto readResult = wal::ReadAll(path);
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 5);
}

TEST_F(WALWriterTest, AppendBatch) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());
    auto& writer = result.value();

    std::vector<wal::Entry> batch;
    for (int i = 0; i < 5; ++i) {
        batch.push_back(CreateTestEntry(wal::OperationType::INSERT, std::to_string(i), 3, i + 1));
    }

    auto status = writer.appendBatch(batch);
    ASSERT_TRUE(status.ok());
    writer.close();

    auto readResult = wal::ReadAll(path);
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 5);
}

TEST_F(WALWriterTest, AppendNotOpen) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());

    auto& writer = result.value();
    writer.close();

    auto entry = CreateTestEntry();
    auto status = writer.append(entry);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), utils::StatusCode::kIoError);
}

TEST_F(WALWriterTest, AppendDeferredNotOpen) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());

    auto& writer = result.value();
    writer.close();

    auto entry = CreateTestEntry();
    auto status = writer.appendDeferred(entry);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), utils::StatusCode::kIoError);
}

TEST_F(WALWriterTest, AppendBatchNotOpen) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());

    auto& writer = result.value();
    writer.close();

    std::vector<wal::Entry> batch;
    auto status = writer.appendBatch(batch);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), utils::StatusCode::kIoError);
}

TEST_F(WALWriterTest, SyncNotOpen) {
    std::string path = GetTestPath("test.wal");
    auto result = wal::WALWriter::open(path);
    ASSERT_TRUE(result.ok());

    auto& writer = result.value();
    writer.close();

    auto status = writer.sync();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), utils::StatusCode::kIoError);
}

TEST_F(WALWriterTest, MoveConstructor) {
    auto result1 = wal::WALWriter::open(GetTestPath("test1.wal"));
    ASSERT_TRUE(result1.ok());

    wal::WALWriter writer2 = std::move(result1.value());

    auto entry = CreateTestEntry();
    auto status = writer2.append(entry);
    EXPECT_TRUE(status.ok());
    writer2.close();

    EXPECT_TRUE(std::filesystem::exists(GetTestPath("test1.wal")));
}

TEST_F(WALWriterTest, MoveAssignment) {
    auto result1 = wal::WALWriter::open(GetTestPath("test1.wal"));
    ASSERT_TRUE(result1.ok());

    auto result2 = wal::WALWriter::open(GetTestPath("test2.wal"));
    ASSERT_TRUE(result2.ok());

    // Move-assign writer1 into writer2
    result2.value() = std::move(result1.value());

    // writer2 should now write to test1.wal's file
    auto entry = CreateTestEntry();
    auto status = result2.value().append(entry);
    EXPECT_TRUE(status.ok());
    result2.value().close();

    auto readResult = wal::ReadAll(GetTestPath("test1.wal"));
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 1);
}

TEST_F(WALWriterTest, DestructorClosesFd) {
    std::string path = GetTestPath("test.wal");
    {
        auto result = wal::WALWriter::open(path);
        ASSERT_TRUE(result.ok());
        auto& writer = result.value();
        auto entry = CreateTestEntry();
        writer.appendDeferred(entry);
    }

    auto readResult = wal::ReadAll(path);
    ASSERT_TRUE(readResult.ok());
    EXPECT_EQ(readResult.value().entries.size(), 1);
}
