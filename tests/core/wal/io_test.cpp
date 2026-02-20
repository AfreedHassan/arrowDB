#include "common.h"

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

TEST_F(WALTest, OpenBinaryReaderNotADirectory) {
    auto filePath = GetTestPath("not_a_dir.txt");
    std::ofstream(filePath) << "just a file";

    auto result = wal::OpenBinaryReader(filePath, "db.wal");
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), arrow::utils::StatusCode::kNotFound);
}

TEST_F(WALTest, OpenBinaryWriterNotADirectory) {
    auto filePath = GetTestPath("not_a_dir.txt");
    std::ofstream(filePath) << "just a file";

    auto result = wal::OpenBinaryWriter(filePath, "db.wal", false);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), arrow::utils::StatusCode::kIoError);
}

TEST_F(WALTest, OpenBinaryWriterCreateDir) {
    auto dirPath = GetTestPath("new_writer_dir/subdir");
    auto result = wal::OpenBinaryWriter(dirPath, "db.wal", false);
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(std::filesystem::exists(dirPath / "db.wal"));
}

TEST_F(WALTest, ReadAllNonexistent) {
    auto result = wal::ReadAll("/nonexistent/path/db.wal");
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), arrow::utils::StatusCode::kNotFound);
}
