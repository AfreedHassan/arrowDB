#include "common.h"

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

TEST_F(WALTest, HeaderToJson) {
    wal::Header header;
    header.magic = wal::kWalMagic;
    header.version = 1;
    header.flags = 0;
    header.creationTime = 1234567890;
    header.padding = 0;
    header.headerCrc32 = header.computeCrc32();

    auto j = header.toJson();
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j["magic"], wal::kWalMagic);
    EXPECT_EQ(j["version"], 1);
    EXPECT_EQ(j["flags"], 0);
    EXPECT_EQ(j["creationTime"], 1234567890);
}

TEST_F(WALTest, IsHeaderValidBadMagic) {
    wal::Header h;
    h.magic = 0xDEADBEEF;
    h.headerCrc32 = h.computeCrc32();
    auto status = wal::IsHeaderValid(h);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kBadHeader);
}

TEST_F(WALTest, IsHeaderValidBadCrc) {
    wal::Header h;
    h.magic = wal::kWalMagic;
    h.headerCrc32 = 0xDEADBEEF;
    auto status = wal::IsHeaderValid(h);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), arrow::utils::StatusCode::kChecksumMismatch);
}
