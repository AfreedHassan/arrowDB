#include "common.h"
#include <gtest/gtest.h>

using namespace arrow;
using namespace arrow::wal;

// Helper to write a valid WAL header
static void writeValidHeader(BinaryWriter& w) {
  Header h;
  h.magic = kWalMagic;
  h.version = 1;
  h.flags = 0;
  h.creationTime = 0;
  h.headerCrc32 = h.computeCrc32();
  h.padding = 0;
  WriteHeader(h, w);
}

// ── ParseEntry with invalid operation type ──────────────────

TEST_F(WALTest, ParseEntryInvalidOpType) {
  auto walPath = GetTestPath("invalid_op_test");
  std::filesystem::create_directories(walPath);
  auto dbPath = walPath / "db.wal";

  {
    auto writer = std::make_unique<std::ofstream>(dbPath, std::ios::binary);
    BinaryWriter w(std::move(writer));
    writeValidHeader(w);

    // Write entry with invalid operation type (0xFF, valid range is 1-6)
    uint16_t badType = 0x00FF;
    w.write(badType);
    uint16_t version = 1;
    w.write(version);
    uint64_t lsn = 1, txid = 1;
    w.write(lsn);
    w.write(txid);
  }

  auto result = ReadAll(dbPath);
  EXPECT_FALSE(result.ok());
}

// ── ParseEntry with oversized dimension ─────────────────────

TEST_F(WALTest, ParseEntryOversizedDimension) {
  auto walPath = GetTestPath("oversized_dim_test");
  std::filesystem::create_directories(walPath);
  auto dbPath = walPath / "db.wal";

  {
    auto writer = std::make_unique<std::ofstream>(dbPath, std::ios::binary);
    BinaryWriter w(std::move(writer));
    writeValidHeader(w);

    uint16_t type = 3;  // INSERT
    w.write(type);
    uint16_t version = 1;
    w.write(version);
    uint64_t lsn = 1, txid = 1;
    w.write(lsn);
    w.write(txid);
    uint32_t headerCRC = 0;
    w.write(headerCRC);
    uint32_t payloadLength = 0;
    w.write(payloadLength);
    char vectorID[128] = {};
    vectorID[0] = '1';
    w.write(vectorID);
    uint32_t dimension = 100000;  // > kMaxDimension (65536)
    w.write(dimension);
    uint8_t padding = 0;
    w.write(padding);
  }

  auto result = ReadAll(dbPath);
  EXPECT_FALSE(result.ok());
}

// ── ParseEntry with truncated embedding ─────────────────────

TEST_F(WALTest, ParseEntryTruncatedEmbedding) {
  auto walPath = GetTestPath("truncated_embed_test");
  std::filesystem::create_directories(walPath);
  auto dbPath = walPath / "db.wal";

  {
    auto writer = std::make_unique<std::ofstream>(dbPath, std::ios::binary);
    BinaryWriter w(std::move(writer));
    writeValidHeader(w);

    uint16_t type = 3;  // INSERT
    w.write(type);
    uint16_t version = 1;
    w.write(version);
    uint64_t lsn = 1, txid = 1;
    w.write(lsn);
    w.write(txid);
    uint32_t headerCRC = 0;
    w.write(headerCRC);
    uint32_t payloadLength = 0;
    w.write(payloadLength);
    char vectorID[128] = {};
    vectorID[0] = '1';
    w.write(vectorID);
    uint32_t dimension = 100;  // Claims 100 floats = 400 bytes
    w.write(dimension);
    uint8_t padding = 0;
    w.write(padding);
    // File ends here — no embedding data written
  }

  auto result = ReadAll(dbPath);
  EXPECT_FALSE(result.ok());
}
