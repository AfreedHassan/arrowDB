#include <gtest/gtest.h>
#include "utils/crc32.h"
#include <cstring>
#include <string>

using namespace arrow::utils;

TEST(CRC32Test, EmptyInput) {
  uint32_t result = crc32(nullptr, 0);
  EXPECT_EQ(result, 0x00000000u);
}

TEST(CRC32Test, KnownVector) {
  // Standard CRC32 check value: CRC32("123456789") == 0xCBF43926
  const std::string data = "123456789";
  uint32_t result = crc32(data.data(), data.size());
  EXPECT_EQ(result, 0xCBF43926u);
}

TEST(CRC32Test, IncrementalConsistency) {
  // CRC of full buffer should equal CRC of two halves combined incrementally
  const std::string data = "Hello, World! This is a CRC32 test.";
  size_t half = data.size() / 2;

  uint32_t fullCrc = crc32(data.data(), data.size());

  // Compute incrementally: first half, then second half using first result as seed
  uint32_t partialCrc = crc32(data.data(), half);
  uint32_t incrementalCrc = crc32(data.data() + half, data.size() - half, partialCrc);

  EXPECT_EQ(fullCrc, incrementalCrc);
}
