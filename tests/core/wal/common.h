#ifndef TESTS_CORE_WAL_COMMON_H_
#define TESTS_CORE_WAL_COMMON_H_

#include "wal/wal.h"
#include "wal/wal_writer.h"
#include "wal/binary.h"
#include "arrow/utils/status.h"
#include "arrow/collection.h"
#include "test_util.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <random>
#include <gtest/gtest.h>

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

    std::filesystem::path GetTestPath(const std::string& dirname) {
        return testDir / dirname;
    }

    wal::Entry CreateTestEntry(wal::OperationType type = wal::OperationType::INSERT,
                               const std::string& id = "1",
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
            .payloadLength = 0,
            .dimension = dim,
            .padding = 0,
            .embedding = vec,
            .payloadCRC = 0
        };
        entry.setVectorID(id);
        entry.payloadLength = entry.computePayloadLength();
        entry.headerCRC = entry.computeHeaderCrc();
        entry.payloadCRC = entry.computePayloadCrc();
        return entry;
    }
};

class WALWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = std::filesystem::temp_directory_path() / "arrow_wal_writer_test";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    std::filesystem::path testDir;
    std::string GetTestPath(const std::string& name) {
        return (testDir / name).string();
    }

    wal::Entry CreateTestEntry(wal::OperationType type = wal::OperationType::INSERT,
                               const std::string& id = "1",
                               uint32_t dim = 3,
                               uint64_t lsn = 1) {
        std::mt19937 gen(42);
        std::vector<float> vec = RandomVector(dim, gen);
        wal::Entry entry{
            .type = type,
            .version = 1,
            .lsn = lsn,
            .txid = 1,
            .headerCRC = 0,
            .payloadLength = 0,
            .dimension = dim,
            .padding = 0,
            .embedding = vec,
            .payloadCRC = 0
        };
        entry.setVectorID(id);
        entry.payloadLength = entry.computePayloadLength();
        entry.headerCRC = entry.computeHeaderCrc();
        entry.payloadCRC = entry.computePayloadCrc();
        return entry;
    }
};

#endif  // TESTS_CORE_WAL_COMMON_H_
