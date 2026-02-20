#include "common.h"

TEST_F(WALTest, EntryBuilderInsertSuccess) {
    wal::EntryBuilder builder;
    std::vector<float> embedding = {1.0f, 2.0f, 3.0f};
    auto result = builder.buildInsert("v1", 3, embedding);
    ASSERT_TRUE(result.ok());

    auto& entry = result.value();
    EXPECT_EQ(entry.type, wal::OperationType::INSERT);
    EXPECT_EQ(entry.getVectorID(), "v1");
    EXPECT_EQ(entry.dimension, 3u);
    EXPECT_EQ(entry.embedding, embedding);

    // Verify CRCs are valid
    auto valid = wal::IsEntryValid(entry);
    EXPECT_TRUE(valid.ok()) << valid.message();
}

TEST_F(WALTest, EntryBuilderInsertIDTooLong) {
    wal::EntryBuilder builder;
    std::string longID(128, 'x');
    auto result = builder.buildInsert(longID, 3, {1.0f, 2.0f, 3.0f});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), arrow::utils::StatusCode::kInvalidArgument);
}

TEST_F(WALTest, EntryBuilderDeleteSuccess) {
    wal::EntryBuilder builder;
    auto result = builder.buildDelete("v1");
    ASSERT_TRUE(result.ok());

    auto& entry = result.value();
    EXPECT_EQ(entry.type, wal::OperationType::DELETE);
    EXPECT_EQ(entry.getVectorID(), "v1");
    EXPECT_EQ(entry.dimension, 0u);
    EXPECT_TRUE(entry.embedding.empty());
}

TEST_F(WALTest, EntryBuilderDeleteIDTooLong) {
    wal::EntryBuilder builder;
    std::string longID(128, 'x');
    auto result = builder.buildDelete(longID);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), arrow::utils::StatusCode::kInvalidArgument);
}

TEST_F(WALTest, EntryBuilderLSNIncrementing) {
    wal::EntryBuilder builder;
    auto r1 = builder.buildInsert("v1", 2, {1.0f, 2.0f});
    auto r2 = builder.buildInsert("v2", 2, {3.0f, 4.0f});
    ASSERT_TRUE(r1.ok());
    ASSERT_TRUE(r2.ok());
    EXPECT_LT(r1.value().lsn, r2.value().lsn);
}
