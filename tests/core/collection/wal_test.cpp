#include "common.h"
#include "wal/wal.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
using nlohmann::json;

using namespace wal;

TEST_F(CollectionWalTest, WalLoggingEnabledWithPersistencePath) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("wal_enabled");
  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(uuidv4(1), vec);

    EXPECT_TRUE(status.ok()) << status.message();
  }

  {
    auto walResult = wal::WAL::open(persistencePath + "/wal");
    ASSERT_TRUE(walResult.ok()) << walResult.status().message();
    auto& wal = walResult.value();

    auto headerResult = wal::LoadHeader(persistencePath + "/wal");
    ASSERT_TRUE(headerResult.ok()) << headerResult.status().message();
    EXPECT_EQ(headerResult.value().magic, 0x41574C01);

    auto entriesResult = wal.readAll();
    ASSERT_TRUE(entriesResult.ok()) << entriesResult.status().message();
    const auto& walContents = entriesResult.value();
    EXPECT_EQ(walContents.entries.size(), 1);
    EXPECT_EQ(walContents.entries[0].type, wal::OperationType::INSERT);
    EXPECT_EQ(walContents.entries[0].getVectorID(), uuidv4(1));
  }
}

TEST_F(CollectionWalTest, WalNotCreatedWithoutPersistencePath) {
  auto config = GetTestConfig();
  Collection collection(config);

  std::vector<float> vec = RandomVector(128, gen);
  auto status = collection.insert(uuidv4(1), vec);

  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(collection.size(), 1);
}

TEST_F(CollectionWalTest, WalLogOnInsert) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("insert_wal");
  auto createResult = Collection::create(config, persistencePath);
  ASSERT_TRUE(createResult.ok()) << createResult.status().message();
  Collection collection = std::move(createResult.value());

  const size_t numInserts = 10;
  for (size_t i = 0; i < numInserts; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(uuidv4(i), vec);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  EXPECT_EQ(collection.size(), numInserts);

  auto walResult = wal::WAL::open(persistencePath + "/wal");
  ASSERT_TRUE(walResult.ok()) << walResult.status().message();
  auto& wal = walResult.value();
  auto entriesResult = wal.readAll();
  ASSERT_TRUE(entriesResult.ok()) << entriesResult.status().message();
  const auto &entries = entriesResult.value().entries;
  EXPECT_EQ(entries.size(), numInserts);

  for (size_t i = 0; i < numInserts; ++i) {
    EXPECT_EQ(entries[i].type, wal::OperationType::INSERT);
    EXPECT_EQ(entries[i].getVectorID(), uuidv4(i));
    EXPECT_EQ(entries[i].dimension, 128);
    EXPECT_FALSE(entries[i].embedding.empty());
  }
}

TEST_F(CollectionWalTest, WalLogOnDelete) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("delete_wal");
  auto createResult = Collection::create(config, persistencePath);
  ASSERT_TRUE(createResult.ok()) << createResult.status().message();
  Collection collection = std::move(createResult.value());

  const size_t numInserts = 5;
  for (size_t i = 0; i < numInserts; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(uuidv4(i), vec);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  auto deleteStatus = collection.remove(uuidv4(2));
  ASSERT_TRUE(deleteStatus.ok()) << deleteStatus.message();

  auto walResult = WAL::open(persistencePath + "/wal");
  ASSERT_TRUE(walResult.ok()) << walResult.status().message();
  auto& wal = walResult.value();
  auto entriesResult = wal.readAll();
  ASSERT_TRUE(entriesResult.ok()) << entriesResult.status().message();
  const auto &entries = entriesResult.value().entries;

  EXPECT_EQ(entries.size(), numInserts + 1);

  int insertCount = 0;
  int deleteCount = 0;
  for (const auto &entry : entries) {
    if (entry.type == wal::OperationType::INSERT) {
      insertCount++;
    } else if (entry.type == wal::OperationType::DELETE) {
      deleteCount++;
      EXPECT_EQ(entry.getVectorID(), uuidv4(2));
    }
  }
  EXPECT_EQ(insertCount, numInserts);
  EXPECT_EQ(deleteCount, 1);
}

TEST_F(CollectionWalTest, CheckpointTruncatesWalAfterSave) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("checkpoint_wal");
  auto createResult = Collection::create(config, persistencePath);
  ASSERT_TRUE(createResult.ok()) << createResult.status().message();
  Collection collection = std::move(createResult.value());

  const size_t numInserts = 10;
  for (size_t i = 0; i < numInserts; ++i) {
    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection.insert(uuidv4(i), vec);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  auto walBeforeResult = WAL::open(persistencePath + "/wal");
  ASSERT_TRUE(walBeforeResult.ok());
  auto& walBefore = walBeforeResult.value();
  auto entriesBefore = walBefore.readAll();
  ASSERT_TRUE(entriesBefore.ok());
  EXPECT_EQ(entriesBefore.value().entries.size(), numInserts);

  auto saveStatus = collection.save(persistencePath);
  ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();

  auto walAfterResult = WAL::open(persistencePath + "/wal");
  ASSERT_TRUE(walAfterResult.ok());
  auto& walAfter = walAfterResult.value();
  auto entriesAfter = walAfter.readAll();
  ASSERT_TRUE(entriesAfter.ok());
  EXPECT_EQ(entriesAfter.value().entries.size(), 0);

  auto headerResult = wal::LoadHeader(persistencePath + "/wal");
  ASSERT_TRUE(headerResult.ok());
  EXPECT_EQ(headerResult.value().magic, 0x41574C01);
}

TEST_F(CollectionWalTest, CrashRecoveryReplaysWal) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("crash_recovery");

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
    EXPECT_EQ(collection.size(), 10);
  }

  {
    auto loadResult2 = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult2.ok()) << loadResult2.status().message();
    Collection collection2 = std::move(loadResult2.value());

    for (size_t i = 10; i < 20; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection2.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    EXPECT_EQ(collection2.size(), 20);
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 20);
  EXPECT_TRUE(recovered.recoveredFromWal());
}

TEST_F(CollectionWalTest, LoadWithoutCrashDoesNotReplayWal) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("no_crash");

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    for (size_t i = 0; i < 5; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 5);
  EXPECT_FALSE(recovered.recoveredFromWal());
}

TEST_F(CollectionWalTest, WalReplayPreservesMetadata) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("metadata_wal");

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();

      Metadata meta;
      meta["idx"] = static_cast<int64_t>(i);
      collection.setMetadata(uuidv4(i), meta);
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  {
    auto loadResult2 = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult2.ok()) << loadResult2.status().message();
    Collection collection2 = std::move(loadResult2.value());

    std::vector<float> vec = RandomVector(128, gen);
    auto status = collection2.insert(uuidv4(10), vec);
    ASSERT_TRUE(status.ok()) << status.message();

    Metadata meta;
    meta["idx"] = static_cast<int64_t>(10);
    collection2.setMetadata(uuidv4(10), meta);
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 11);
  EXPECT_TRUE(recovered.recoveredFromWal());
}

TEST_F(CollectionWalTest, DeleteReplayMarksVectorAsDeleted) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("delete_replay");

  std::vector<float> vector5;

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      if (i == 5) {
        vector5 = vec;
      }
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  {
    auto loadResult2 = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult2.ok()) << loadResult2.status().message();
    Collection collection2 = std::move(loadResult2.value());

    auto deleteStatus = collection2.remove(uuidv4(5));
    ASSERT_TRUE(deleteStatus.ok()) << deleteStatus.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_TRUE(recovered.recoveredFromWal());
  EXPECT_EQ(recovered.size(), 10);

  auto results = recovered.search(vector5, 10);
  for (const auto& result : results) {
    EXPECT_NE(result.id, "5") << "Deleted vector 5 should not appear in search results";
  }
}

TEST_F(CollectionWalTest, ContinuityAcrossRestarts) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("continuity");

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    for (size_t i = 0; i < 5; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    EXPECT_EQ(collection.size(), 5);

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  {
    auto loadResult = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
    Collection collection = std::move(loadResult.value());

    EXPECT_EQ(collection.size(), 5);

    for (size_t i = 5; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    EXPECT_EQ(collection.size(), 10);

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 10);
}

TEST_F(CollectionWalTest, EmptyWalDoesNotCauseRecovery) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("empty_wal");

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    for (size_t i = 0; i < 5; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_EQ(recovered.size(), 5);
  EXPECT_FALSE(recovered.recoveredFromWal());
}

TEST_F(CollectionWalTest, RecoveryMetadataIsPersisted) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("recovery_meta");

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    for (size_t i = 0; i < 10; ++i) {
      std::vector<float> vec = RandomVector(128, gen);
      auto status = collection.insert(uuidv4(i), vec);
      ASSERT_TRUE(status.ok()) << status.message();
    }

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  auto metaPath = std::filesystem::path(persistencePath) / "meta.json";
  std::ifstream file(metaPath);
  ASSERT_TRUE(file.is_open());

  json j;
  file >> j;
  file.close();

  EXPECT_TRUE(j.contains("recovery"));
  const auto &recovery = j["recovery"];
  EXPECT_TRUE(recovery.contains("lastPersistedLsn"));
  EXPECT_TRUE(recovery.contains("lastPersistedTxid"));
  EXPECT_TRUE(recovery.contains("cleanShutdown"));

  EXPECT_EQ(recovery["lastPersistedLsn"].get<uint64_t>(), 10);
  EXPECT_EQ(recovery["lastPersistedTxid"].get<uint64_t>(), 10);
  EXPECT_TRUE(recovery["cleanShutdown"].get<bool>());
}

TEST_F(CollectionWalTest, UpdateReplayAfterCrash) {
  auto config = GetTestConfig();
  std::string persistencePath = GetTestPath("update_replay");

  std::vector<float> originalVec;
  std::vector<float> updatedVec;

  {
    auto createResult = Collection::create(config, persistencePath);
    ASSERT_TRUE(createResult.ok()) << createResult.status().message();
    Collection collection = std::move(createResult.value());

    originalVec = RandomVector(128, gen);
    auto s = collection.insert("target", originalVec, {{"version", int64_t(1)}});
    ASSERT_TRUE(s.ok()) << s.message();

    auto saveStatus = collection.save(persistencePath);
    ASSERT_TRUE(saveStatus.ok()) << saveStatus.message();
  }

  {
    auto loadResult = Collection::load(persistencePath);
    ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
    Collection collection = std::move(loadResult.value());

    updatedVec = RandomVector(128, gen);
    auto s = collection.update("target", updatedVec, {{"version", int64_t(2)}});
    ASSERT_TRUE(s.ok()) << s.message();
  }

  auto loadResult = Collection::load(persistencePath);
  ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
  Collection recovered = std::move(loadResult.value());

  EXPECT_TRUE(recovered.recoveredFromWal());

  auto getResult = recovered.get("target");
  ASSERT_TRUE(getResult.ok()) << getResult.status().message();

  const auto& recoveredVec = getResult.value();
  ASSERT_EQ(recoveredVec.size(), updatedVec.size());
  for (size_t i = 0; i < updatedVec.size(); ++i) {
    EXPECT_NEAR(recoveredVec[i], updatedVec[i], 1e-5f)
        << "Mismatch at index " << i;
  }
}
