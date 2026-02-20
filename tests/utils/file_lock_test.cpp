#include <gtest/gtest.h>
#include "utils/file_lock.h"
#include <filesystem>

using namespace arrow;

class FileLockTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir = std::filesystem::temp_directory_path() / "arrow_file_lock_test";
    std::filesystem::create_directories(testDir);
  }

  void TearDown() override {
    if (std::filesystem::exists(testDir)) {
      std::filesystem::remove_all(testDir);
    }
  }

  std::filesystem::path testDir;
};

TEST_F(FileLockTest, AcquireLockOnNewDirectory) {
  auto lockDir = testDir / "lock_dir";
  auto result = FileLock::acquire(lockDir);
  EXPECT_TRUE(result.ok()) << result.status().message();
}

TEST_F(FileLockTest, AcquireLockTwiceFails) {
  auto lockDir = testDir / "double_lock";
  auto lock1 = FileLock::acquire(lockDir);
  ASSERT_TRUE(lock1.ok()) << lock1.status().message();

  auto lock2 = FileLock::acquire(lockDir);
  EXPECT_FALSE(lock2.ok());
  EXPECT_EQ(lock2.status().code(), utils::StatusCode::kIoError);
}

TEST_F(FileLockTest, LockReleasedOnDestruction) {
  auto lockDir = testDir / "raii_lock";

  // Acquire and release via RAII
  {
    auto lock1 = FileLock::acquire(lockDir);
    ASSERT_TRUE(lock1.ok()) << lock1.status().message();
    // lock1 goes out of scope here, releasing the lock
  }

  // Should be able to acquire again
  auto lock2 = FileLock::acquire(lockDir);
  EXPECT_TRUE(lock2.ok()) << lock2.status().message();
}

TEST_F(FileLockTest, MoveSemantics) {
  auto lockDir = testDir / "move_lock";

  auto result = FileLock::acquire(lockDir);
  ASSERT_TRUE(result.ok()) << result.status().message();

  // Move-construct into a new FileLock
  FileLock movedLock = std::move(result.value());

  // The lock should still be held by movedLock, so acquiring again should fail
  auto result2 = FileLock::acquire(lockDir);
  EXPECT_FALSE(result2.ok());
  EXPECT_EQ(result2.status().code(), utils::StatusCode::kIoError);
}

TEST_F(FileLockTest, MoveAssignment) {
  auto lockDir1 = testDir / "move_assign_src";
  auto lockDir2 = testDir / "move_assign_dst";

  auto result1 = FileLock::acquire(lockDir1);
  ASSERT_TRUE(result1.ok()) << result1.status().message();

  auto result2 = FileLock::acquire(lockDir2);
  ASSERT_TRUE(result2.ok()) << result2.status().message();

  // Move-assign lock1 into lock2 (should release lock2, transfer lock1)
  result2.value() = std::move(result1.value());

  // lockDir1 should still be locked (held by result2 now)
  auto result3 = FileLock::acquire(lockDir1);
  EXPECT_FALSE(result3.ok());

  // lockDir2 should be unlocked (released by move-assign)
  auto result4 = FileLock::acquire(lockDir2);
  EXPECT_TRUE(result4.ok()) << result4.status().message();
}
