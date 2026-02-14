#include <gtest/gtest.h>
#include "internal/id_space.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace arrow;

class IDSpaceTest : public ::testing::Test {
protected:
	void SetUp() override {
		testDir = std::filesystem::temp_directory_path() / "arrow_id_space_test";
		std::filesystem::create_directories(testDir);
	}

	void TearDown() override {
		if (std::filesystem::exists(testDir)) {
			std::filesystem::remove_all(testDir);
		}
	}

	std::filesystem::path testDir;

	std::string GetTestPath(const std::string& filename) {
		return (testDir / filename).string();
	}
};

TEST_F(IDSpaceTest, BasicAssignLookup) {
	IDSpace idSpace;

	auto result0 = idSpace.assign("a");
	ASSERT_TRUE(result0.ok());
	EXPECT_EQ(result0.value(), 0);

	auto result1 = idSpace.assign("b");
	ASSERT_TRUE(result1.ok());
	EXPECT_EQ(result1.value(), 1);

	EXPECT_EQ(idSpace.size(), 2);

	auto lookupA = idSpace.lookup("a");
	ASSERT_TRUE(lookupA.ok());
	EXPECT_EQ(lookupA.value(), 0);

	auto lookupB = idSpace.lookup("b");
	ASSERT_TRUE(lookupB.ok());
	EXPECT_EQ(lookupB.value(), 1);
}

TEST_F(IDSpaceTest, ResolveInternalID) {
	IDSpace idSpace;

	idSpace.assign("a");
	idSpace.assign("b");
	idSpace.assign("c");

	auto resolve0 = idSpace.resolve(0);
	ASSERT_TRUE(resolve0.ok());
	EXPECT_EQ(resolve0.value(), "a");

	auto resolve1 = idSpace.resolve(1);
	ASSERT_TRUE(resolve1.ok());
	EXPECT_EQ(resolve1.value(), "b");

	auto resolve2 = idSpace.resolve(2);
	ASSERT_TRUE(resolve2.ok());
	EXPECT_EQ(resolve2.value(), "c");
}

TEST_F(IDSpaceTest, AssignExistingID) {
	IDSpace idSpace;

	auto result1 = idSpace.assign("uuid1");
	ASSERT_TRUE(result1.ok());
	EXPECT_EQ(result1.value(), 0);

	auto result2 = idSpace.assign("uuid1");
	ASSERT_TRUE(result2.ok());
	EXPECT_EQ(result2.value(), 0);

	EXPECT_EQ(idSpace.size(), 1);
}

TEST_F(IDSpaceTest, LookupNonexistent) {
	IDSpace idSpace;

	auto result = idSpace.lookup("nonexistent");
	EXPECT_FALSE(result.ok());
	EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(IDSpaceTest, ResolveNonexistent) {
	IDSpace idSpace;

	auto result = idSpace.resolve(42);
	EXPECT_FALSE(result.ok());
	EXPECT_EQ(result.status().code(), utils::StatusCode::kNotFound);
}

TEST_F(IDSpaceTest, PersistenceRoundTrip) {
	IDSpace original;
	original.assign("a");
	original.assign("b");
	original.assign("c");

	std::string path = GetTestPath("id_map.bin");

	auto saveResult = original.save(path);
	ASSERT_TRUE(saveResult.ok());

	auto loadResult = IDSpace::load(path);
	ASSERT_TRUE(loadResult.ok());

	IDSpace& loaded = loadResult.value();
	EXPECT_EQ(loaded.size(), 3);

	EXPECT_EQ(loaded.lookup("a").value(), 0);
	EXPECT_EQ(loaded.lookup("b").value(), 1);
	EXPECT_EQ(loaded.lookup("c").value(), 2);

	EXPECT_EQ(loaded.resolve(0).value(), "a");
	EXPECT_EQ(loaded.resolve(1).value(), "b");
	EXPECT_EQ(loaded.resolve(2).value(), "c");
}

TEST_F(IDSpaceTest, EmptyPersistence) {
	IDSpace original;

	std::string path = GetTestPath("empty_id_map.bin");

	auto saveResult = original.save(path);
	ASSERT_TRUE(saveResult.ok());

	auto loadResult = IDSpace::load(path);
	ASSERT_TRUE(loadResult.ok());

	IDSpace& loaded = loadResult.value();
	EXPECT_EQ(loaded.size(), 0);
}

TEST_F(IDSpaceTest, EmptyStringID) {
	IDSpace idSpace;

	auto result = idSpace.assign("");
	ASSERT_TRUE(result.ok());
	EXPECT_EQ(result.value(), 0);

	auto lookup = idSpace.lookup("");
	ASSERT_TRUE(lookup.ok());
	EXPECT_EQ(lookup.value(), 0);
}

TEST_F(IDSpaceTest, MaxLengthStringID) {
	IDSpace idSpace;
	std::string maxStr(127, 'x');

	auto result = idSpace.assign(maxStr);
	ASSERT_TRUE(result.ok());
	EXPECT_EQ(result.value(), 0);

	auto lookup = idSpace.lookup(maxStr);
	ASSERT_TRUE(lookup.ok());
	EXPECT_EQ(lookup.value(), 0);
}

TEST_F(IDSpaceTest, StringIDTooLong) {
	IDSpace idSpace;
	std::string tooLongStr(128, 'x');

	auto result = idSpace.assign(tooLongStr);
	EXPECT_FALSE(result.ok());
	EXPECT_EQ(result.status().code(), utils::StatusCode::kInvalidArgument);
}

TEST_F(IDSpaceTest, LoadNonexistentFile) {
	std::string path = GetTestPath("nonexistent_id_map.bin");

	auto result = IDSpace::load(path);
	EXPECT_FALSE(result.ok());
	EXPECT_EQ(result.status().code(), utils::StatusCode::kIoError);
}

TEST_F(IDSpaceTest, LoadCorruptedFile) {
	IDSpace idSpace;
	idSpace.assign("a");
	idSpace.assign("b");

	std::string path = GetTestPath("corrupted_id_map.bin");
	idSpace.save(path);

	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	file.write("corrupt", 7);
	file.close();

	auto result = IDSpace::load(path);
	EXPECT_FALSE(result.ok());
	EXPECT_EQ(result.status().code(), utils::StatusCode::kCorruption);
}

TEST_F(IDSpaceTest, LoadTruncatedFile) {
	std::string path = GetTestPath("truncated_id_map.bin");
	std::ofstream file(path, std::ios::binary);
	uint64_t count = 5;
	file.write(reinterpret_cast<const char*>(&count), sizeof(count));
	file.write("partial", 8);
	file.close();

	auto result = IDSpace::load(path);
	EXPECT_FALSE(result.ok());
	EXPECT_EQ(result.status().code(), utils::StatusCode::kCorruption);
}

TEST_F(IDSpaceTest, WalReplaySimulation) {
	IDSpace idSpace;

	std::vector<std::string> vectorIDs = {
		"550e8400-e29b-41d4-a716-446655440000",
		"6ba7b810-9dad-11d1-80b4-00c04fd430c8",
		"6ba7b811-9dad-11d1-80b4-00c04fd430c8"
	};

	for (const auto& vectorID : vectorIDs) {
		auto result = idSpace.assign(vectorID);
		ASSERT_TRUE(result.ok());
	}

	EXPECT_EQ(idSpace.size(), 3);

	for (size_t i = 0; i < vectorIDs.size(); ++i) {
		auto lookup = idSpace.lookup(vectorIDs[i]);
		ASSERT_TRUE(lookup.ok());
		EXPECT_EQ(lookup.value(), i);

		auto resolve = idSpace.resolve(i);
		ASSERT_TRUE(resolve.ok());
		EXPECT_EQ(resolve.value(), vectorIDs[i]);
	}
}

TEST_F(IDSpaceTest, DenseInternalIDs) {
	IDSpace idSpace;

	std::vector<std::string> ids = {"a", "b", "c", "d", "e"};

	for (const auto& id : ids) {
		auto result = idSpace.assign(id);
		ASSERT_TRUE(result.ok());
	}

	for (size_t i = 0; i < ids.size(); ++i) {
		auto result = idSpace.resolve(i);
		ASSERT_TRUE(result.ok());
		EXPECT_EQ(result.value(), ids[i]);
	}
}

TEST_F(IDSpaceTest, UUIDStrings) {
	IDSpace idSpace;

	std::string uuid1 = "550e8400-e29b-41d4-a716-446655440000";
	std::string uuid2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";
	std::string uuid3 = "6ba7b811-9dad-11d1-80b4-00c04fd430c8";

	auto r1 = idSpace.assign(uuid1);
	auto r2 = idSpace.assign(uuid2);
	auto r3 = idSpace.assign(uuid3);

	ASSERT_TRUE(r1.ok());
	ASSERT_TRUE(r2.ok());
	ASSERT_TRUE(r3.ok());

	EXPECT_EQ(r1.value(), 0);
	EXPECT_EQ(r2.value(), 1);
	EXPECT_EQ(r3.value(), 2);

	auto lookup1 = idSpace.lookup(uuid1);
	auto lookup2 = idSpace.lookup(uuid2);
	auto lookup3 = idSpace.lookup(uuid3);

	ASSERT_TRUE(lookup1.ok());
	ASSERT_TRUE(lookup2.ok());
	ASSERT_TRUE(lookup3.ok());

	EXPECT_EQ(lookup1.value(), 0);
	EXPECT_EQ(lookup2.value(), 1);
	EXPECT_EQ(lookup3.value(), 2);
}

TEST_F(IDSpaceTest, PersistenceWithUUIDs) {
	IDSpace original;

	std::vector<std::string> uuids = {
		"550e8400-e29b-41d4-a716-446655440000",
		"6ba7b810-9dad-11d1-80b4-00c04fd430c8",
		"6ba7b811-9dad-11d1-80b4-00c04fd430c8"
	};

	for (const auto& uuid : uuids) {
		original.assign(uuid);
	}

	std::string path = GetTestPath("uuid_id_map.bin");
	ASSERT_TRUE(original.save(path).ok());

	auto loadResult = IDSpace::load(path);
	ASSERT_TRUE(loadResult.ok());

	IDSpace& loaded = loadResult.value();
	EXPECT_EQ(loaded.size(), 3);

	for (size_t i = 0; i < uuids.size(); ++i) {
		auto lookup = loaded.lookup(uuids[i]);
		ASSERT_TRUE(lookup.ok()) << "Failed to lookup UUID: " << uuids[i];
		EXPECT_EQ(lookup.value(), i);

		auto resolve = loaded.resolve(i);
		ASSERT_TRUE(resolve.ok());
		EXPECT_EQ(resolve.value(), uuids[i]);
	}
}
