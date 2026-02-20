#include <gtest/gtest.h>
#include "core/id_space.h"
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

TEST_F(IDSpaceTest, EmptyStringIDPersistence) {
	IDSpace original;
	original.assign("");
	original.assign("normal");

	std::string path = GetTestPath("empty_str_id_map.bin");
	ASSERT_TRUE(original.save(path).ok());

	auto loadResult = IDSpace::load(path);
	ASSERT_TRUE(loadResult.ok());

	IDSpace& loaded = loadResult.value();
	EXPECT_EQ(loaded.size(), 2);

	auto lookup = loaded.lookup("");
	ASSERT_TRUE(lookup.ok()) << "Empty string VectorID lost on reload";
	EXPECT_EQ(lookup.value(), 0);

	auto lookup2 = loaded.lookup("normal");
	ASSERT_TRUE(lookup2.ok());
	EXPECT_EQ(lookup2.value(), 1);
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
	// First byte 'c' (99) != kFormatVersion (2), so version mismatch
	EXPECT_EQ(result.status().code(), utils::StatusCode::kVersionMismatch);
}

TEST_F(IDSpaceTest, LoadTruncatedFile) {
	std::string path = GetTestPath("truncated_id_map.bin");
	std::ofstream file(path, std::ios::binary);
	// Write valid version header followed by truncated data
	uint8_t version = 2;
	file.write(reinterpret_cast<const char*>(&version), sizeof(version));
	uint64_t count = 5;
	file.write(reinterpret_cast<const char*>(&count), sizeof(count));
	file.write("partial", 7);
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

TEST_F(IDSpaceTest, TombstoneRoundTrip) {
	IDSpace idSpace;

	// Assign IDs
	for (int i = 0; i < 10; ++i) {
		auto result = idSpace.assign("vec_" + std::to_string(i));
		ASSERT_TRUE(result.ok());
		EXPECT_EQ(result.value(), static_cast<InternalID>(i));
	}
	EXPECT_EQ(idSpace.size(), 10);

	// Remove some IDs (create tombstones)
	ASSERT_TRUE(idSpace.remove("vec_2").ok());
	ASSERT_TRUE(idSpace.remove("vec_5").ok());
	ASSERT_TRUE(idSpace.remove("vec_8").ok());
	EXPECT_EQ(idSpace.size(), 7);

	// Verify removed IDs are not found
	EXPECT_FALSE(idSpace.lookup("vec_2").ok());
	EXPECT_FALSE(idSpace.lookup("vec_5").ok());
	EXPECT_FALSE(idSpace.lookup("vec_8").ok());

	// Save to disk
	std::string path = GetTestPath("tombstone_id_map.bin");
	auto saveResult = idSpace.save(path);
	ASSERT_TRUE(saveResult.ok()) << saveResult.message();

	// Load from disk
	auto loadResult = IDSpace::load(path);
	ASSERT_TRUE(loadResult.ok()) << loadResult.status().message();
	IDSpace& loaded = loadResult.value();

	// Verify size reflects only live entries
	EXPECT_EQ(loaded.size(), 7);

	// Verify removed IDs are still removed after round-trip
	EXPECT_FALSE(loaded.lookup("vec_2").ok());
	EXPECT_EQ(loaded.lookup("vec_2").status().code(), utils::StatusCode::kNotFound);
	EXPECT_FALSE(loaded.lookup("vec_5").ok());
	EXPECT_FALSE(loaded.lookup("vec_8").ok());

	// Verify live IDs still resolve correctly
	for (int i = 0; i < 10; ++i) {
		if (i == 2 || i == 5 || i == 8) continue;
		auto lookup = loaded.lookup("vec_" + std::to_string(i));
		EXPECT_TRUE(lookup.ok()) << "vec_" << i << " should still be live";
		EXPECT_EQ(lookup.value(), static_cast<InternalID>(i));

		auto resolve = loaded.resolve(static_cast<InternalID>(i));
		EXPECT_TRUE(resolve.ok()) << "InternalID " << i << " should still resolve";
		EXPECT_EQ(resolve.value(), "vec_" + std::to_string(i));
	}

	// Verify tombstoned IDs do not resolve
	EXPECT_FALSE(loaded.resolve(2).ok());
	EXPECT_FALSE(loaded.resolve(5).ok());
	EXPECT_FALSE(loaded.resolve(8).ok());
}

TEST_F(IDSpaceTest, LoadEmptyFile) {
	std::string path = GetTestPath("empty_file.bin");
	std::ofstream file(path, std::ios::binary);
	file.close();

	auto result = IDSpace::load(path);
	EXPECT_FALSE(result.ok());
}

TEST_F(IDSpaceTest, LoadTruncatedAfterVersion) {
	std::string path = GetTestPath("version_only.bin");
	std::ofstream file(path, std::ios::binary);
	uint8_t version = 2;
	file.write(reinterpret_cast<const char*>(&version), sizeof(version));
	file.close();

	auto result = IDSpace::load(path);
	EXPECT_FALSE(result.ok());
}

TEST_F(IDSpaceTest, LoadOversizedCount) {
	std::string path = GetTestPath("oversized_count.bin");
	std::ofstream file(path, std::ios::binary);
	uint8_t version = 2;
	file.write(reinterpret_cast<const char*>(&version), sizeof(version));
	uint64_t count = 200000000;
	file.write(reinterpret_cast<const char*>(&count), sizeof(count));
	file.close();

	auto result = IDSpace::load(path);
	EXPECT_FALSE(result.ok());
}

TEST_F(IDSpaceTest, LoadStringLengthOverflow) {
	std::string path = GetTestPath("overflow_string.bin");
	std::ofstream file(path, std::ios::binary);
	uint8_t version = 2;
	file.write(reinterpret_cast<const char*>(&version), sizeof(version));
	uint64_t count = 1;
	file.write(reinterpret_cast<const char*>(&count), sizeof(count));
	uint64_t strLen = 200;
	file.write(reinterpret_cast<const char*>(&strLen), sizeof(strLen));
	std::string garbage(200, 'x');
	file.write(garbage.data(), garbage.size());
	file.close();

	auto result = IDSpace::load(path);
	EXPECT_FALSE(result.ok());
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
