#include "internal/id_space.h"
#include "internal/binary.h"
#include "internal/filesync.h"
#include <fstream>

namespace arrow {

utils::Result<InternalID> IDSpace::assign(const std::string& vectorID) {
	if (vectorID.size() > kMaxVectorIDSize) {
		return utils::Status(utils::StatusCode::kInvalidArgument, "Vector ID exceeds maximum size");
	}

	auto it = lookupMap.find(vectorID);
	if (it != lookupMap.end()) {
		return it->second;
	}

	InternalID id = nextId_++;
	lookupMap[vectorID] = id;
	resolveList.push_back(vectorID);

	return id;
}


utils::Result<InternalID> IDSpace::lookup(const std::string& vectorID) const {
	auto it = lookupMap.find(vectorID);
	if (it == lookupMap.end()) {
		return utils::Status(utils::StatusCode::kNotFound, "Vector ID not found");
	}
	return it->second;
}

utils::Result<std::string_view> IDSpace::resolve(InternalID id) const {
	if (id >= resolveList.size()) {
		return utils::Status(utils::StatusCode::kNotFound, "Internal ID out of bounds");
	}
	if (removedIds_.contains(id)) {
		return utils::Status(utils::StatusCode::kNotFound, "Internal ID has been removed");
	}
	return std::string_view(resolveList[id]);
}

utils::Status IDSpace::remove(const VectorID& vectorID) {
  auto it = lookupMap.find(vectorID);
  if (it == lookupMap.end()) {
    return utils::Status(utils::StatusCode::kNotFound, "Vector ID not found");
  }
  InternalID id = it->second;
  removedIds_.insert(id);
  // Clear the resolveList entry (tombstone — empty string)
  if (id < resolveList.size()) {
    resolveList[id].clear();
  }
  lookupMap.erase(it);
  return utils::OkStatus();
}

utils::Status IDSpace::save(const std::filesystem::path& path) const {
	std::filesystem::path tempPath = path;
	tempPath += ".tmp";

	std::ofstream file(tempPath, std::ios::binary);
	if (!file) {
		return utils::Status(utils::StatusCode::kIoError, "Failed to open file for writing");
	}

	BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

	uint64_t count = resolveList.size();
	writer.write(count);

	for (const std::string& str : resolveList) {
		writer.writeString(str);
	}

	writer.flush();

	if (writer.fail()) {
		return utils::Status(utils::StatusCode::kIoError, "Write error while saving id_space");
	}

	if (!utils::syncFile(tempPath.string())) {
		return utils::Status(utils::StatusCode::kIoError, "Failed to fsync id_space temp file");
	}

	std::error_code ec;
	std::filesystem::rename(tempPath, path, ec);
	if (ec) {
		return utils::Status(utils::StatusCode::kIoError, "Failed to rename temp file");
	}

	// Fsync parent directory to ensure rename is durable
	utils::syncDir(path.parent_path().string());

	return utils::Status();
}

utils::Result<IDSpace> IDSpace::load(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return utils::Status(utils::StatusCode::kIoError, "Failed to open file for reading");
	}

	BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

	IDSpace idSpace;

	uint64_t count;
	if (!reader.read(count)) {
		return utils::Status(utils::StatusCode::kCorruption, "Failed to read count");
	}

	// Sanity bound: prevent OOM from corrupt count values.
	// 100M vectors is a generous upper bound for a single collection.
	static constexpr uint64_t kMaxReasonableCount = 100'000'000;
	if (count > kMaxReasonableCount) {
		return utils::Status(utils::StatusCode::kCorruption,
			"ID space count (" + std::to_string(count) + ") exceeds maximum (" +
			std::to_string(kMaxReasonableCount) + ")");
	}

	for (uint64_t i = 0; i < count; ++i) {
		try {
			std::string str;
			reader.read(str);

			if (str.size() > kMaxVectorIDSize) {
				return utils::Status(utils::StatusCode::kCorruption, "String length exceeds maximum");
			}

			if (!reader.good()) {
				return utils::Status(utils::StatusCode::kCorruption, "Failed to read string data");
			}

			InternalID id = idSpace.nextId_++;
			if (str.empty()) {
				// Tombstoned entry — preserve slot but mark as removed
				idSpace.resolveList.push_back(std::string{});
				idSpace.removedIds_.insert(id);
			} else {
				idSpace.lookupMap[str] = id;
				idSpace.resolveList.push_back(std::move(str));
			}
		} catch (...) {
			return utils::Status(utils::StatusCode::kCorruption, "Failed to read string data");
		}
	}

	return idSpace;
}

}
