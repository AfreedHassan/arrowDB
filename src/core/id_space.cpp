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

utils::Result<InternalID> IDSpace::reserve(const std::string& vectorID) const {
	if (vectorID.size() > kMaxVectorIDSize) {
		return utils::Status(utils::StatusCode::kInvalidArgument, "Vector ID exceeds maximum size");
	}

	auto it = lookupMap.find(vectorID);
	if (it != lookupMap.end()) {
		return it->second;
	}

	return nextId_;
}

utils::Status IDSpace::commit(const std::string& vectorID, InternalID id) {
	if (lookupMap.contains(vectorID)) {
		return utils::Status(utils::StatusCode::kAlreadyExists, "Vector ID already committed");
	}

	lookupMap[vectorID] = id;
	resolveList.push_back(vectorID);
	nextId_ = id + 1;
	return utils::OkStatus();
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
	return std::string_view(resolveList[id]);
}

utils::Status IDSpace::remove(const VectorID& vectorID) {
  lookupMap.erase(vectorID);
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

	if (!utils::syncFile(tempPath.string())) {
		return utils::Status(utils::StatusCode::kIoError, "Failed to fsync id_space temp file");
	}

	std::error_code ec;
	std::filesystem::rename(tempPath, path, ec);
	if (ec) {
		return utils::Status(utils::StatusCode::kIoError, "Failed to rename temp file");
	}

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
			idSpace.lookupMap[str] = id;
			idSpace.resolveList.push_back(std::move(str));
		} catch (...) {
			return utils::Status(utils::StatusCode::kCorruption, "Failed to read string data");
		}
	}

	return idSpace;
}

}
