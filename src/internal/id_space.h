#ifndef ARROW_ID_SPACE_H
#define ARROW_ID_SPACE_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <filesystem>

#include "arrow/utils/result.h"
#include "arrow/types.h"

namespace arrow {

class IDSpace {
public:
	IDSpace() = default;

	IDSpace(const IDSpace&) = delete;
	IDSpace& operator=(const IDSpace&) = delete;
	IDSpace(IDSpace&&) noexcept = default;
	IDSpace& operator=(IDSpace&&) noexcept = default;

	utils::Result<InternalID> assign(const VectorID& vectorID);
	utils::Result<InternalID> lookup(const VectorID& vectorID) const;
	utils::Result<std::string_view> resolve(InternalID id) const;
	utils::Status remove(const VectorID& vectorID);

	size_t size() const noexcept { return lookupMap.size(); }

	utils::Status save(const std::filesystem::path& path) const;
	static utils::Result<IDSpace> load(const std::filesystem::path& path);

private:
	static constexpr size_t kMaxVectorIDSize = 127;

	std::unordered_map<VectorID, InternalID> lookupMap;
	std::vector<VectorID> resolveList;
	std::unordered_set<InternalID> removedIds_;
	InternalID nextId_ = 0;
};

}

#endif // ARROW_ID_SPACE_H
