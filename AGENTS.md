# AGENTS.md

If any part of the instructions seem unclear or you are not sure how to do something, stop immediately and ask instead of guessing. Follow the principles of the codebase and always choose the simplest method of accomplishing something instead of something complex.

## Project Overview

ArrowDB is a lightweight vector database in C++23 with HNSW indexing, WAL-based durability, and optional scalar quantization (SQ8).

## Build Commands

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)  # macOS
```

**Optional build flags:**
- `cmake .. -DARROW_SANITIZE=address` - Enable address sanitizer
- `cmake .. -DARROW_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug` - Enable code coverage

**Rust embedder (optional):** `cd embed && cargo build --release`

## Run Tests

**All tests:** `./build/tests`

**Skip external data tests (SIFT/embedder):**
```bash
./build/tests --gtest_filter="-DatasetTest.*:EmbeddingDebug.*:ThreadSafety.*:SIFTTest.*"
```

**Single test suite:**
```bash
./build/tests --gtest_filter=CollectionTest.*
./build/tests --gtest_filter=HNSWIndexTest.*
./build/tests --gtest_filter=WALTest.*
./build/tests --gtest_filter=CollectionTest.InsertVectors
```

**Test categories (via ctest):**
```bash
ctest -R AllHNSWTests       # HNSW-related tests
ctest -R AllCoreTests       # Core functionality (Collection, WAL, Metadata)
ctest -R UnitTests          # HNSW, Metadata, WAL unit tests
ctest -R IntegrationTests   # Collection, Metadata integration
```

**Test utilities:**
```cpp
#include "test_util.h"
using arrow::testing::RandomVector;
std::mt19937 gen(42);
std::vector<float> vec = RandomVector(128, gen);  // Normalized vector
```

## Code Style Guidelines

### Naming Conventions
- Classes/Structs: `PascalCase` (e.g., `Collection`, `HNSWIndex`)
- Methods/Functions: `camelCase` (e.g., `insert`, `search`)
- Free Functions: `CamelCase` (e.g., `DoThis`, `ComputeLikeThis`)
- Variables: `camelCase` (e.g., `maxElements`, `efConstruction`)
- Private members: `snake_case_` (e.g., `dim_`, `metric_`, `index_`)
- Constants: `kPascalCase` (e.g., `kDefaultDim`)

### Namespace
All code in `arrow` namespace. Utilities in `arrow::utils`.

### Import Order
1. Local headers (quotes, alphabetical)
2. Third-party libraries (angle brackets)
3. Standard library (angle brackets, alphabetical)

```cpp
#include "arrow/types.h"
#include "arrow/collection.h"
#include "index/hnsw_index.h"
#include <hnswlib/hnswlib.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <memory>
```

### Formatting
- Spaces for indentation (1 TUS = 4 spaces)
- Opening braces on same line, closing on new line
- NO comments unless explicitly requested

```cpp
class Collection {
public:
	void insert(VectorID id, const std::vector<float>& vec) {
		index_->insert(id, vec);
	}
private:
	std::unique_ptr<HNSWIndex> index_;
};
```

### Types
- `size_t` for sizes/counts
- `uint32_t`, `uint64_t`, `int64_t` for bit-width requirements
- `float` for vector embeddings, `std::vector<float>` for vectors
- `std::unique_ptr` for owned resources
- `utils::Result<T>` for fallible operations (preferred)
- `std::optional<T>` for operations that may return no value

### Error Handling
**NO EXCEPTIONS** - Handle errors explicitly. Use:
- `utils::Status` for status-only returns
- `utils::Result<T>` for fallible operations (wraps `std::expected<T, Status>`)
- `std::optional<T>` for operations that may return no value
- Error codes or bool for simple success/failure

```cpp
utils::Status insert(VectorID id, const std::vector<float>& vec) {
	if (vec.size() != dim_) {
		return utils::Status(utils::StatusCode::kDimensionMismatch,
			"Vector dimension mismatch: expected " + std::to_string(dim_) +
			", got " + std::to_string(vec.size()));
	}
	index_->insert(id, vec);
	return utils::OkStatus();
}

utils::Result<BatchInsertResult> insertBatch(
	const std::vector<std::pair<VectorID, std::vector<float>>>& batch) {
	if (batch.empty()) {
		return utils::Status(utils::StatusCode::kInvalidArgument, "Batch cannot be empty");
	}
	BatchInsertResult result;
	return result;
}
```

### RAII and Resource Management
- Use RAII patterns for all resources (files, memory, locks)
- Prefer `std::unique_ptr` over raw pointers
- Classes managing resources should be non-copyable, movable

```cpp
class HNSWIndex {
public:
	HNSWIndex(const HNSWIndex&) = delete;
	HNSWIndex& operator=(const HNSWIndex&) = delete;
	HNSWIndex(HNSWIndex&&) noexcept;
	HNSWIndex& operator=(HNSWIndex&&) noexcept;
};
```

### Google Test Conventions
- Test fixtures: `ClassNameTest : public ::testing::Test`
- Test names: `PascalCase` descriptive names (e.g., `InsertAndSearch`)
- Use `SetUp()` and `TearDown()` for fixture setup/cleanup
- `EXPECT_*` for non-critical, `ASSERT_*` for test-terminating assertions

```cpp
TEST_F(HNSWIndexTest, InsertAndSearch) {
	HNSWIndex index(3, Space::Cosine);
	index.insert(1, {1.0f, 0.0f, 0.0f});
	EXPECT_EQ(index.size(), 1);
}
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `style`, `test`, `chore`

**Example:**
```
feat(hnsw): add HNSW index implementation

Implements HNSW for efficient approximate nearest neighbor search.
Supports Euclidean and Cosine distance metrics.
```

## WAL Entry Structure

```cpp
struct Entry {
    OperationType type;
    uint16_t version;
    uint64_t lsn;
    uint64_t txid;
    uint32_t headerCRC;
    uint32_t payloadLength = 0;
    char vectorID[kVectorIDSize] = {};  // Fixed 128-byte array
    uint32_t dimension = 0;
    uint8_t padding;
    std::vector<float> embedding;
    uint32_t payloadCRC = 0;

    std::string getVectorID() const;
    utils::Status setVectorID(const std::string& id);
};
```

**Important constants:**
- `kVectorIDSize = 128` - Wire/struct size
- `kMaxVectorIDSize = 127` - Max usable string length

**Creating WAL entries:**
```cpp
wal::Entry entry{
    .type = wal::OperationType::INSERT,
    .version = 1,
    .lsn = 1,
    .txid = 1,
};
entry.setVectorID("my-vector-id");  // Use helper after initialization
```

## BOUNDARIES

Never ever delete files with rm or similar commands
