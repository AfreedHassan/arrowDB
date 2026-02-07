# AGENTS.md

If any part of the instructions seem unclear or you are not sure how to do something, stop immediately and ask instead of guessing. Follow the principles of the codebase and always choose the simplest method of accomplishing something instead of something complex. 

## Project Overview 

This is a lightweight vector database implementation in C++ with an Embedding Pipeline making use of Rust.

## Build Commands

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Run Tests

**Note:** Tests must be run directly via the test binary, not through ctest:

```bash
cd build
make tests                           # Build tests only
./tests                              # Run all tests
./tests --gtest_filter="*WAL*"       # Run WAL-related tests
./tests --gtest_filter="*Collection*" # Run Collection tests
./tests --gtest_filter=CollectionTest.InsertVectors  # Single test case
```

**Common test filters:**
- `"*WAL*"` - All WAL tests (36 tests)
- `"*Collection*"` - All Collection tests (40 tests)
- `"*HNSW*"` - HNSW index tests
- `"*Binary*"` - Binary serialization tests
- `"*Metadata*"` - Metadata tests
- `"*IDSpace*"` - ID space management tests

### Test Organization
Tests are organized by category in CMakeLists.txt:
- **HNSW Tests**: HNSWIndexTest, Distance Kernels (IP/L2), VectorStorage, SIFT
- **Embedder Tests**: EmbeddingDebug, Dataset, ThreadSafety
- **Core Tests**: Collection, WAL, Metadata, Binary, ArrowDB

Test categories can be run together:
```bash
ctest -R AllHNSWTests          # All HNSW-related tests
ctest -R AllCoreTests          # All core functionality tests
ctest -R UnitTests             # HNSW, Metadata, WAL unit tests
ctest -R IntegrationTests      # Collection, Metadata integration tests
```

### Test Utilities
Use `test_util.h` helper functions for generating test data:
```cpp
#include "test_util.h"

using arrow::testing::RandomVector;

std::mt19937 gen(42);
std::vector<float> vec = RandomVector(128, gen);  // Normalized vector
```

## WAL Entry Structure

The WAL Entry struct uses a fixed 128-byte array for vectorID storage:

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

    std::string getVectorID() const;           // Extract string from buffer
    utils::Status setVectorID(const std::string& id);  // Set with bounds check
};
```

**Important constants:**
- `kVectorIDSize = 128` - Wire/struct size of vectorID field
- `kMaxVectorIDSize = 127` - Max usable string length (1 byte reserved for null terminator)

**Creating WAL entries:**
```cpp
wal::Entry entry{
    .type = wal::OperationType::INSERT,
    .version = 1,
    .lsn = 1,
    .txid = 1,
    // ... other fields (NOT vectorID here)
};
entry.setVectorID("my-vector-id");  // Use helper method after initialization
```


## Code Style Guidelines

### Namespace
All code in `arrow` namespace. Use `namespace arrow {` at file level.

### Naming Conventions
- Classes/Structs: `PascalCase` (e.g., `Collection`, `HNSWIndex`)
- Methods/Functions: `camelCase` (e.g., `insert`, `search`, `saveIndex`)
- Free Functions: `CamelCase` (e.g., `DoThis`, `DoThat`, `ComputeLikeThis`)
- Variables: `camelCase` (e.g., `maxElements`, `efConstruction`)
- Private members: `snake_case_` (e.g., `dim_`, `metric_`, `index_`)
- Constants: `kPascalCase` (e.g., `kDefaultDim`)
- Enums/Type aliases: `PascalCase` (e.g., `DistanceMetric::Cosine`, `VectorID`)

### File Organization
- Headers: `include/arrow/` - Public interfaces
- Source: `src/core/`, `src/index/` - Implementation
- Tests: `tests/` - Google Test files
- Utils: `include/arrow/utils/` - Helper functions

### Import Order
1. Local headers (quotes, alphabetical)
2. Third-party libraries (angle brackets)
3. Standard library (angle brackets, alphabetical)

```cpp
#include "arrow/types.h"
#include "arrow/hnsw_index.h"
#include <hnswlib/hnswlib.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <memory>
```

### Formatting
- Use spaces for indentation (1 TUS = 4 spaces)
- NO comments unless explicitly requested
- Opening braces on same line, closing on new line

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
- Use `size_t` for sizes/counts
- Use `uint32_t`, `uint64_t`, `int64_t` for bit-width requirements
- Use `float` for vector embeddings, `std::vector<float>` for vectors
- Use `std::unique_ptr` for owned resources
- Use `utils::Result<T>` for fallible operations (preferred)
- Use `std::optional<T>` for operations that may return no value

### Error Handling
- **NO EXCEPTIONS** - Project does not use exceptions at all
- Handle errors explicitly within functions instead of throwing
- Use `utils::Status` for status-only returns (success/failure with error info)
- Use `utils::Result<T>` for fallible operations (wraps `std::expected<T, Status>`)
- Use `std::optional<T>` for operations that may return no value
- Return error codes or bool for simple success/failure cases

**StatusCode enum values:**
- `kOk` - Success
- `kInvalidArgument`, `kNotFound`, `kAlreadyExists`, `kUnimplemented` - Generic errors
- `kDimensionMismatch` - Vector dimension errors
- `kIoError`, `kEof`, `kCorruption`, `kChecksumMismatch` - I/O & persistence errors
- `kBadRecord`, `kBadHeader`, `kVersionMismatch` - WAL / recovery errors
- `kInternal` - Internal invariants

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
		return utils::Status(utils::StatusCode::kInvalidArgument,
			"Batch cannot be empty");
	}
	BatchInsertResult result;
	return result;
}
```

### Const Correctness
- Mark member functions `const` if they don't modify state
- Pass by `const&` for read-only parameters

```cpp
const std::string& name() const { return config_.name; }
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
- Use `EXPECT_*` for non-critical, `ASSERT_*` for test-terminating assertions

```cpp
TEST_F(HNSWIndexTest, InsertAndSearch) {
	HNSWIndex index(3, DistanceMetric::Cosine);
	index.insert(1, {1.0f, 0.0f, 0.0f});
	EXPECT_EQ(index.size(), 1);
}
```

### Doxygen Comments
Use Doxygen-style comments only for public APIs:
```cpp
/// Insert a vector into the collection.
void insert(VectorID id, const std::vector<float>& vec);
```

### CMake Integration
- Source files added to `add_library()` or `add_executable()` in CMakeLists.txt
- New tests: add to TEST_SOURCES glob pattern or individual test entries
- Link test binaries with `GTest::gtest` and `GTest::gtest_main`

### After Changes
Always run tests: `cd build && make && ctest --output-on-failure`
No lint commands configured. Focus on following existing patterns.

### Commit Message Format

Use conventional commit messages for all commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring (no functional change)
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(hnsw): add HNSW index implementation

Implements HNSW (Hierarchical Navigable Small World) index for efficient
approximate nearest neighbor search. Supports Euclidean and Cosine distance metrics.

Closes #123

---

refactor(wal): use std::unique_ptr for stream ownership

Makes BinaryReader and BinaryWriter take ownership of filestreams
using std::unique_ptr. Provides automatic RAII resource management
and eliminates manual file lifecycle concerns.

Affects WAL tests - ensures proper file handle cleanup.
```

## BOUNDARIES

Never ever delete files with rm or similar commands
