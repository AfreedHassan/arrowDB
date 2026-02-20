# ArrowDB

A lightweight vector database in C++23 for similarity search.

## Features

- **HNSW index** - Fast approximate nearest neighbor search with configurable M and ef parameters
- **WAL durability** - Write-ahead logging with CRC32 checksums and crash recovery
- **Metadata filtering** - Filter search results by metadata predicates (eq, gt, lt, in, and/or/not)
- **SQ8 quantization** - Optional scalar quantization for reduced memory and faster search
- **Thread-safe** - Concurrent reads and writes with `std::shared_mutex`
- **Persistence** - Atomic save/load with fsync and file locking
- **Bindings** - Native Node.js and Python bindings

## Quick Start (C++)

```cpp
#include <arrow/arrow.h>

arrow::CollectionConfig config;
config.name = "my_collection";
config.dimensions = 384;
config.space = arrow::Space::Cosine;

// In-memory collection
auto collection = arrow::Collection(config);

// Insert vectors
std::vector<float> embedding(384, 0.1f);
auto id = collection.insert(embedding, {{"category", std::string("science")}});

// Search
auto results = collection.search(embedding, 10);
for (auto& r : results.value()) {
    std::cout << r.id << " score=" << r.score << "\n";
}

// Filtered search
auto filter = arrow::MetadataFilter::eq("category", std::string("science"));
auto filtered = collection.search(embedding, 10, filter);

// Persistent collection
auto persistent = arrow::Collection::create(config, "/tmp/my_db");
```

## Quick Start (Python)

```bash
cd bindings/python && uv pip install -e .
```

```python
import arrowdb

# Using the Client API (manages persistence automatically)
with arrowdb.Client("/tmp/my_db") as client:
    config = arrowdb.CollectionConfig(name="docs", dimensions=384)
    col = client.get_or_create_collection("docs", config)

    col.insert("doc1", [0.1] * 384, {"category": "science"})
    results = col.search([0.1] * 384, k=10)

    # Filtered search
    f = arrowdb.MetadataFilter.eq("category", "science")
    results = col.search([0.1] * 384, k=10, filter=f)
```

## Quick Start (Node.js)

```bash
cd bindings/node && bun install && bun run build
```

```typescript
import { Client, Collection } from "arrowdb";

const client = new Client("/tmp/my_db");
const col = client.createCollection("docs", { dimensions: 384 });

col.insert("doc1", new Float32Array(384).fill(0.1), { category: "science" });
const results = col.search(new Float32Array(384).fill(0.1), 10);

client.close();
```

> **Note:** All Node.js operations are synchronous and block the event loop. For server use, run ArrowDB operations in a [Worker thread](https://nodejs.org/api/worker_threads.html).

## API Overview

| Operation | Description |
|-----------|-------------|
| `insert(id, vec, metadata)` | Insert a vector with optional metadata |
| `insertBatch(docs)` | Batch insert with partial success semantics |
| `search(query, k, ef)` | K-nearest neighbor search |
| `search(query, k, filter, ef)` | Filtered search by metadata |
| `query(vec, k, ef)` | Search with metadata in results |
| `get(id)` | Retrieve a vector by ID |
| `update(id, vec, metadata)` | Update an existing vector |
| `upsert(id, vec, metadata)` | Insert or update |
| `remove(id)` | Delete a vector (lazy, excluded from search) |
| `setMetadata(id, meta)` | Set metadata on a vector |
| `getMetadata(id)` | Get metadata for a vector |
| `optimize()` | Apply SQ8 quantization + BFS graph reorder |
| `save(path)` / `load(path)` | Persist and restore a collection |
| `close()` | Flush WAL and save state |

## Configuration

```cpp
arrow::CollectionConfig config;
config.name = "my_collection";
config.dimensions = 384;               // Vector dimension
config.space = arrow::Space::Cosine;   // Cosine, L2, or InnerProduct

config.index_config.max_elements = 1000000;          // Initial HNSW capacity (auto-grows)
config.index_config.quantization = arrow::Quantization::INT8;  // SQ8 quantization
config.index_config.hnsw_params.M = 16;              // Max connections per node
config.index_config.hnsw_params.ef_construction = 200; // Build beam width
config.index_config.hnsw_params.ef_search = 200;     // Search beam width
```

## Metadata Filtering

```cpp
using arrow::MetadataFilter;

auto f1 = MetadataFilter::eq("status", std::string("active"));
auto f2 = MetadataFilter::gt("score", 0.8);
auto f3 = MetadataFilter::in("tag", {std::string("ml"), std::string("ai")});

// Compose with and/or/not
auto combined = MetadataFilter::and_(f1, MetadataFilter::or_(f2, f3));
auto negated  = MetadataFilter::not_(f1);

auto results = collection.search(query, 10, combined);
```

## Persistence

Collections are saved as directories:

```
collection_dir/
├── meta.json       # Collection config + schema version + recovery metadata
├── index.bin       # HNSW index binary
├── id_space.bin    # VectorID <-> InternalID mapping
├── metadata.json   # Per-vector metadata
└── wal/
    └── db.wal      # Write-ahead log
```

- **Atomic writes**: data files written to `.tmp`, fsynced, then renamed
- **Commit point**: `meta.json` is written last; incomplete saves are detected on load
- **WAL replay**: on dirty shutdown, pending WAL entries are replayed before the collection becomes available
- **File locking**: `flock()` prevents concurrent access from multiple processes

## Performance

Benchmarks on Apple Silicon (M-series), 384-dimensional vectors, Cosine space, M=16, ef=200:

| Operation | 10K vectors | 100K vectors |
|-----------|------------|-------------|
| Insert (in-memory) | 10,150 vec/s | 3,642 vec/s |
| Insert (persistent) | 8,742 vec/s | 3,272 vec/s |
| Insert (batch) | 9,909 vec/s | 3,459 vec/s |
| Save | 45ms | 146ms |
| Search | 110us (9,061 qps) | 356us (2,807 qps) |
| Filtered search (50% selectivity) | 201us (4,979 qps) | 879us (1,138 qps) |

## Building

```bash
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)   # macOS
# make -j$(nproc)              # Linux
```

### Binaries

- `./build/arrowDB` - Usage info
- `./build/tests` - Test suite
- `./build/benchmarks` - Performance benchmarks

### Running Tests

```bash
# All tests (skip embedder tests that require external model)
./build/tests --gtest_filter="-DatasetTest.*:EmbeddingDebug.*:ThreadSafety.*:SIFTTest.*"

# Specific suites
./build/tests --gtest_filter=CollectionTest.*
./build/tests --gtest_filter=HNSWIndexTest.*
./build/tests --gtest_filter=WALTest.*
```

### Dependencies (fetched automatically via CMake)

- hnswlib v0.8.0
- Google Test v1.14.0
- Google Benchmark v1.8.3
- nlohmann/json v3.11.3
- CRoaring (for bitmap-accelerated filtered search)

## Architecture

```
Collection (public API, thread-safe via shared_mutex)
├── HNSWIndex (wraps custom HNSW implementation)
│   └── hnsw::HierarchicalNSW (graph + vector storage)
├── IDSpace (VectorID <-> InternalID mapping with tombstones)
├── WAL (write-ahead log for durability)
└── CollectionPersistence (atomic save/load)
```

- **Collection** uses the Pimpl pattern to hide internals and provide a stable ABI
- **Custom HNSW** implementation at `src/index/hnsw/hnsw.cpp` with built-in thread safety, scalar quantization, and BFS graph reordering
- **WAL entries are logged after successful index insert** to avoid ghost entries on crash
- **Error handling** uses `Status` and `Result<T>` (std::expected wrapper), not exceptions

## License

MIT
