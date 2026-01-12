# ArrowDB

A lightweight vector database implementation in C++. 

## Features

- **Vector Storage**: Configurable dimensions, support for multiple distance metrics (Cosine, L2, Inner Product)
- **Similarity Search**: Fast approximate nearest neighbor search using HNSW index
- **Batch Operations**: Efficient bulk insert and search for improved throughput
- **Persistence**: Save and load collections with write-ahead logging for durability
- **Data Ingestion**: Load 200K+ pre-computed embeddings from OpenWebText via Rust FFI
- **Semantic Search**: Search with query embeddings to find similar content
- **Text Embedding**: Generate embeddings using all-MiniLM-L6-v2 ONNX model
- **Multi-Language**: C++ core with Rust integration for text processing and file I/O

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Executables
  - **arrowDB** - Interactive REPL for vector search (run without args) or CLI mode (with args)
  - **tests** - Test suite
  - **benchmarks** - Performance benchmarks

### Interactive REPL Mode

Run ArrowDB without arguments to enter interactive mode:

```bash
./arrowDB
```

This launches an interactive REPL with the following commands:

**Commands:**
- `.search <query>` - Search for similar vectors using your query text
- `.help` - Display help message
- `.exit` - Exit the REPL

**Example session:**
```
arrowdb> .search machine learning algorithms
# ... search results displayed ...

arrowdb> .search deep learning transformers
# ... more results ...

arrowdb> .exit
Goodbye!
```

### Command-Line Mode

Run ArrowDB with arguments for single command execution:

```bash
./arrowDB search "your query text" [-c <collection>] [-t <text_file>]
./arrowDB ingest -e <embeddings_file> -i <ids_file> -t <text_file>
```

### Loading OpenWebText Dataset (200K Vectors)

The Embedder class now includes dataset loading capabilities via Rust FFI:

```cpp
#include "embedder/embedder.h"

Embedder embedder("models/all-MiniLM-L6-v2.onnx");

// Load OpenWebText dataset with embeddings
auto result = embedder.loadOpenWebText(
    "openwebtext.txt",              // Text chunks file (one per line)
    "openwebtext-embeddings.bin",   // Binary embeddings file (float32)
    200000,                          // Number of chunks to load
    40,                              // Minimum text length (characters)
    200                              // Maximum text length (characters)
);

if (result) {
    // Access loaded data
    const auto& chunks = result->chunks;       // std::vector<std::string>
    const auto& embeddings = result->embeddings; // std::vector<std::vector<float>>

    // Insert into collection
    for (size_t i = 0; i < chunks.size(); ++i) {
        collection.insert(i, embeddings[i]);
    }
} else {
    std::cerr << "Failed to load dataset\n";
}
```

### Creating Dataset Files

**Text file format** (openwebtext.txt):
```
One chunk of text with minimum 40 characters length
Another text chunk meeting the length requirements
And so on, one per line
```

**Embeddings file format** (openwebtext-embeddings.bin):
- Binary file with float32 values
- Format: flat array of all embeddings concatenated
- Total size: num_chunks × 384 × 4 bytes
- Little-endian byte order

### Ingest Data
```bash
 ./arrowDB
```

### Search Collection
```bash
./search "your query text here"
```

### Testing
Run the full test suite:
```bash
./tests
```

Run specific test suites:
```bash
./tests --gtest_filter=CollectionTest.*
./tests --gtest_filter=DatasetTest.*
./tests --gtest_filter=HNSWIndexTest.*
```
## Architecture

### Core Components

**Collection** - Main interface for vector database operations
- Insert/search operations
- Metadata management
- Persistence to disk

**HNSWIndex** - Hierarchical Navigable Small World graph
- Approximate nearest neighbor search
- Configurable M and ef parameters
- Multiple distance metrics

**Embedder** - Text embedding via ONNX Runtime
- all-MiniLM-L6-v2 model integration
- Dataset loading from files
- L2 normalization

**WAL (Write-Ahead Log)** - Durability and recovery
- Binary format with CRC32 checksums
- Transaction support
- Atomic operations

### Data Formats

**Collections** are persisted as directories:
```
collection_name/
├── meta.json       # Configuration metadata
├── index.bin       # HNSW index binary
└── metadata.json   # Vector metadata
```

**WAL** uses binary format with:
- Magic number and version
- CRC32 checksums (header + payload)
- Transaction entries (INSERT, UPDATE, DELETE)

## Requirements

- C++23 compatible compiler
- CMake 3.16 or higher
- ONNX Runtime (for embeddings)
- Rust toolchain (for dataset loading)

## Performance

Based on benchmarks with default config (M=64, EF=200):
- **Insert throughput**: ~3-14k vectors/sec
- **Search latency**: ~8-9ms for 100K vectors
- **Recall**: ~91-92% recall@10 for approximate search
- **Memory**: ~1KB per vector in HNSW graph

## Testing

Comprehensive test suite included:
```bash
make test           # Run all tests
./tests --gtest_filter=DatasetTest.*  # Dataset loading tests
./tests --gtest_filter=CollectionTest.*  # Collection tests
./tests --gtest_filter=HNSWIndexTest.*  # Index tests
```

All 165 tests pass with no memory leaks.

## License

This project is provided as-is and is built for fun (and maybe prod).
