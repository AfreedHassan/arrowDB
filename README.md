# ArrowDB

A lightweight vector database implementation in C++. 

## Features

- Vector Storage: Configurable dimensions, support for multiple distance metrics (Cosine, L2, Inner Product)
- Similarity Search: Fast approximate nearest neighbor search
- Batch Operations: Efficient bulk insert and search for improved throughput
- Persistence: Save and load collections with write-ahead logging for durability
- Data Ingestion: Tools to load embeddings and metadata from external sources
- Semantic Search: Search with query embeddings to find similar content

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Executables
  - arrowDB - Main database tool
  - tests - Test suite
  - benchmarks - Performance benchmarks

### Ingest Data
```bash
 ./arrowDB 
```

### Search Collection
```bash
./search "your query text here" 
```

###  Testing
Run the full test suite:
```bash
./tests
```

Run specific tests:
```bash   
./tests --gtest_filter=CollectionTest.*
```
## Requirements

- C++23 compatible compiler
- CMake 3.16 or higher
