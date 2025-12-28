# ArrowDB

A lightweight vector database implementation in C++. 

## Features

- Vector storage with configurable dimensions
- Flat search with cosine similarity (dot product)
- REPL interface for interactive use
- C++23 standard

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Run the executable:
```bash
.build/arrowDB
```

The program currently creates a collection and displays its configuration. A REPL interface is available for interactive commands (type `.exit` to quit).

## Project Structure

- `src/main.cpp` - Main entry point and REPL
- `includes/vector_store.h` - Vector storage implementation
- `includes/flat_search.h` - Flat search algorithms
- `includes/arrow_utils.h` - Utility functions for vector operations
- `includes/vdb.h` - Core database structures

## Requirements

- C++23 compatible compiler
- CMake 3.16 or higher
