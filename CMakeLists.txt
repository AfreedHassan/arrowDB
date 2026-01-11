cmake_minimum_required(VERSION 3.16...3.29)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(arrowDB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # compile_commands.json for clangd

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ─────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
  GIT_TAG v0.8.0
)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(hnswlib googletest benchmark json)


# ─────────────────────────────────────────────────────────────
# Rust Embedding Library
# ─────────────────────────────────────────────────────────────
set(EMBED_LIB_DIR ${CMAKE_SOURCE_DIR}/embed/target/release)
set(EMBED_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/embed/include)

find_library(ARROW_EMBED_LIB
    NAMES arrow_embed libarrow_embed
    PATHS ${EMBED_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT ARROW_EMBED_LIB)
  message(WARNING "Embed library not found. Build it with: cd embed && cargo build --release")
    set(ARROW_EMBED_LIB "${EMBED_LIB_DIR}/libarrow_embed.a")
endif()

message(STATUS "Embed library: ${ARROW_EMBED_LIB}")

set(ARROW_EMBEDDING_MODEL_DIR ${CMAKE_SOURCE_DIR}/embed/models)


# ─────────────────────────────────────────────────────────────
# Arrow Library (static)
# ─────────────────────────────────────────────────────────────
add_library(arrow STATIC
    src/core/collection.cpp
    src/core/db.cpp
    src/core/wal.cpp
    src/index/hnsw_index.cpp
)

# PUBLIC: Users of the library get these include paths
# PRIVATE: Only the library itself gets these (for internal headers)
target_include_directories(arrow
    PUBLIC include
    PRIVATE src
)

target_link_libraries(arrow
    PUBLIC nlohmann_json::nlohmann_json
    PRIVATE hnswlib
)

target_compile_options(arrow PRIVATE -O3 -march=native)


# ─────────────────────────────────────────────────────────────
# CLI executable
# ─────────────────────────────────────────────────────────────
add_executable(arrowDB
    src/cli/main.cpp
    src/embedder/embedder.cpp
)

# CLI only sees PUBLIC includes from arrow library
target_include_directories(arrowDB PRIVATE
    src  # For CLI internal headers and embedder
    ${EMBED_INCLUDE_DIR}
)

target_link_libraries(arrowDB PRIVATE
    arrow
    ${ARROW_EMBED_LIB}
)

target_compile_definitions(arrowDB PRIVATE
    ARROW_EMBEDDING_MODEL_DIR="${ARROW_EMBEDDING_MODEL_DIR}"
    ARROW_EMBEDDING_DEFAULT_MODEL="all-MiniLM-L6-v2.onnx"
)

# Platform-specific libraries required by Rust Embed Library
if(APPLE)
    target_link_libraries(arrowDB PRIVATE
        "-framework CoreFoundation"
        "-framework Security"
        "-framework SystemConfiguration"
    )
endif()

target_compile_options(arrowDB PRIVATE -O3 -march=native)


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────
enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")

add_executable(tests ${TEST_SOURCES})

# Tests need PRIVATE access to internal headers for testing internals
target_include_directories(tests PRIVATE
    include
    src
    tests
)

target_link_libraries(tests PRIVATE
    arrow
    hnswlib
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(tests PRIVATE -O3 -march=native)

# Test entries
add_test(NAME HNSWIndexTests COMMAND tests --gtest_filter=HNSWIndexTest.*)
add_test(NAME CollectionTests COMMAND tests --gtest_filter=CollectionTest.*)
add_test(NAME MetadataTests COMMAND tests --gtest_filter=MetadataTest.*)
add_test(NAME MetadataIntegrationTests COMMAND tests --gtest_filter=MetadataIntegrationTest.*)
add_test(NAME BinaryTests COMMAND tests --gtest_filter=BinaryTest.*)
add_test(NAME WALUnitTests COMMAND tests --gtest_filter=WALTest.*)
add_test(NAME CollectionWalIntegrationTests COMMAND tests --gtest_filter=CollectionWalTest.*)
add_test(NAME ArrowDBTests COMMAND tests --gtest_filter=ArrowDBTest.*)

add_test(NAME UnitTests COMMAND tests --gtest_filter="HNSWIndexTest.*:MetadataUnitTest.*")
add_test(NAME IntegrationTests COMMAND tests --gtest_filter="CollectionTest.*:MetadataIntegrationTest.*")


# ─────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────
file(GLOB BENCH_SOURCES "benchmarks/*.cpp")

add_executable(benchmarks ${BENCH_SOURCES})

# Benchmarks need internal access for testing HNSW directly
target_include_directories(benchmarks PRIVATE
    include
    src
)

target_link_libraries(benchmarks PRIVATE
    arrow
    hnswlib
    benchmark::benchmark
    benchmark::benchmark_main
)

target_compile_options(benchmarks PRIVATE -O3 -march=native)
