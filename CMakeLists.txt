cmake_minimum_required(VERSION 3.16...3.29)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(arrowDB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # compile_commands.json for clangd
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

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
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)  # Don't build benchmark tests
FetchContent_MakeAvailable(hnswlib googletest benchmark json)



# ─────────────────────────────────────────────────────────────
# Main executable
# ─────────────────────────────────────────────────────────────
add_executable(arrowDB src/main.cpp)
target_sources(arrowDB PRIVATE
    src/main.cpp
		src/core/wal.cpp
		src/index/hnsw_index.cpp
)
target_include_directories(arrowDB PRIVATE include)
target_link_libraries(arrowDB 
	PRIVATE 
		hnswlib 
		nlohmann_json::nlohmann_json 
)
target_compile_options(arrowDB PRIVATE -O3 -march=native)  # Enable CPU-specific SIMD


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────
enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")

add_executable(tests ${TEST_SOURCES})
target_sources(tests PRIVATE
		src/index/hnsw_index.cpp
		src/core/wal.cpp
)
target_include_directories(tests PRIVATE include tests)
target_link_libraries(tests PRIVATE 
    hnswlib
    GTest::gtest 
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
target_compile_options(tests PRIVATE -O3 -march=native)  # Enable CPU-specific SIMD

# Add separate test entries for unit and integration tests using GoogleTest filters
add_test(NAME HNSWIndexTests COMMAND tests --gtest_filter=HNSWIndexTest.*)
add_test(NAME CollectionTests COMMAND tests --gtest_filter=CollectionTest.*)
add_test(NAME MetadataTests COMMAND tests --gtest_filter=MetadataTest.*)
add_test(NAME MetadataIntegrationTests COMMAND tests --gtest_filter=MetadataIntegrationTest.*)
add_test(NAME WALUnitTests COMMAND tests --gtest_filter=WALTest.*)
#add_test(NAME SIFTTests COMMAND tests --gtest_filter=SIFTTest.*)

# Grouped test entries for convenience
add_test(NAME UnitTests COMMAND tests --gtest_filter="HNSWIndexTest.*:MetadataUnitTest.*")
add_test(NAME IntegrationTests COMMAND tests --gtest_filter="CollectionTest.*:MetadataIntegrationTest.*")

# Also add a combined test that runs everything
#add_test(NAME AllTests COMMAND tests)


# ─────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────
file(GLOB BENCH_SOURCES "benchmarks/*.cpp")

add_executable(benchmarks ${BENCH_SOURCES})
target_sources(benchmarks PRIVATE
    src/index/hnsw_index.cpp
)
target_include_directories(benchmarks PRIVATE include)
target_link_libraries(benchmarks PRIVATE 
    hnswlib
    benchmark::benchmark
    benchmark::benchmark_main
    nlohmann_json::nlohmann_json
)
target_compile_options(benchmarks PRIVATE -O3 -march=native)
