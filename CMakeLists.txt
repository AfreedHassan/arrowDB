cmake_minimum_required(VERSION 3.16)
project(arrowDB LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main executable
add_executable(arrowDB src/main.cpp)
target_sources(arrowDB
	PRIVATE 
		src/main.cpp
)
target_include_directories(arrowDB PRIVATE includes)

# Google Test setup
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Automatically discover all test files
file(GLOB TEST_SOURCES "tests/*.cpp")

# Test executable (includes all test files)
add_executable(tests ${TEST_SOURCES})
target_include_directories(tests PRIVATE includes)
target_link_libraries(tests PRIVATE 
    GTest::gtest 
    GTest::gtest_main
)

# Register tests - Google Test will automatically discover all TEST() macros
add_test(NAME AllTests COMMAND tests)
