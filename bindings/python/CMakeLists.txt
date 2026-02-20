cmake_minimum_required(VERSION 3.16...3.29)
project(arrowdb_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find nanobind (installed via pip)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find nanobind from pip install, fallback to FetchContent
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_DIR_RESULT)

if(NB_DIR_RESULT EQUAL 0 AND EXISTS "${NB_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.4.0
    )
    FetchContent_MakeAvailable(nanobind)
endif()

# ArrowDB project root (two levels up from bindings/python)
set(ARROW_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(ARROW_PROJECT_ROOT "${ARROW_PROJECT_ROOT}" ABSOLUTE)

# Build the ArrowDB C++ library
add_subdirectory(${ARROW_PROJECT_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/arrow)

# Platform-specific link libraries for Rust embed lib
set(ARROW_PLATFORM_LIBS "")
if(APPLE)
    set(ARROW_PLATFORM_LIBS
        "-framework CoreFoundation"
        "-framework Security"
        "-framework SystemConfiguration"
    )
endif()

# Build the Python extension module
nanobind_add_module(_arrowdb
    ext/module.cpp
    ext/bind_status.cpp
    ext/bind_types.cpp
    ext/bind_options.cpp
    ext/bind_filter.cpp
    ext/bind_collection.cpp
    ext/bind_client.cpp
)

target_include_directories(_arrowdb PRIVATE
    ${ARROW_PROJECT_ROOT}/include
)

target_link_libraries(_arrowdb PRIVATE
    arrow
    ${ARROW_PLATFORM_LIBS}
)

install(TARGETS _arrowdb LIBRARY DESTINATION arrowdb)
