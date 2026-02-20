cmake_minimum_required(VERSION 3.16...3.29)
project(arrowdb_node LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ArrowDB project root
set(ARROW_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(ARROW_PROJECT_ROOT "${ARROW_PROJECT_ROOT}" ABSOLUTE)

# Build the ArrowDB C++ library
add_subdirectory(${ARROW_PROJECT_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/arrow)

# cmake-js provides these
include_directories(${CMAKE_JS_INC})

# node-addon-api headers
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
include_directories(${NODE_ADDON_API_DIR})

# Platform-specific link libraries for Rust embed lib
set(ARROW_PLATFORM_LIBS "")
if(APPLE)
    set(ARROW_PLATFORM_LIBS
        "-framework CoreFoundation"
        "-framework Security"
        "-framework SystemConfiguration"
    )
endif()

# Single compilation unit that #includes all binding code
add_library(arrowdb_addon SHARED
    ext/addon.cpp
    ext/bind_types.cpp
    ext/bind_options.cpp
    ext/bind_filter.cpp
    ${CMAKE_JS_SRC}
)

target_include_directories(arrowdb_addon PRIVATE
    ${ARROW_PROJECT_ROOT}/include
    ext
)

target_link_libraries(arrowdb_addon PRIVATE
    arrow
    ${CMAKE_JS_LIB}
    ${ARROW_PLATFORM_LIBS}
)

set_target_properties(arrowdb_addon PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    OUTPUT_NAME "arrowdb_addon"
)

target_compile_definitions(arrowdb_addon PRIVATE
    NAPI_VERSION=9
    NODE_ADDON_API_DISABLE_DEPRECATED
)
