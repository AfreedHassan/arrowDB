cmake_minimum_required(VERSION 3.16)
project(vdb LANGUAGES CXX)

add_executable(vdb src/main.cpp)
set_target_properties(vdb PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
