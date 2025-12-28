cmake_minimum_required(VERSION 3.16)
project(arrowDB LANGUAGES CXX)

add_executable(arrowDB src/main.cpp)
target_sources(arrowDB
	PRIVATE 
		src/main.cpp
)
set_target_properties(arrowDB PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
