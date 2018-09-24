cmake_minimum_required(VERSION 3.2)
#project(openpose_extern_wrapper)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src)
add_subdirectory(test)
