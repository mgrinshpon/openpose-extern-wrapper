project(openpose_extern_wrapper C CXX)

set(SOURCE_FILES wrapper.cpp)

find_package(OpenCV CONFIG REQUIRED)
find_package(OpenPose CONFIG REQUIRED)
include_directories(${OpenPose_INCLUDE_DIRS})

include_directories("${CMAKE_SOURCE_DIR}/include/openpose_wrapper")

add_library(openpose_extern_wrapper SHARED STATIC ${SOURCE_FILES})

install(TARGETS openpose_extern_wrapper DESTINATION lib)