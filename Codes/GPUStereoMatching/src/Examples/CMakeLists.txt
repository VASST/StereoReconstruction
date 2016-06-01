CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

project( GPU_StereoMatcher_Example )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories ( ${Common_INCLUDES} )
include_directories ( ${GPUStereoCorrespondence_INCLUDES} )

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries( ${PROJECT_NAME} Common GPUStereoCorrespondence ${OpenCV_LIBRARIES} ${OpenCL_LIBRARIES} )