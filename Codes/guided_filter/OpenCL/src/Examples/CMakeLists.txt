CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

project( Guided_Filter_Image )

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories ( ${COMMON_INCLUDES} )
include_directories ( ${GUIDEDFILTER_INCLUDES} )

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries( ${PROJECT_NAME} Common GuidedFilter ${OpenCV_LIBRARIES} ${OpenCL_LIBRARIES} )