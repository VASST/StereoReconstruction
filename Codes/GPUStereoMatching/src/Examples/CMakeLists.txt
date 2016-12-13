CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

project( Examples )

#find_package( PCL REQUIRED)

include_directories( ${PCL_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories ( ${Common_INCLUDES} )
include_directories ( ${GPUStereoCorrespondence_INCLUDES} )
include_directories( ${Filters_INCLUDES} )
include_directories( ${ThirdParty_INCLUDES})

add_executable(GPU_StereoMatcher_Example stereo_main.cpp)

target_link_libraries( GPU_StereoMatcher_Example Common Filters ThirdParty GPUStereoCorrespondence ${OpenCV_LIBRARIES} ${OpenCL_LIBRARIES} )

add_executable( GPU_COCV_Example COCV_test_main.cpp)
target_link_libraries(GPU_COCV_Example Common Filters ThirdParty GPUStereoCorrespondence ${OpenCV_LIBRARIES} ${OpenCL_LIBRARIES} ${PCL_LIBRARIES})


add_executable( GPU_GuidedFilter_Example guided_filter_main.cpp)
target_link_libraries( GPU_GuidedFilter_Example Common Filters ${OpenCV_LIBRARIES} ${OpenCL_LIBRARIES} )