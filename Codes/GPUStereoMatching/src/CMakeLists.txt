cmake_minimum_required ( VERSION 2.8.12 )
project ( GPUStereoMatching )

option ( BUILD_EXAMPLES "Build examples" OFF)

find_package ( OpenCL REQUIRED )
find_package ( OpenCV REQUIRED )

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )
set ( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )

set ( Common_INCLUDES ${PROJECT_SOURCE_DIR}/Common)
set ( GPUStereoCorrespondence_INCLUDES ${PROJECT_SOURCE_DIR}/GPUStereoCorrespondence)

include_directories ( 
	${OpenCL_INCLUDE_DIR} 
	${OpenCV_INCLUDE_DIRS} 
)

add_subdirectory ( Common )
add_subdirectory( GPUStereoCorrespondence )

if( BUILD_EXAMPLES )
	 
	 add_subdirectory( Examples )
endif( BUILD_EXAMPLES )
