cmake_minimum_required ( VERSION 2.8.12 )
project ( GPUStereoMatching )

option ( BUILD_EXAMPLES "Build examples" OFF)

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )
set ( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )
set( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

find_package ( OpenCL REQUIRED )
find_package ( OpenGL REQUIRED )
find_package ( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

set ( Common_INCLUDES ${PROJECT_SOURCE_DIR}/Common)
set ( GPUStereoCorrespondence_INCLUDES ${PROJECT_SOURCE_DIR}/GPUStereoCorrespondence)
set ( Filters_INCLUDES ${PROJECT_SOURCE_DIR}/Filters)
set ( ThirdParty_INCLUDES $PROJECT_SOURCE_DIR}/ThirdParty )

include_directories ( 
	${OpenCL_INCLUDE_DIR} 
	${OPENGL_INCLUDE_DIRS} 
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}	
)

add_subdirectory ( Common )
add_subdirectory( GPUStereoCorrespondence )
add_subdirectory( Filters )
add_subdirectory( ThirdParty )

if( BUILD_EXAMPLES )
	 
	 add_subdirectory( Examples )
endif( BUILD_EXAMPLES )
