cmake_minimum_required ( VERSION 2.8.12 )
project ( vtkGuidedFilter )

option ( BUILD_TESTS "Build tests" OFF )
option ( BUILD_EXAMPLES "Build examples" OFF)

find_package ( OpenCL REQUIRED )
find_package ( OpenGL REQUIRED )

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )
set ( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )

set ( COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/Common)
set ( GUIDEDFILTER_INCLUDES ${PROJECT_SOURCE_DIR}/GuidedFilter)

include_directories ( 
	${OpenCL_INCLUDE_DIR} 
	${OPENGL_INCLUDE_DIRS} 
)

add_subdirectory ( Common )
add_subdirectory( GuidedFilter )

if ( BUILD_TESTS )
    enable_testing (  )
    
    find_package ( GTest QUIET )
    if ( NOT GTEST_FOUND )
        message ( STATUS "gtest not found:" )
        message ( STATUS " - Library will be downloaded from source" )
        add_subdirectory ( external/gtest )
    endif ( NOT GTEST_FOUND )
    
    add_subdirectory ( Tests )
endif ( BUILD_TESTS )

if( BUILD_EXAMPLES )
	 
	 add_subdirectory( Examples )
endif( BUILD_EXAMPLES )
