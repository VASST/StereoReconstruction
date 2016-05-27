cmake_minimum_required ( VERSION 2.8.12 )
project ( vtkGuidedFilter )

find_package ( OpenCL REQUIRED )
find_package ( OpenGL REQUIRED )

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )
set ( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )

set ( COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/Common)

include_directories ( 
	${OpenCL_INCLUDE_DIR} 
	${OPENGL_INCLUDE_DIRS} 
)

add_subdirectory ( Common )

