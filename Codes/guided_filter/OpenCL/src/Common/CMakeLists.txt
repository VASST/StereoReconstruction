add_library ( Common STATIC vtkCLUtils.cpp vtkCLUtils.hpp)

target_link_libraries ( 
	Common 
	${OPENCL_LIBRARIES} 
	${OPENGL_LIBRARIES} 
)

target_include_directories ( 
	Common PUBLIC 
	${COMMON_INCLUDES} 
)

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )