add_library ( vtkCLUtils STATIC vtkCLUtils.cpp vtkCLUtils.hpp)

target_link_libraries ( 
	vtkCLUtils 
	${OPENCL_LIBRARIES} 
	${OPENGL_LIBRARIES} 
)

target_include_directories ( 
	vtkCLUtils PUBLIC 
	${COMMON_INCLUDES} 
)

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )