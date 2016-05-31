project( Common  )

add_library ( ${PROJECT_NAME} vtkCLUtils.cpp vtkCLUtils.hpp)

target_link_libraries ( 
	${PROJECT_NAME}
	${OpenCL_LIBRARIES}
	${OPENGL_LIBRARIES} 
)

include_directories ( ${COMMON_INCLUDES} )

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )