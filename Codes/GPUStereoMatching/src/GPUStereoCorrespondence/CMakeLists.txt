project( GPUStereoCorrespondence  )

add_library ( ${PROJECT_NAME} GPUStereoDenseCorrespondence.cpp GPUStereoDenseCorrespondence.hpp)

target_link_libraries ( 
	${PROJECT_NAME}
	${OpenCL_LIBRARIES}
)

include_directories ( ${Common_INCLUDES} )
include_directories( ${GPUStereoCorrespondence_INCLUDES} )

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )