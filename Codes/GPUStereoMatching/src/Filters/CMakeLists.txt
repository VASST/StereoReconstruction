add_library ( Filters vtkGuidedFilterAlgo.hpp vtkGuidedFilterAlgo.cpp
			  vtkGuidedFilterMath.hpp vtkGuidedFilterMath.cpp
			  vtkGuidedFilterCommon.hpp )

target_link_libraries ( 
	Filters 
	Common 
	${OpenCL_LIBRARIES}
	)

include_directories ( ${Common_INCLUDES} )
include_directories ( ${Filters_INCLUDES} )

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )