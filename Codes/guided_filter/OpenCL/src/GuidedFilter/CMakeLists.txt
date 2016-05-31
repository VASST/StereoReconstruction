add_library ( GuidedFilter vtkGuidedFilterAlgo.hpp vtkGuidedFilterAlgo.cpp
			  vtkGuidedFilterMath.hpp vtkGuidedFilterMath.cpp
			  vtkGuidedFilterCommon.hpp )

target_link_libraries ( 
	GuidedFilter 
	Common 
	${OpenCL_LIBRARIES}
	)

include_directories ( ${COMMON_INCLUDES} )
include_directories ( ${GUIDEDFILTER_INCLUDES} )

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )