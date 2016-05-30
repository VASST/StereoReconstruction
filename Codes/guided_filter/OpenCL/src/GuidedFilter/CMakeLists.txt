add_library ( GuidedFilter STATIC vtkGuidedFilterAlgo.hpp vtkGuidedFilterAlgo.cpp
			  vtkGuidedFilterMath.hpp vtkGuidedFilterMath.cpp
			  vtkGuidedFilterCommon.hpp )

target_link_libraries ( 
	GuidedFilter 
	Common 
)

target_include_directories ( 
	GuidedFilter PUBLIC 
	${COMMON_INCLUDES}
	${GUIDEDFILTER_INCLUDES}
)

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )