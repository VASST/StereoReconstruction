project( GPUStereoCorrespondence  )

add_library ( ${PROJECT_NAME} 
			   GradientFilter.cpp GradientFilter.hpp
			   GrayscaleFilter.hpp GrayscaleFilter.cpp
			   BFCostAggregator.hpp BFCostAggregator.cpp
			   CostVolume.hpp CostVolume.cpp)

target_link_libraries ( 
	${PROJECT_NAME}
	${OpenCL_LIBRARIES}
	Common
	Filters
)

include_directories ( ${Common_INCLUDES} )
include_directories( ${GPUStereoCorrespondence_INCLUDES} )
include_directories( ${Filters_INCLUDES} )

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )