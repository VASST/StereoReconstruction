add_library ( ThirdParty pgm.h pgm.cpp
				JointWMF.hpp)

target_link_libraries ( 
	ThirdParty 
	)

#install ( TARGETS vtkCLUtils DESTINATION lib )
#install ( FILES ${COMMON_INCLUDES}/vtkCLUtils.hpp DESTINATION include )