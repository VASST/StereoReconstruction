# StereoReconstruction

1. Install prost. Get the repository from here (https://github.com/ujayarat/prost.git). 
2. It is required to add some additional proximal and linear operators to prost. To do so, create the file CustomSources.cmake in the directory prost/cmake/ with the following contents:

set(PROST_CUSTOM_SOURCES
  "relative_path_to_sublabel_relax"/cvpr2016/prost/block_dataterm_sublabel.cu
  "relative_path_to_sublabel_relax"/cvpr2016/prost/prox_ind_epi_polyhedral_1d.cu
  "relative_path_to_sublabel_relax"/cvpr2016/prost/prox_ind_epi_conjquad_1d.cu
  "relative_path_to_sublabel_relax"/eccv2016/prost/prox_ind_epi_polyhedral.cu
  
  "relative_path_to_sublabel_relax"/cvpr2016/prost/block_dataterm_sublabel.hpp
  "relative_path_to_sublabel_relax"/cvpr2016/prost/prox_ind_epi_polyhedral_1d.hpp
  "relative_path_to_sublabel_relax"/cvpr2016/prost/prox_ind_epi_conjquad_1d.hpp
  "relative_path_to_sublabel_relax"/eccv2016/prost/prox_ind_epi_polyhedral.hpp
  )
  
set(MATLAB_CUSTOM_SOURCES
  "relative_path_to_sublabel_relax"/cvpr2016/prost/custom.cpp
  "relative_path_to_sublabel_relax"/eccv2016/prost/custom.cpp
  )
  
  Replace "relative_path_to_sublabel_relax" with the "StereoReconstructionRepo/Codes/Sub-Label_Stereo/"
  
  3. Run CMake and build
  
