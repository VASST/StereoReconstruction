% compile HuberL1CVPrecond_mex.cu

% Set NVCC 
 setenv('MW_NVCC_PATH','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin');
 
 % invoke mexcuda
 mexcuda -v -O -largeArrayDims -outdir ./miccai2013/ ./miccai2013/HuberL1CVPrecond_mex.cu
