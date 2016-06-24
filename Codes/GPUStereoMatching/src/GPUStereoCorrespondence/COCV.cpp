/*==========================================================================

  Copyright (c) 2016 Uditha L. Jayarathne, ujayarat@robarts.ca

  Use, modification and redistribution of the software, in source or
  binary forms, are permitted provided that the following terms and
  conditions are met:

  1) Redistribution of the source code, in verbatim or modified
  form, must retain the above copyright notice, this license,
  the following disclaimer, and any notices that refer to this
  license and/or the following disclaimer.

  2) Redistribution in binary form must include the above copyright
  notice, a copy of this license and the following disclaimer
  in the documentation or with other materials provided with the
  distribution.

  3) Modified copies of the source code must be clearly marked as such,
  and must not be misrepresented as verbatim copies of the source code.

  THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
  WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
  MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
  OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
  THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.
  =========================================================================*/

#include <iostream>

#include "COCV.hpp"

//---------------------------------------------------------------------------------------------------
/*! \param[in] _env opencl environment.
 *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
 *              The class requires **two** `(2)` **command queues** (on the same device).
 */
COCV::COCV (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info):
							env( _env ),  info (_info), 
							context (env.getContext (info.pIdx)), 
							queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
							wta_kernel( env.getProgram(info.pgIdx), "WTA_Kernel" ), 
							dt_kernel(env.getProgram (info.pgIdx), "DiffuseTensor"), 
							precond_kernel(env.getProgram(info.pgIdx), "DiffusionPreconditioning"), 
							dual_update_kernel(env.getProgram(info.pgIdx), "HuberL2DualUpdate"),
							primal_update_kernel(env.getProgram(info.pgIdx), "HuberL2PrimalUpdate"),
							head_primal_update_kernel(env.getProgram(info.pgIdx), "HuberL2HeadPrimalUpdate"),
							pixel_wise_search_kernel(env.getProgram(info.pgIdx), "CostVolumePixelWiseSearch"),
							error_kernel(env.getProgram(info.pgIdx), "HuberL1CVError"),
							waitList (1), 
							radius( 1.f ), eps( 0.01 )
{
}

//-----------------------------------------------------------------------------------------------------
COCV::~COCV()
{
}

//------------------------------------------------------------------------------------------------------------
/*! \details This interface exists to allow CL memory sharing between different kernels.
 *
 *  \param[in] mem enumeration value specifying the requested memory object.
 *  \return A reference to the requested memory object.
 */
cl::Memory& COCV::get (COCV::Memory mem)
{
   switch (mem)
   {
		case COCV::Memory::H_COST_IN:
             return hCostBufferIn;
		case COCV::Memory::H_IMG_IN:
             return hImgBufferIn;
        case COCV::Memory::H_OUT:
             return hBufferOut;
        case COCV::Memory::D_COST_IN:
             return dCostBufferIn;
		case COCV::Memory::D_IMG_IN:
             return dImgBufferIn;
		case COCV::Memory::D_OUT:
			return dBufferOut;
   }
}

//------------------------------------------------------------------------------------------------------------------
/*! \details The transfer happens from a staging buffer on the host to the 
 *           associated (specified) device buffer.
 *  \note The transfer is handled by the first command queue.
 *  
 *  \param[in] mem enumeration value specifying an input device buffer.
 *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
 *                 data from `ptr` will be copied to the associated staging buffer.
 *  \param[in] block a flag to indicate whether to perform a blocking 
 *                   or a non-blocking operation.
 *  \param[in] events a wait-list of events.
 *  \param[out] event event associated with the write operation to the device buffer.
 */
void COCV::write( COCV::Memory mem, 
						 void *ptr, bool block, const std::vector<cl::Event> *events, 
							cl::Event *event)
{
    if (staging == Staging::I || staging == Staging::IO)
    {
         switch (mem)
         {
			case COCV::Memory::D_COST_IN:
                 if (ptr != nullptr)
					 std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height*numLayers, hCostPtrIn);
                 queue.enqueueWriteBuffer (dCostBufferIn, block, 0, costBufferSize, hCostPtrIn, events, event);
                 break;

			case COCV::Memory::D_IMG_IN:
				if (ptr != nullptr)
					 std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hImgPtrIn);
                 queue.enqueueWriteBuffer (dImgBufferIn, block, 0, bufferSize, hImgPtrIn, events, event);
                 break;
			 default:
                break;
		 }
	}
}

//----------------------------------------------------------------------------------------------------------------------
/*! \details The transfer happens from a device buffer to the associated 
 *           (specified) staging buffer on the host.
 *  \note The transfer is handled by the first command queue.
 *  
 *  \param[in] mem enumeration value specifying an output staging buffer.
 *  \param[in] block a flag to indicate whether to perform a blocking 
 *                   or a non-blocking operation.
 *  \param[in] events a wait-list of events.
 *  \param[out] event event associated with the read operation to the staging buffer.
 */
void* COCV::read ( COCV::Memory mem, bool block, 
							const std::vector<cl::Event> *events, cl::Event *event)
{
     if (staging == Staging::O || staging == Staging::IO)
     {
         switch (mem)
         {
		 case COCV::Memory::H_OUT:
				 queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                  return hPtrOut;
             default:
                  return nullptr;
		 }
	 }
        return nullptr;
}

//------------------------------------------------------------------------------------------------------------
/*! \details The initialization happes here
 *  \param[in] _width of the image
 *  \param[in] _height of the image
 *  \param[in] _d_levels disparity levels 
 *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
 */
void COCV::init(int _width, int _height, int _d_min, int _d_max, int _radius, float _alpha, float _beta, float _theta, float _lambda,
								float _eps, Staging _staging)
{
	width = _width; height = _height; d_min = _d_min; d_max = _d_max; 
	numLayers = d_max - d_min +1;
	alpha  = _alpha; beta = _beta; eps = _eps; theta = _theta; lambda = _lambda;
	radius = _radius;
	bufferSize = width * height * sizeof (cl_float);
	costBufferSize = numLayers * width * height * sizeof(cl_float);
	staging = _staging;

    try
    {
       if ((width == 0) || (height == 0))
           throw "The image cannot have zeroed dimensions";
	}
    catch (const char *error)
    {
        std::cerr << "Error[COCV]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}

    // Create staging buffers
    bool io = false;
    switch (staging)
    {
        case Staging::NONE:
             hCostPtrIn = nullptr;
			 hImgPtrIn = nullptr;
			 hGradXPtrIn = nullptr;
			 hGradYPtrIn = nullptr;
             hPtrOut = nullptr;
             break;

        case Staging::IO:
             io = true;

        case Staging::I:
             if (hCostBufferIn () == nullptr)
                 hCostBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, costBufferSize);
			 if (hImgBufferIn () == nullptr)
                 hImgBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

             hCostPtrIn = (cl_float *) queue.enqueueMapBuffer (
				         hCostBufferIn, CL_FALSE, CL_MAP_WRITE, 0, costBufferSize);
			 hImgPtrIn = (cl_float *) queue.enqueueMapBuffer (
				         hImgBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);

             queue.enqueueUnmapMemObject (hCostBufferIn, hCostPtrIn);
			 queue.enqueueUnmapMemObject (hImgBufferIn, hImgPtrIn);

             if (!io)
             {
                 queue.finish ();
                 hPtrOut = nullptr;
                 break;
             }

        case Staging::O:
            if (hBufferOut () == nullptr)
                hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

             hPtrOut = (cl_float *) queue.enqueueMapBuffer (
					         hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
             queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
             queue.finish ();

             if (!io)
			 { 
				 hCostPtrIn = nullptr; hImgPtrIn = nullptr;
				 hGradXPtrIn = nullptr; hGradYPtrIn = nullptr;
			 }
             break;
	}

	// Create device buffers
    if (dCostBufferIn () == nullptr)
        dCostBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, costBufferSize);
	if (dImgBufferIn () == nullptr)
        dImgBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);

	// Out buffer is of size width *height
    if (dBufferOut () == nullptr)
        dBufferOut = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);

	dTensorBuffer  = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize*4 );
	dual_step_sigma0 = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	dual_step_sigma1 = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	primal_step_tau = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	head_primal = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	dual_p0 = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	dual_p1 = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	old_primal = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	new_primal = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	aux  = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	max_disp_cost = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	min_disp_cost = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	min_disp = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize );
	error_img = cl::Buffer( context,  CL_MEM_READ_WRITE, bufferSize );


	// Setup WTA kernel
	wta_kernel.setArg( 0, dCostBufferIn );
	wta_kernel.setArg( 1, min_disp_cost );
	wta_kernel.setArg( 2, max_disp_cost );
	wta_kernel.setArg( 3, min_disp );
	wta_kernel.setArg( 4, d_min );
	wta_kernel.setArg( 5, d_max );
	wta_kernel.setArg( 6, radius );
	wta_kernel.setArg( 7, 10000.f); // MAX_COST

	// Setup DiffusionTensor kernel.
	dt_kernel.setArg( 0, dImgBufferIn);
	dt_kernel.setArg( 1, dTensorBuffer );
	dt_kernel.setArg( 2, alpha );
	dt_kernel.setArg( 3, beta );
	dt_kernel.setArg( 4, d_min );
	dt_kernel.setArg( 5, d_max );
	dt_kernel.setArg( 6, radius );

	// Setup DiffusionPreconditioning kernel.
	precond_kernel.setArg( 0, dTensorBuffer );
	precond_kernel.setArg( 1, dual_step_sigma0 );
	precond_kernel.setArg( 2, dual_step_sigma1 );
	precond_kernel.setArg( 3, primal_step_tau );
	precond_kernel.setArg( 4, d_min );
	precond_kernel.setArg( 5, d_max );
	precond_kernel.setArg( 6, radius);

	// Setup HuberL2DualUpdate
	dual_update_kernel.setArg( 0, dTensorBuffer);
	dual_update_kernel.setArg( 1, head_primal );
	dual_update_kernel.setArg( 2, dual_step_sigma0);
	dual_update_kernel.setArg( 3, dual_step_sigma1);
	dual_update_kernel.setArg( 4, dual_p0);
	dual_update_kernel.setArg( 5, dual_p1);
	dual_update_kernel.setArg( 6, eps );
	dual_update_kernel.setArg( 7, d_max );
	dual_update_kernel.setArg( 8, radius);

	// Setup HuberL2PrimalUpdate
	primal_update_kernel.setArg( 0, dTensorBuffer);
	primal_update_kernel.setArg( 1, old_primal );
	primal_update_kernel.setArg( 2, primal_step_tau );
	primal_update_kernel.setArg( 3, dual_p0 );
	primal_update_kernel.setArg( 4, dual_p1 );
	primal_update_kernel.setArg( 5, aux );
	primal_update_kernel.setArg( 6, new_primal );
	primal_update_kernel.setArg( 7, eps );
	primal_update_kernel.setArg( 8, d_max );
	primal_update_kernel.setArg( 9, radius );
	primal_update_kernel.setArg( 10, theta );

	// Setup HuberL2HeadPrimalUpdate
	head_primal_update_kernel.setArg( 0, head_primal );
	head_primal_update_kernel.setArg( 1, new_primal );
	head_primal_update_kernel.setArg( 2, old_primal );
	head_primal_update_kernel.setArg( 3, d_max );
	head_primal_update_kernel.setArg( 4, radius );
	head_primal_update_kernel.setArg( 5, theta );

	// Setup CostVolumePixelWiseSearch
	pixel_wise_search_kernel.setArg( 0, aux );
	pixel_wise_search_kernel.setArg( 1, max_disp_cost);
	pixel_wise_search_kernel.setArg( 2, min_disp_cost );
	pixel_wise_search_kernel.setArg( 3, new_primal );
	pixel_wise_search_kernel.setArg( 4, dCostBufferIn );
	pixel_wise_search_kernel.setArg( 5, d_min );
	pixel_wise_search_kernel.setArg( 6, d_max );
	pixel_wise_search_kernel.setArg( 7, radius );
	pixel_wise_search_kernel.setArg( 8, theta );
	pixel_wise_search_kernel.setArg( 9, lambda );

	// Setup HuberL1CVError
	error_kernel.setArg( 0, new_primal);
	error_kernel.setArg( 1, dTensorBuffer);
	error_kernel.setArg( 2, dCostBufferIn );
	error_kernel.setArg( 3, error_img );
	error_kernel.setArg( 4, d_min );
	error_kernel.setArg( 5, d_max );
	error_kernel.setArg( 6, radius );
	error_kernel.setArg( 7, lambda );
	error_kernel.setArg( 8, eps );


	// Set workspaces to three dimensions, d being the third dimension
    global = cl::NDRange (width, height);
}

//-------------------------------------------------------------------------------------------------------
/*! \details The function call is non-blocking.
 *
 *  \param[in] events a wait-list of events.
 *  \param[out] event event associated with the kernel execution.
 */
void COCV::run (const std::vector<cl::Event> *events, cl::Event *event)
{
	try
	{

		cl_int err;
		
		// Run WTA kernel
		err = queue.enqueueNDRangeKernel (wta_kernel, cl::NullRange, global, cl::NullRange);

		//--- Primal-Dual + Cost volume optimization ----//
		
		// Calculate Diffusion tensors
		err = queue.enqueueNDRangeKernel (dt_kernel, cl::NullRange, global, cl::NullRange);

		// Do preconditioning on the linear operator (D and nabla )		
		err = queue.enqueueNDRangeKernel ( precond_kernel, cl::NullRange, global, cl::NullRange );

		int num_itr = 150;

		for( int i=0; i<num_itr; i++)
		{
			// Dual update
			err = queue.enqueueNDRangeKernel( dual_update_kernel, cl::NullRange, global, cl::NullRange );

			/// Primal update
			err = queue.enqueueNDRangeKernel( primal_update_kernel, cl::NullRange, global, cl::NullRange );

			// Head primal update
			err = queue.enqueueNDRangeKernel( head_primal_update_kernel, cl::NullRange, global, cl::NullRange );

			// Pixel-wise line search in cost-volume
			err = queue.enqueueNDRangeKernel( pixel_wise_search_kernel, cl::NullRange, global, cl::NullRange );

			// TODO:: Update theta

			err = queue.enqueueNDRangeKernel( error_kernel, cl::NullRange, global, cl::NullRange );

		}		

	}
	catch (const char *error)
    {
        std::cerr << "Error[COCV]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}
}
