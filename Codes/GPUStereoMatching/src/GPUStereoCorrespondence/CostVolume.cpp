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
#include "CostVolume.hpp"

//---------------------------------------------------------------------------------------------------
/*! \param[in] _env opencl environment.
 *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
 *              The class requires **two** `(2)` **command queues** (on the same device).
 */
CostVolume::CostVolume (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info):
							env( _env ),  info (_info), 
							context (env.getContext (info.pIdx)), 
							queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
							cost_calc(env.getProgram (info.pgIdx), "compute_cost"), 
							waitList_cost_calc (1), 
							color_th( 0.f ), grad_th( 0.f), alpha( .5f ), d_min( 5 ), d_max( 30 )
{
}

//-----------------------------------------------------------------------------------------------------
CostVolume::~CostVolume()
{
}

//------------------------------------------------------------------------------------------------------------
/*! \details This interface exists to allow CL memory sharing between different kernels.
 *
 *  \param[in] mem enumeration value specifying the requested memory object.
 *  \return A reference to the requested memory object.
 */
cl::Memory& CostVolume::get (CostVolume::Memory mem)
{
   switch (mem)
   {
		case CostVolume::Memory::H_IN_L:
             return hLBufferIn;
		case CostVolume::Memory::H_IN_R:
             return hRBufferIn;
		case CostVolume::Memory::H_IN_LGRAD:
			return hLGRADBufferIn;
		case CostVolume::Memory::H_IN_RGRAD:
			return hRGRADBufferIn;
        case CostVolume::Memory::H_OUT:
             return hBufferOut;
        case CostVolume::Memory::D_IN_L:
             return dLBufferIn;
		case CostVolume::Memory::D_IN_R:
             return dRBufferIn;
		case CostVolume::Memory::D_IN_LGRAD:
			return dLGRADBufferIn;
		case CostVolume::Memory::D_IN_RGRAD:
			return dRGRADBufferIn;
        case CostVolume::Memory::D_OUT:
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
void CostVolume::write( CostVolume::Memory mem, 
						 void *ptr, bool block, const std::vector<cl::Event> *events, 
							cl::Event *event)
{
    if (staging == Staging::I || staging == Staging::IO)
    {
         switch (mem)
         {
			case CostVolume::Memory::D_IN_L:
                 if (ptr != nullptr)
                    std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hLPtrIn);
                 queue.enqueueWriteBuffer (dLBufferIn, block, 0, bufferInSize, hLPtrIn, events, event);
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
void* CostVolume::read ( CostVolume::Memory mem, bool block, 
							const std::vector<cl::Event> *events, cl::Event *event)
{
     if (staging == Staging::O || staging == Staging::IO)
     {
         switch (mem)
         {
		 case CostVolume::Memory::H_OUT:
				 queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
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
void CostVolume::init(int _width, int _height, int _d_min, int _d_max, int _cth, int _gth, double _alpha, Staging _staging)
{
	width = _width; height = _height; d_min = _d_min; d_max = _d_max; color_th = _cth, grad_th = _gth;
	alpha = _alpha;
	bufferInSize = width * height * sizeof (cl_float);
	bufferOutSize = (d_max-d_min+1) * width * height * sizeof(cl_float);
	staging = _staging;

    try
    {
       if ((width == 0) || (height == 0))
           throw "The image cannot have zeroed dimensions";

	   if ((width * height) % 4 != 0)
           throw "The number of elements in the array has to be a multiple of 4";
	}
    catch (const char *error)
    {
        std::cerr << "Error[CostVolume]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}

    // Create staging buffers
    bool io = false;
    switch (staging)
    {
        case Staging::NONE:
             hLPtrIn = nullptr;
			 hRPtrIn = nullptr;
			 hLGRADPtrIn = nullptr;
			 hRGRADPtrIn = nullptr;
             hPtrOut = nullptr;
             break;

        case Staging::IO:
             io = true;

        case Staging::I:
             if (hLBufferIn () == nullptr)
                 hLBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
			 if (hRBufferIn () == nullptr)
                 hRBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
			 if (hLGRADBufferIn () == nullptr )
                 hLGRADBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
			 if (hRGRADBufferIn () == nullptr )
                 hRGRADBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

             hLPtrIn = (cl_float *) queue.enqueueMapBuffer (
				         hLBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
			 hRPtrIn = (cl_float *) queue.enqueueMapBuffer (
				         hRBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
			 hLGRADPtrIn = (cl_float *) queue.enqueueMapBuffer (
						 hLGRADBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
			 hRGRADPtrIn = (cl_float *) queue.enqueueMapBuffer (
						 hRGRADBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);

             queue.enqueueUnmapMemObject (hLBufferIn, hLPtrIn);
			 queue.enqueueUnmapMemObject (hRBufferIn, hRPtrIn);
			 queue.enqueueUnmapMemObject( hLGRADBufferIn, hLGRADPtrIn );
			 queue.enqueueUnmapMemObject( hRGRADBufferIn, hRGRADPtrIn );

             if (!io)
             {
                 queue.finish ();
                 hPtrOut = nullptr;
                 break;
             }

        case Staging::O:
            if (hBufferOut () == nullptr)
                hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

             hPtrOut = (cl_float *) queue.enqueueMapBuffer (
					         hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
             queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
             queue.finish ();

             if (!io)
			 { 
				 hLPtrIn = nullptr; hRPtrIn = nullptr;
				 hLGRADPtrIn = nullptr; hRGRADPtrIn = nullptr;
			 }
             break;
	}

	// Create device buffers
    if (dLBufferIn () == nullptr)
        dLBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
	if (dRBufferIn () == nullptr)
        dRBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
	if( dLGRADBufferIn() == nullptr)
		dLGRADBufferIn = cl::Buffer( context, CL_MEM_READ_ONLY, bufferInSize);
	if( dRGRADBufferIn() == nullptr)
		dRGRADBufferIn = cl::Buffer( context, CL_MEM_READ_ONLY, bufferInSize);
	// Out buffer is of size d_levels *width *height
    if (dBufferOut () == nullptr)
        dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

	cost_calc.setArg( 0, dLBufferIn );
	cost_calc.setArg( 1, dRBufferIn );
	cost_calc.setArg( 2, dLGRADBufferIn);
	cost_calc.setArg( 3, dRGRADBufferIn);
	cost_calc.setArg( 4, dBufferOut );
	int patch_r = 3;
	int vx = 2*(patch_r-1)+1;
	cost_calc.setArg( 5, static_cast<int>(patch_r));
	cost_calc.setArg( 6, (int)d_min);
	cost_calc.setArg( 7, (int)d_max);
	cost_calc.setArg( 8, static_cast<float>(color_th));
	cost_calc.setArg( 9, static_cast<float>(grad_th));
	cost_calc.setArg( 10, static_cast<float>(alpha));
	cost_calc.setArg( 11, 2);
	cost_calc.setArg( 12, static_cast<float>(1000));

	// Set workspaces to three dimensions, d being the third dimension
    global = cl::NDRange (width, height, (d_max-d_min+1));
}

//-------------------------------------------------------------------------------------------------------
/*! \details The function call is non-blocking.
 *
 *  \param[in] events a wait-list of events.
 *  \param[out] event event associated with the kernel execution.
 */
void CostVolume::run (const std::vector<cl::Event> *events, cl::Event *event)
{
	try
	{
		cl_int err = queue.enqueueNDRangeKernel (cost_calc, cl::NullRange, global, cl::NullRange);
		waitList_cost_calc[0] = cost_calc_event;
	}
	catch (const char *error)
    {
        std::cerr << "Error[CostVolume]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}
}