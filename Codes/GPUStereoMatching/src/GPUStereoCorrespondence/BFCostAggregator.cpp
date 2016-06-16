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
#include <sstream>
#include "BFCostAggregator.hpp"

//---------------------------------------------------------------------------------------------------
/*! \param[in] _env opencl environment.
 *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
 *              The class requires **two** `(2)` **command queues** (on the same device).
 */
CostAggregator::CostAggregator (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info):
							env( _env ),  info (_info), 
							context (env.getContext (info.pIdx)), 
							queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
							bf_kernel(env.getProgram (info.pgIdx), "boxFilterSAT_Tr"),
							box_filter (_env, _info), 
							waitList_bf (1), 
							numLayers(1), radius(3), scaling(1.f)
{   
}

//-----------------------------------------------------------------------------------------------------
CostAggregator::~CostAggregator()
{
}

//------------------------------------------------------------------------------------------------------------
/*! \details This interface exists to allow CL memory sharing between different kernels.
 *
 *  \param[in] mem enumeration value specifying the requested memory object.
 *  \return A reference to the requested memory object.
 */
cl::Memory& CostAggregator::get (CostAggregator::Memory mem)
{
   switch (mem)
   {
		case CostAggregator::Memory::H_IN:
             return hBufferIn;		
        case CostAggregator::Memory::H_OUT:
             return hBufferOut;
        case CostAggregator::Memory::D_IN:
             return dBufferIn;		
        case CostAggregator::Memory::D_OUT:
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
void CostAggregator::write( CostAggregator::Memory mem, 
						 void *ptr, bool block, const std::vector<cl::Event> *events, 
							cl::Event *event)
{
    if (staging == Staging::I || staging == Staging::IO)
    {
         switch (mem)
         {
			case CostAggregator::Memory::D_IN:
                 if (ptr != nullptr)
                    std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                 queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferSizeIn, hPtrIn, events, event);
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
void* CostAggregator::read ( CostAggregator::Memory mem, bool block, 
							const std::vector<cl::Event> *events, cl::Event *event)
{
     if (staging == Staging::O || staging == Staging::IO)
     {
         switch (mem)
         {
		 case CostAggregator::Memory::H_OUT:
				 queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSizeIn, hPtrOut, events, event);
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
void CostAggregator::init(int _width, int _height, int _d_min, int _d_max, int _radius, Staging _staging)
{
	width = _width; height = _height; radius = _radius;
	numLayers = _d_max - _d_min + 1;
	bufferSizeIn = numLayers*width * height * sizeof (cl_float);
	subBufferSize = width * height * sizeof(cl_float);
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
        std::cerr << "Error[BFCostAggregator]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}

    // Create staging buffers
    bool io = false;
    switch (staging)
    {
        case Staging::NONE:
             hPtrIn = nullptr;
             hPtrOut = nullptr;
             break;

        case Staging::IO:
             io = true;

        case Staging::I:
             if (hBufferIn () == nullptr)
                 hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSizeIn);

			 hPtrIn = (cl_float *) queue.enqueueMapBuffer (
				         hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSizeIn);

             queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

             if (!io)
             {
                 queue.finish ();
                 hPtrOut = nullptr;
                 break;
             }

        case Staging::O:
            if (hBufferOut () == nullptr)
                hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSizeIn);

             hPtrOut = (cl_float *) queue.enqueueMapBuffer (
					         hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSizeIn);
             queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
             queue.finish ();

             if (!io)
				 hPtrIn = nullptr;

             break;
	}

	// Create device buffers
    if (dBufferIn () == nullptr)
        dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSizeIn);

	// Out buffer is of size d_levels *width *height
    if (dBufferOut () == nullptr)
        dBufferOut = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSizeIn);

	cl_buffer_region region = {0, subBufferSize};
	dSubBufferIn = dBufferIn.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);
	dSubBufferOut = dBufferOut.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);

	box_filter.get ( cl_algo::GF::BoxFilterSAT::Memory::D_IN ) = dSubBufferIn;
	box_filter.get ( cl_algo::GF::BoxFilterSAT::Memory::D_OUT ) = dSubBufferOut;
	box_filter.init( width, height, radius, scaling, cl_algo::GF::Staging::IO);

}

//-------------------------------------------------------------------------------------------------------
/*! \details The function call is non-blocking.
 *
 *  \param[in] events a wait-list of events.
 *  \param[out] event event associated with the kernel execution.
 */
void CostAggregator::run (const std::vector<cl::Event> *events, cl::Event *event)
{
	try
	{
		// Loop box filter through different layers
		for(int i=0; i<numLayers; i++){
			cl_buffer_region region = {i*subBufferSize, subBufferSize};

			box_filter.get ( cl_algo::GF::BoxFilterSAT::Memory::D_IN ) = dBufferIn.createSubBuffer(CL_MEM_READ_ONLY, 
																				CL_BUFFER_CREATE_TYPE_REGION, &region);
			box_filter.get (cl_algo::GF::BoxFilterSAT::Memory::D_OUT ) = dBufferOut.createSubBuffer(CL_MEM_READ_WRITE, 
																				CL_BUFFER_CREATE_TYPE_REGION, &region);
			// Run box filter.
			box_filter.run();
		}
	}
	catch (const char *error)
    {
        std::cerr << "Error[CostVolume]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}
}