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

#include "GradientFilter.hpp"

//---------------------------------------------------------------------------------------------------
/*! \param[in] _env opencl environment.
 *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
 *              The class requires **two** `(2)` **command queues** (on the same device).
 */
GradientFilter::GradientFilter (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info):
																env( _env ),  info (_info), 
																context (env.getContext (info.pIdx)), 
																queue0 (env.getQueue (info.ctxIdx, info.qIdx[0])), 
																kernel(env.getProgram (info.pgIdx), "gradient"), 
																waitList_delx (1)
{
}

//-----------------------------------------------------------------------------------------------------
GradientFilter::~GradientFilter()
{
}

//------------------------------------------------------------------------------------------------------------
/*! \details This interface exists to allow CL memory sharing between different kernels.
 *
 *  \param[in] mem enumeration value specifying the requested memory object.
 *  \return A reference to the requested memory object.
 */
cl::Memory& GradientFilter::get (GradientFilter::Memory mem)
{
   switch (mem)
   {
		case GradientFilter::Memory::H_IN:
             return hBufferIn;
        case GradientFilter::Memory::H_X_OUT:
             return hXBufferOut;
		case GradientFilter::Memory::H_Y_OUT:
			return hYBufferOut;
        case GradientFilter::Memory::D_IN:
             return dBufferIn;
        case GradientFilter::Memory::D_X_OUT:
             return dXBufferOut;
		case GradientFilter::Memory::D_Y_OUT:
             return dYBufferOut;
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
void GradientFilter::write( GradientFilter::Memory mem, 
											void *ptr, bool block, const std::vector<cl::Event> *events, 
												cl::Event *event)
{
    if (staging == Staging::I || staging == Staging::IO)
    {
         switch (mem)
         {
			case GradientFilter::Memory::D_IN:
                 if (ptr != nullptr)
                    std::copy ((cl_float *) ptr, (cl_float *) ptr + width * height, hPtrIn);
                 queue0.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
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
void* GradientFilter::read ( GradientFilter::Memory mem, bool block, 
												const std::vector<cl::Event> *events, cl::Event *event)
{
     if (staging == Staging::O || staging == Staging::IO)
     {
         switch (mem)
         {
		 case GradientFilter::Memory::H_X_OUT:
				 queue0.enqueueReadBuffer (dXBufferOut, block, 0, bufferSize, hXPtrOut, events, event);
                  return hXPtrOut;
		 case GradientFilter::Memory::H_Y_OUT:
				 queue0.enqueueReadBuffer (dYBufferOut, block, 0, bufferSize, hYPtrOut, events, event);
                  return hYPtrOut;
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
 *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
 */
void GradientFilter::init(int _width, int _height, Staging _staging)
{
	width = _width; height = _height;
	bufferSize = width * height * sizeof (cl_float);
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
        std::cerr << "Error[GPUStereoDenseCorrespondence]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}

    // Create staging buffers
    bool io = false;
    switch (staging)
    {
        case Staging::NONE:
             hPtrIn = nullptr;
             hXPtrOut = nullptr;
			 hYPtrOut = nullptr;
             break;

        case Staging::IO:
             io = true;

        case Staging::I:
             if (hBufferIn () == nullptr)
                 hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

             hPtrIn = (cl_float *) queue0.enqueueMapBuffer (
				         hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
             queue0.enqueueUnmapMemObject (hBufferIn, hPtrIn);

             if (!io)
             {
                 queue0.finish ();
                 hXPtrOut = nullptr;
				 hYPtrOut = nullptr;
                 break;
             }

        case Staging::O:
            if (hXBufferOut () == nullptr)
                hXBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);
			if (hYBufferOut () == nullptr)
                hYBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

			hXPtrOut = (cl_float *) queue0.enqueueMapBuffer (
					         hXBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
			hYPtrOut = (cl_float *) queue0.enqueueMapBuffer (
					         hYBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);

             queue0.enqueueUnmapMemObject (hXBufferOut, hXPtrOut);
			 queue0.enqueueUnmapMemObject (hYBufferOut, hYPtrOut);
             queue0.finish ();

             if (!io) hPtrIn = nullptr;
             break;
	}

	// Create device buffers
    if (dBufferIn () == nullptr)
        dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
    if (dXBufferOut () == nullptr)
        dXBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);
	if (dYBufferOut () == nullptr)
        dYBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferSize);

	kernel.setArg( 0, dBufferIn );
	kernel.setArg( 1, dXBufferOut );
	kernel.setArg( 2, dYBufferOut );


	// Set workspaces
    global = cl::NDRange (width, height);
}

//-------------------------------------------------------------------------------------------------------
/*! \details The function call is non-blocking.
 *
 *  \param[in] events a wait-list of events.
 *  \param[out] event event associated with the kernel execution.
 */
void GradientFilter::run (const std::vector<cl::Event> *events, cl::Event *event)
{
	try
	{
		cl_int err = queue0.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange);
		waitList_delx[0] = delx_event;
	}
	catch (const char *error)
    {
        std::cerr << "Error[GPUStereoDenseCorrespondence]: " << error << std::endl;
        exit (EXIT_FAILURE);
	}
}