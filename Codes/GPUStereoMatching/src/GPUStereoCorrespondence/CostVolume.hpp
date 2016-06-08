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
#ifndef COSTVOLUME_HPP
#define COSTVOLUME_HPP

#include <vtkCLUtils.hpp>

class CostVolume
{
public:
	/*! \brief Enumerates the memory objects handled by the class.
     *  \note `H_*` names refer to staging buffers on the host.
     *  \note `D_*` names refer to buffers on the device.
     */
     enum class Memory : uint8_t
    {
          H_IN_LR,   /*!< Input staging buffer for left image channel R. */
		  H_IN_LG,   /*!< Input staging buffer for left image channel G. */
		  H_IN_LB,   /*!< Input staging buffer for left image channel B. */
		  H_IN_RR,   /*!< Input staging buffer for right image channel R. */
		  H_IN_RG,   /*!< Input staging buffer for right image channel G. */
		  H_IN_RB,   /*!< Input staging buffer for right channel B. */
		  H_IN_LGRAD, /*!< Input staging buffer for left image gradient. */
		  H_IN_RGRAD, /*!< Input staging buffer for right image gradient. */
          H_OUT,  /*!< Output staging buffer. */
          D_IN_LR,   /*!< Input buffer for left image channel R. */
		  D_IN_LG,   /*!< Input buffer for left image channel G. */
		  D_IN_LB,   /*!< Input buffer for left image channel B. */
		  D_IN_RR,   /*!< Input buffer for right image channel R. */
		  D_IN_RG,   /*!< Input buffer for right image channel G. */
		  D_IN_RB,   /*!< Input buffer for right image channel B. */
		  D_IN_LGRAD, /*!< Input buffer for left image gradient. */
		  D_IN_RGRAD, /*!< Input buffer for right image gradient. */
          D_OUT,  /*!< Output buffer. */
    };

	/*! \brief Enumerates staging buffer configurations.
     *  \details It's meant to be used when making a call to the `init` 
     *           method of one of the `cl_algo` classes. 
     *           It specifies which staging buffers to be instantiated.
     */
    enum class Staging : uint8_t
    {
        NONE,  /*!< Do not instantiate any staging buffers. */
        I,     /*!< Instantiate the input staging buffers. */
        O,     /*!< Instantiate the output staging buffers. */
        IO     /*!< Instantiate both input and output staging buffers. */
    };
	 
	/*! \brief Configures an OpenCL environment as specified by `_info`. */
    CostVolume (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);

	/*! \brief Default destructor */
	~CostVolume();
        
	/*! \brief Returns a reference to an internal memory object. */
    cl::Memory& get (CostVolume::Memory mem);
        
	/*! \brief Configures kernel execution parameters. */
    void init (int _width, int _height, int _d_min, int _d_max, int _cth, int _gth, double _alpha, Staging _staging = Staging::IO);

    /*! \brief Performs a data transfer to a device buffer. */
    void write (CostVolume::Memory mem = CostVolume::Memory::D_IN_LR, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        
	/*! \brief Performs a data transfer to a staging buffer. */
    void* read (CostVolume::Memory mem = CostVolume::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        
	/*! \brief Executes the necessary kernels. */
    void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

	cl_float *hLRPtrIn;  /*!< Mapping of the input staging buffer for left image R channel. */
	cl_float *hLGPtrIn;  /*!< Mapping of the input staging buffer for left image G channel. */
	cl_float *hLBPtrIn;  /*!< Mapping of the input staging buffer for left image B channel. */
	cl_float *hRRPtrIn;  /*!< Mapping of the input staging buffer for right image R channel. */
	cl_float *hRGPtrIn;  /*!< Mapping of the input staging buffer for right image G channel. */
	cl_float *hRBPtrIn;  /*!< Mapping of the input staging buffer for right image B channel. */
	cl_float *hLGRADPtrIn;  /*!< Mapping of the input staging buffer for left image gradient. */
	cl_float *hRGRADPtrIn;  /*!< Mapping of the input staging buffer for left image gradient. */
    cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

private:
	clutils::CLEnv &env;
    clutils::CLEnvInfo<1> info;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel cost_calc;
    cl::NDRange global;
    Staging staging;
    unsigned int width, height, bufferSize, outBufferSize;
    cl::Buffer hLRBufferIn, hLGBufferIn, hLBBufferIn, hRRBufferIn, hRGBufferIn, hRBBufferIn, hLGRADBufferIn, hRGRADBufferIn, hBufferOut;
    cl::Buffer dLRBufferIn, dLGBufferIn, dLBBufferIn, dRRBufferIn, dRGBufferIn, dRBBufferIn, dLGRADBufferIn, dRGRADBufferIn, dBufferOut;
    cl::Event cost_calc_event;
    std::vector<cl::Event> waitList_cost_calc;

	/* Parameters for cost construction */
	double color_th, grad_th, alpha;
	unsigned int d_min, d_max;
};

#endif // COSTVOLUME_HPP

