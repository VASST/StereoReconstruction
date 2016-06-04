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
#include <vector> 
#include <vtkCLUtils.hpp>

#include <GPUStereoDenseCorrespondence.hpp>
#include <vtkGuidedFilterAlgo.hpp>

// Opencv includes
#include <cv.hpp>
#include <highgui.h>

int main(int argc, char* argv)
{
	cv::Mat imgL = cv::imread("../Data/000100LD4.png");
	cv::Mat imgR = cv::imread("../Data/000100RD4.png");

	const unsigned int width(imgL.cols), height(imgL.rows);
	const unsigned int bufferSize = width * height * sizeof (cl_float);

	// Setup CL environment
	std::vector< std::string > kernel_file;
	kernel_file.push_back("../Kernels/vtkGuidedFilter.cl"); // Filter kernels
	
	clutils::CLEnv clEnv( kernel_file );
	cl::Context context = clEnv.getContext(); 
	clEnv.addQueue (0, 0);  // Adds a second queue

	// Configure kernel execution parameters
	std::vector<unsigned int> v;
	v.push_back( 0 );
    clutils::CLEnvInfo<1> infoRGB (0, 0, 0, v, 0);
    const cl_algo::GF::SeparateRGBConfig C1 = cl_algo::GF::SeparateRGBConfig::UCHAR_FLOAT;
    cl_algo::GF::SeparateRGB<C1> rgb (clEnv, infoRGB);
    rgb.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_G) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_B) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb.init (width, height, cl_algo::GF::Staging::I);

	// Configure kernel execution parameters for Dense Stereo (DS)
	std::vector<unsigned int> v2;
	v2.push_back( 0 );
    clutils::CLEnvInfo<1> infoDS (0, 0, 0, v2, 0);
	GPUStereoDenseCorrespondence DS( clEnv, infoDS);
	DS.get( GPUStereoDenseCorrespondence::Memory::D_IN) = rgb.get(cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R);
	DS.get( GPUStereoDenseCorrespondence::Memory::D_OUT) = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize);
	DS.init( width, height, GPUStereoDenseCorrespondence::Staging::O); 


	auto t_start = std::chrono::high_resolution_clock::now();
	// Copy data to device
	rgb.write (cl_algo::GF::SeparateRGB<C1>::Memory::D_IN, (void*)imgL.datastart);
        
	// Execute kernels	   
	cl::Event event;
    std::vector<cl::Event> waitList (1);
    rgb.run (nullptr, &event); waitList[0] = event;
	DS.run( &waitList );

	// Copy results to host
    cl_float *results = (cl_float *) DS.read ();

	// End time
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Elapsed time  : "
              << std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()
              << " us." << std::endl;

	cv::imshow("Left_Image", imgL);
	cv::moveWindow("Left_Image", 0, 0);
	cv::imshow("Right_Image", imgR);
	cv::moveWindow("Right_Image", width, 0);
	cv::imshow("Del_X", cv::Mat(height, width, CV_32FC1, results));
	cv::moveWindow("Del_X", 2*width, 0);

	cv::waitKey(0);


	return 0;
}