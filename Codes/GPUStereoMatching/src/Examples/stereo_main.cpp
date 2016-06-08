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

#include <GradientFilter.hpp>
#include <CostVolume.hpp>
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

	/* CPU DX computation */
	cv::Mat bgr[3];
	cv::split(imgL, bgr);
	cv::Mat out( width, height, CV_32FC1);
	cv::Sobel( bgr[0], out, CV_32FC1, 1, 0);
	cv::imwrite("Del_x.png", out);
	/*--------------------------------------------------------------------------------------------- */

	// TODO 
	// Gaussian smooth the image to improve derivative computation. 

	// Setup CL environment
	std::vector< std::string > kernel_file;
	kernel_file.push_back("../Kernels/vtkGuidedFilter.cl"); // Filter kernels
	
	clutils::CLEnv clEnv( kernel_file );
	cl::Context context = clEnv.getContext(); 
	clEnv.addQueue (0, 0);  // Adds a second queue

	// Configure kernel execution parameters
	std::vector<unsigned int> v;
	v.push_back( 0 );
    clutils::CLEnvInfo<1> infoRGB_L (0, 0, 0, v, 0);
    const cl_algo::GF::SeparateRGBConfig C1 = cl_algo::GF::SeparateRGBConfig::UCHAR_FLOAT;
    cl_algo::GF::SeparateRGB<C1> rgb_L (clEnv, infoRGB_L);
    rgb_L.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb_L.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_G) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb_L.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_B) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb_L.init (width, height, cl_algo::GF::Staging::I);

	std::vector<unsigned int> v2;
	v2.push_back( 0 );
	clutils::CLEnvInfo<1> infoRGB_R (0, 0, 0, v2, 0);
	cl_algo::GF::SeparateRGB<C1> rgb_R (clEnv, infoRGB_R);
    rgb_R.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb_R.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_G) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb_R.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_B) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    rgb_R.init (width, height, cl_algo::GF::Staging::IO);

	// Configure kernel execution parameters for Dense Stereo (GradF_L/R)
	std::vector<unsigned int> v3;
	v3.push_back( 0 );
    clutils::CLEnvInfo<1> infoGradF_L (0, 0, 0, v3, 0);
	GradientFilter GradF_L( clEnv, infoGradF_L);
	GradF_L.get( GradientFilter::Memory::D_IN) = rgb_L.get(cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R);
	GradF_L.get( GradientFilter::Memory::D_OUT) = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize);
	GradF_L.init( width, height, GradientFilter::Staging::O); 

	std::vector<unsigned int> v4;
	v4.push_back( 0 );
    clutils::CLEnvInfo<1> infoGradF_R (0, 0, 0, v4, 0);
	GradientFilter GradF_R( clEnv, infoGradF_R);
	GradF_R.get( GradientFilter::Memory::D_IN) = rgb_R.get(cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R);
	GradF_R.get( GradientFilter::Memory::D_OUT) = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize);
	GradF_R.init( width, height, GradientFilter::Staging::O); 

	// Configure CostVolume (CV)
	std::vector<unsigned int> v5;
	v5.push_back( 0 );
	clutils::CLEnvInfo<1> infoCV( 0, 0, 0, v5, 0);
	CostVolume CV( clEnv, infoCV);
	CV.get ( CostVolume::Memory::D_IN_LR ) = rgb_L.get( cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R);
	CV.get ( CostVolume::Memory::D_IN_LG ) = rgb_L.get( cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_G);
	CV.get ( CostVolume::Memory::D_IN_LB ) = rgb_L.get( cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_B);
	CV.get ( CostVolume::Memory::D_IN_LGRAD ) = GradF_L.get( GradientFilter::Memory::D_OUT);
	CV.get ( CostVolume::Memory::D_IN_RR ) = rgb_R.get( cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R);
	CV.get ( CostVolume::Memory::D_IN_RG ) = rgb_R.get( cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_G);
	CV.get ( CostVolume::Memory::D_IN_RB ) = rgb_R.get( cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_B);
	CV.get ( CostVolume::Memory::D_IN_RGRAD ) = GradF_R.get( GradientFilter::Memory::D_OUT);
	CV.init( width, height, 5, 30, 7, 2, 0.1, CostVolume::Staging::O);

	// TODO


	auto t_start = std::chrono::high_resolution_clock::now();
	// Copy data to device
	rgb_L.write (cl_algo::GF::SeparateRGB<C1>::Memory::D_IN, (void*)imgL.datastart);
	rgb_R.write (cl_algo::GF::SeparateRGB<C1>::Memory::D_IN, (void*)imgR.datastart);
        
	// Execute kernels	   
	cl::Event eventL, eventR;
    std::vector<cl::Event> waitListL (1), waitListR (1);
    rgb_L.run (nullptr, &eventL); waitListL[0] = eventL;
	rgb_R.run(nullptr, &eventR); waitListR[0] = eventR;
	GradF_L.run( &waitListL );
	GradF_R.run( &waitListR );
	CV.run ();

	// Copy results to host
    cl_float *left_gradient = (cl_float *) GradF_L.read ();
	cl_float *right_gradient = (cl_float *) GradF_R.read ();
	cl_float *cost_out = (cl_float *)CV.read ();
	//cl_float *debug_out = (cl_float *)rgb_R.read (cl_algo::GF::SeparateRGB<C1>::Memory::H_OUT_B );

	// End time
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Elapsed time  : "
              << std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()
              << " us." << std::endl;

	cv::imshow("Left_Image", imgL);
	cv::moveWindow("Left_Image", 0, 0);
	cv::imshow("Right_Image", imgR);
	cv::moveWindow("Right_Image", width, 0);
	cv::imshow("Left_Gradient(x)", cv::Mat(height, width, CV_32FC1, left_gradient));
	cv::moveWindow("Left_Gradient(x)", 2*width, 0);
	cv::imshow("Right_Gradient(x)", cv::Mat(height, width, CV_32FC1, right_gradient));
	cv::moveWindow("Right_Gradient(x)", 3*width, 0);

	int cost_slice_num = 20;
	cl_float *cost_slice_mem = new float[ width*height ];
	memcpy(cost_slice_mem, cost_out + width*height*cost_slice_num, sizeof(cl_float)*width*height); 
	cv::Mat cost(height, width, CV_32FC1, cost_slice_mem);

	/*double min, max;
	cv::minMaxIdx(cost, &min, &max);
	cv::Mat adj;
	cost.convertTo( adj, CV_8UC1, 255/max);*/

	cv::imshow("Cost", cost );
	cv::moveWindow("Cost", 4*width, 0);

	/* For debugging */
	//cv::imshow("Debug", cv::Mat(height, width, CV_32FC1, debug_out));

	cv::waitKey(0);

	// Release memory
	delete cost_slice_mem;

	return 0;
}