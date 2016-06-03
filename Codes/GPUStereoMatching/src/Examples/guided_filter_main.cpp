/*! \file guidedFilter_image.cpp
 *  \brief An example showcasing the effect of the `Guided Filter` algorithm on RGB images.
 *  \details This example demonstrates the performance of the [Guided Image Filtering]
 *           (http://research.microsoft.com/en-us/um/people/kahe/eccv10/).
 *           It loads an RGB image, performs guided filtering on each of the 
 *           channels separately, and displays the original and filtered 
 *           images on the screen.
 *  \author Nick Lamprianidis
 *  \version 1.2.0
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vtkCLUtils.hpp>
#include <vtkGuidedFilterAlgo.hpp>


// Kernel filenames
/*const std::vector<std::string> kernel_files; /* = { "kernels/imageSupport_kernels.cl", 
                                                "kernels/scan_kernels.cl", 
                                                "kernels/transpose_kernels.cl", 
                                                "kernels/boxFilter_kernels.cl",
                                                "kernels/math_kernels.cl", 
                                                "kernels/guidedFilter_kernels.cl" }; */

// OpenCV Window IDs
const char *WinIDIn = "Input image p";
const char *WinIDOut = "Output image q";


int main (int argc, char **argv)
{
    try
    {
        // The photo originates from http://www.hdwallpaperscool.com
        cv::Mat image = cv::imread ("../Data/demo.jpg", CV_LOAD_IMAGE_COLOR);

        // Initialize parameters
        const unsigned int width = image.cols, height = image.rows;
        const unsigned int bufferSize = width * height * sizeof (cl_float);
        const unsigned int gfRadius = 7;
        const float gfEps = std::pow (0.12, 2);

		std::vector< std::string > kernel_files;
		kernel_files.push_back("../Kernels/vtkGuidedFilter.cl"); // Filter kernels
		/*kernel_files.push_back( "../Kernels/imageSupport_kernels.cl" );
		kernel_files.push_back( "../Kernels/scan_kernels.cl" );
	    kernel_files.push_back( "../Kernels/transpose_kernels.cl" );
		kernel_files.push_back( "../Kernels/boxFilter_kernels.cl" );
		kernel_files.push_back( "../Kernels/math_kernels.cl" );
		kernel_files.push_back( "../Kernels/guidedFilter_kernels.cl" );*/

        // Setup the OpenCL environment
        clutils::CLEnv clEnv (kernel_files);
        cl::Context context = clEnv.getContext ();
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

		std::vector<unsigned int> v2;
		v2.push_back( 0 );
		v2.push_back( 1 );
        clutils::CLEnvInfo<2> infoGF (0, 0, 0, v2, 0);
        const cl_algo::GF::GuidedFilterConfig Ip = cl_algo::GF::GuidedFilterConfig::I_EQ_P;
        cl_algo::GF::GuidedFilter<Ip> gfR (clEnv, infoGF);
        gfR.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_IN) = rgb.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_R);
        gfR.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        gfR.init (width, height, gfRadius, gfEps, 0, 0.0001f, 1.f, cl_algo::GF::Staging::NONE);

        cl_algo::GF::GuidedFilter<Ip> gfG (clEnv, infoGF);
        gfG.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_IN) = rgb.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_G);
        gfG.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        gfG.init (width, height, gfRadius, gfEps, 0, 0.0001f, 1.f, cl_algo::GF::Staging::NONE);

        cl_algo::GF::GuidedFilter<Ip> gfB (clEnv, infoGF);
        gfB.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_IN) = rgb.get (cl_algo::GF::SeparateRGB<C1>::Memory::D_OUT_B);
        gfB.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        gfB.init (width, height, gfRadius, gfEps, 0, 0.0001f, 1.f, cl_algo::GF::Staging::NONE);

        const cl_algo::GF::CombineRGBConfig C2 = cl_algo::GF::CombineRGBConfig::FLOAT_FLOAT;
        cl_algo::GF::CombineRGB<C2> fRGB (clEnv, infoRGB);
        fRGB.get (cl_algo::GF::CombineRGB<C2>::Memory::D_IN_R) = gfR.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_OUT);
        fRGB.get (cl_algo::GF::CombineRGB<C2>::Memory::D_IN_G) = gfG.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_OUT);
        fRGB.get (cl_algo::GF::CombineRGB<C2>::Memory::D_IN_B) = gfB.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_OUT);
        fRGB.init (width, height, cl_algo::GF::Staging::O);

        // Copy data to device
		auto t_start = std::chrono::high_resolution_clock::now();
        rgb.write (cl_algo::GF::SeparateRGB<C1>::Memory::D_IN, (void*)image.datastart);

        // Execute kernels
        cl::Event event;
        std::vector<cl::Event> waitList (1);
        rgb.run (nullptr, &event); waitList[0] = event;
        gfR.run (&waitList);
        gfG.run ();
        gfB.run ();
        fRGB.run ();
        
        // Copy results to host
        cl_float *results = (cl_float *) fRGB.read ();

		// End time
		auto t_end = std::chrono::high_resolution_clock::now();
		std::cout << "Elapsed time  : "
              << std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()
              << " us." << std::endl;

        cv::namedWindow (WinIDIn, cv::WINDOW_AUTOSIZE);
        cv::imshow (WinIDIn, image);
        cv::moveWindow (WinIDIn, 0, 0);
        cv::namedWindow (WinIDOut, cv::WINDOW_AUTOSIZE);
        cv::imshow (WinIDOut, cv::Mat(height, width, CV_32FC3, results));
        cv::moveWindow (WinIDOut, width, 0);

        cv::waitKey (0);

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
	
    return 0;
}
