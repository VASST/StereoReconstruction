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

#include <GrayscaleFilter.hpp>
#include <GradientFilter.hpp>
#include <CostVolume.hpp>
#include <vtkGuidedFilterAlgo.hpp>
#include <BFCostAggregator.hpp>
#include <DisparityOptimizer.hpp>
#include <pgm.h>
#include <JointWMF.hpp>

// Opencv includes
#include <cv.hpp>
#include <highgui.h>

// For debugging
void compute_cost_volume(cv::Mat *, cv::Mat *);
cv::Mat get_ground_truth(const char *);

// Slider call-back
void on_trackbar(int, void*);
void on_trackbar_debug(int, void *);
unsigned int width, height;

cl_float *cost_slice_mem, *cost_out, *filtered_cost_out, *cost_cv_debug;
cv::Mat cost, filtered_cost;


int main(int argc, char* argv)
{
	//cv::Mat imgL = cv::imread("../Data/demoL.jpg");
	//cv::Mat imgR = cv::imread("../Data/demoR.jpg");
	//cv::Mat imgL = cv::imread("../Data/000001-hL.png");
	//cv::Mat imgR = cv::imread("../Data/000001-hR.png");
	//cv::Mat imgL = cv::imread("../Data/im-dL.png");
	//cv::Mat imgR = cv::imread("../Data/im-dL.png");
	//cv::Mat imgL = cv::imread("../Data/tsukuba/imL.png");
	//cv::Mat imgR = cv::imread("../Data/tsukuba/imR.png");
	cv::Mat imgL = cv::imread("../Data/teddy/im-dL.png");
	cv::Mat imgR = cv::imread("../Data/teddy/im-dR.png");

	//cv::imshow("Test", imgL);
	//cv::waitKey(0);


	//std::string ground_truth_file = "../Data/tsukuba/disp2.pgm";
	//get_ground_truth(ground_truth_file.c_str());

	

	/*cv::Mat temp(480/2, 640/2, imgL.type());
	cv::resize(imgL, temp, cv::Size(320, 240),CV_INTER_LINEAR);
	cv::imwrite("../Data/teddy/im-dL.png", temp);
	cv::resize(imgR, temp, cv::Size(320, 240),CV_INTER_LINEAR);
	cv::imwrite("../Data/teddy/im-dR.png", temp); */

	width = imgL.cols;  height = imgL.rows;
	const unsigned int bufferSize = width * height * sizeof (cl_float);
	const unsigned int channels(imgL.channels());

	//compute_cost_volume(&imgL, &imgR);

	const unsigned int gfRadius = 5;
    const float gfEps = 0.1;
	const int d_max = 30; 
	const int d_min = 10;
	const int color_th = 7;
	const int grad_th = 4;
	const double alpha = 0.6;

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
	clutils::CLEnvInfo<1> infoRGB2Gray1(0, 0, 0, v, 0);
	GrayscaleFilter I1( clEnv, infoRGB2Gray1);
	I1.get(GrayscaleFilter::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
	I1.init( width, height, GrayscaleFilter::Staging::IO);

	std::vector<unsigned int> v2;
	v2.push_back( 0 );
	clutils::CLEnvInfo<1> infoRGB2Gray2(0, 0, 0, v2, 0);
	GrayscaleFilter I2( clEnv, infoRGB2Gray2);
	I2.get(GrayscaleFilter::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
	I2.init( width, height, GrayscaleFilter::Staging::IO);

	// Configure kernel execution parameters for Dense Stereo (GradF_L/R)
	std::vector<unsigned int> v3;
	v3.push_back( 0 );
    clutils::CLEnvInfo<1> infoGradF_L (0, 0, 0, v3, 0);
	GradientFilter GradF_L( clEnv, infoGradF_L);
	GradF_L.get( GradientFilter::Memory::D_IN) = I1.get(GrayscaleFilter::Memory::D_OUT);
	GradF_L.get( GradientFilter::Memory::D_X_OUT) = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize);
	GradF_L.init( width, height, GradientFilter::Staging::O); 

	std::vector<unsigned int> v4;
	v4.push_back( 0 );
    clutils::CLEnvInfo<1> infoGradF_R (0, 0, 0, v4, 0);
	GradientFilter GradF_R( clEnv, infoGradF_R);
	GradF_R.get( GradientFilter::Memory::D_IN) = I2.get(GrayscaleFilter::Memory::D_OUT);
	GradF_R.get( GradientFilter::Memory::D_X_OUT) = cl::Buffer( context, CL_MEM_READ_WRITE, bufferSize);
	GradF_R.init( width, height, GradientFilter::Staging::O); 

	// Configure CostVolume (CV)
	std::vector<unsigned int> v5;
	v5.push_back( 0 );
	clutils::CLEnvInfo<1> infoCV( 0, 0, 0, v5, 0);
	CostVolume CV( clEnv, infoCV);
	CV.get ( CostVolume::Memory::D_IN_L ) = I1.get(GrayscaleFilter::Memory::D_OUT);
	CV.get ( CostVolume::Memory::D_IN_LGRAD ) = GradF_L.get( GradientFilter::Memory::D_X_OUT);
	CV.get ( CostVolume::Memory::D_IN_R ) = I2.get(GrayscaleFilter::Memory::D_OUT);
	CV.get ( CostVolume::Memory::D_IN_RGRAD ) = GradF_R.get( GradientFilter::Memory::D_X_OUT);
	CV.init( width, height, d_min, d_max, 1, color_th, grad_th, alpha, 2, CostVolume::Staging::O);

	// Configure guided filter (GF)
/*	std::vector<unsigned int> v6;
	v6.push_back( 0 );
	v6.push_back( 1 );
    clutils::CLEnvInfo<2> infoGF (0, 0, 0, v6, 0);
	const cl_algo::GF::GuidedFilterConfig Ip = cl_algo::GF::GuidedFilterConfig::I_NEQ_P;
    cl_algo::GF::GuidedFilter<Ip> GF (clEnv, infoGF);
    GF.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_IN_I) = I1.get (GrayscaleFilter::Memory::D_OUT);
	GF.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_IN_P) = CV.get (CostVolume::Memory::D_OUT);
    GF.get (cl_algo::GF::GuidedFilter<Ip>::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
    GF.init (width, height, gfRadius, gfEps, 0, 0.01f, cl_algo::GF::Staging::O);  */

	std::vector<unsigned int> v7;
	v7.push_back(0);
	clutils::CLEnvInfo<1> infoBF(0, 0, 0, v7, 0);
	CostAggregator CA (clEnv, infoBF);
	CA.get( CostAggregator::Memory::D_IN) = CV.get( CostVolume::Memory::D_OUT);
	CA.init(width, height, d_min, d_max, gfRadius, CostAggregator::Staging::O);

	std::vector<unsigned int> v8;
	v8.push_back(0);
	clutils::CLEnvInfo<1> infoOptimizer(0, 0, 0, v8, 0);
	DisparityOptimizer DO(clEnv, infoOptimizer);
	DO.get( DisparityOptimizer::Memory::D_IN ) = CA.get( CostAggregator::Memory::D_OUT);
	DO.init( width, height, d_min, d_max, DisparityOptimizer::Staging::O); 

	// Start timing.
	auto t_start = std::chrono::high_resolution_clock::now();

	// Copy data to device
	I1.write (GrayscaleFilter::Memory::D_IN, (void*)imgL.datastart);
	I2.write (GrayscaleFilter::Memory::D_IN, (void*)imgR.datastart);
        
	// Execute kernels	   
	cl::Event eventL, eventR, cv_event;
    std::vector<cl::Event> waitListL (1), waitListR (1), cvList (1);
    I1.run (nullptr, &eventL); waitListL[0] = eventL;
	I2.run(nullptr, &eventR); waitListR[0] = eventR;
	GradF_L.run( &waitListL );
	GradF_R.run( &waitListR );
	CV.run (nullptr, &cv_event); cvList[0] = cv_event;
	CA.run ();
	DO.run ();

	// Copy results to host	
	cl_float *disparity = (cl_float *)DO.read ();

	// End time.
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Elapsed time  : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
              << " ms." << std::endl;

	cost_out = (cl_float *)CV.read ();
	filtered_cost_out = (cl_float*)CA.read();		

	cv::imshow("Left_Image", imgL);
	cv::moveWindow("Left_Image", 0, 0);
	cv::imshow("Right_Image", imgR);
	cv::moveWindow("Right_Image", width, 0);

	cv::Mat disparity_img_f(height, width, CV_32FC1, disparity);
	double min, max;
	cv::minMaxIdx( disparity_img_f, &min, &max);
	cv::Mat disparity_img(height, width, CV_8UC1);
	disparity_img_f.convertTo( disparity_img, CV_8UC1, 255.0/(max-min), -min*255.0/(max-min)); 
	cv::imshow("Disparity", disparity_img );
	cv::moveWindow("Disparity", 4*width, 0);

	// WMF the disparity image
	cv::Mat WMF_disparity = JointWMF::filter(disparity_img, imgL, 3);

	cv::Mat color_map;
	cv::applyColorMap(WMF_disparity, color_map, cv::COLORMAP_COOL);
	cv::imshow("Filtered_Colored_Disparity", color_map);
	cv::moveWindow("Filtered_Colored_Disparity", 4*width, height+50);


	int cost_slice_num = 0;
	cost_slice_mem = new float[ width*height ];
	
	cv::namedWindow("Cost");
	cv::createTrackbar("Slice_No", "Cost", &cost_slice_num, (d_max-d_min), on_trackbar, (void*)cost_out );

	cv::namedWindow("Filtered_Cost");
	cv::createTrackbar("Slice_No", "Filtered_Cost", &cost_slice_num, (d_max-d_min), on_trackbar, (void*)filtered_cost_out);
	on_trackbar(0, (void*)cost_out);

	/* For debugging */
	/*cl_float *debug_out = (cl_float *) I1.read();
	
	cv::Mat temp(height, width, CV_32FC1, debug_out);
	double min, max;
	cv::minMaxIdx( temp, &min, &max);

	cv::imshow("Debug", temp); */

	cv::waitKey(0);

	// Release memory
	delete cost_slice_mem;

	return 0;
}

void on_trackbar(int i , void* data)
{
	cost = cv::Mat(height, width, CV_32FC1);
	memcpy(cost.data, cost_out + width*height*i, sizeof(cl_float)*width*height);	
	filtered_cost = cv::Mat(height, width, CV_32FC1);
	memcpy(filtered_cost.data, filtered_cost_out + width*height*i, sizeof(cl_float)*width*height);	

	// output cost needs to be rescaled to 0-255 range.
	double min, max;
	cv::minMaxIdx( cost, &min, &max);
	cv::Mat temp(height, width, CV_8UC1);
	cost.convertTo( temp, CV_8UC1, 255/(max-min)); 
	cv::imshow("Cost", temp );
	cv::moveWindow("Cost", 2*width, 0);

	cv::minMaxIdx( filtered_cost, &min, &max);
	cv::Mat temp2(height, width, CV_8UC1);
	filtered_cost.convertTo( temp2, CV_8UC1, 255/(max-min));	
	cv::imshow("Filtered_Cost", temp2);
	cv::moveWindow("Filtered_Cost", 3*width, 0);
}

void compute_cost_volume(cv::Mat *img1, cv::Mat *img2)
{
	cv::Mat imgG1(img1->size(), CV_8UC1);
	cv::Mat imgG2(img2->size(), CV_8UC1);
	cv::Mat imgGrad1(img1->size(), CV_8UC1);
	cv::Mat imgGrad2(img1->size(), CV_8UC1);

	cv::cvtColor( *img1, imgG1, CV_RGB2GRAY);
	cv::cvtColor( *img2, imgG2, CV_RGB2GRAY);

	// Gradient
	cv::Sobel(imgG1, imgGrad1, CV_8UC1, 1, 0); 
	cv::Sobel(imgG2, imgGrad2, CV_8UC1, 1, 0); 

	int d_max = 15;
	int d_min = -15;
	float alpha = 0.5;
	
	cost_cv_debug = new float[ img1->rows*img1->cols*(d_max-d_min+1)];

	for( int d=d_min; d<d_max; d++)
	{
		for(int y=0; y<imgG1.rows; y++)
		{
			for(int x=0; x<imgG1.cols; x++)
			{
				if( 0 < d+x && d+x < width )
				{
					float color_cost = (float)abs(imgG1.data[ y*imgG1.cols + x ] - imgG2.data[ y*imgG2.cols + x + d ] );
					float grad_cost = (float)abs(imgGrad1.data[ y*imgG1.cols + x ] - imgGrad2.data[ y*imgG2.cols + x + d ] );
					cost_cv_debug[ abs(d-d_min)*img1->rows*img1->cols + y*img1->cols + x ] = (1-alpha)*color_cost + alpha*grad_cost;
				}
				else
					cost_cv_debug[ abs(d-d_min)*img1->rows*img1->cols + y*img1->cols + x ] = 0;
			}
		}
	}

	int i=0;
	cv::namedWindow("Cost_Debug");
	cv::createTrackbar("Cost_Debug_TB", "Cost_Debug", &i, d_max-1, on_trackbar_debug);
	on_trackbar_debug(0, (void*)cost_cv_debug);
}

void on_trackbar_debug(int i , void* data)
{
	cv::Mat out(height, width, CV_32FC1);
	memcpy(out.data, cost_cv_debug + i*height*width, height*width*sizeof(float));

	cv::Mat temp(height, width, CV_8UC1);
	double min, max;
	cv::minMaxIdx( out, &min, &max);
	out.convertTo( temp, CV_8UC1, 255/(max-min)); 

	cv::imshow("Cost_Debug", temp);
}

cv::Mat get_ground_truth(const char * filename)
{
	PGMImage *img = new PGMImage();

	// Read PGM file
	getPGMfile( (char*)filename , img);

	cv::Mat out(img->height, img->width, CV_8UC1, img->data);
	cv::imshow("GT", out);

	return out;
}