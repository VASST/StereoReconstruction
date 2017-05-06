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
#include <DisparityOptimizer.hpp>
#include <COCV.hpp>

#include <pgm.h>
#include <JointWMF.hpp>

// PCL includes
//#include <pcl/io/io.h>
//#include <pcl/io/ply_io.h>
//#include <pcl/point_types.h>


// Opencv includes
#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>

//#define VISUALIZE_POINTCLOUD

// For debugging
void compute_cost_volume(cv::Mat *, cv::Mat *);
cv::Mat get_ground_truth(const char *);

// For visualization
// Converts a cv::Mat to a pcl::PointCloud
//void MatToPointCloud(cv::Mat &, cv::Mat &, pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

//This function creates a PCL visualizer, sets the point cloud to view and returns a pointer
//void createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);


// Slider call-back
void on_trackbar(int, void*);
void on_trackbar_debug(int, void *);
unsigned int width, height;

cl_float *cost_slice_mem, *cost_out, *cost_cv_debug;
cl_float *disparity;
cv::Mat cost;

const unsigned int radius = 5;
const int d_max = 60; 
const int d_min = 10;

//#define VISUALIZE_POINTCLOUD

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
	//cv::Mat imgL = cv::imread("../Data/lap-heart1.png");
	//cv::Mat imgR = cv::imread("../Data/lap-heart2.png");

	width = imgL.cols;  height = imgL.rows;
	const unsigned int bufferSize = width * height * sizeof (cl_float);
	const unsigned int channels(imgL.channels());

	/* Camera matrices */

	// Initialize camera matrices
	double l_cam_data[9] = {391.656525, 0.000000, 165.964371,
							  0.000000, 426.835144, 154.498138,
							  0.000000, 0.000000, 1.000000};
	cv::Mat left_intrinsics(3, 3, CV_64F, &l_cam_data );	
	std::cout << "Left intrinsics: " << left_intrinsics << std::endl;

	double l_cam_dist_data[4] = {-0.196312, 0.129540, 0.004356, 0.006236};
	cv::Mat left_distortion_params(1, 4, CV_64F, &l_cam_dist_data );
	std::cout << "Left Distortion params: " << left_distortion_params << std::endl;

	double r_cam_data[9] = {390.376862, 0.000000, 190.896454,
							0.000000, 426.228882, 145.071411,
							0.000000, 0.000000, 1.000000};
	cv::Mat right_intrinsics(3, 3, CV_64F, &r_cam_data );
	std::cout << "Right intrinsics: " << right_intrinsics << std::endl;

	double r_cam_dist_data[4] = {-0.205824, 0.186125, 0.015374, 0.003660};
	cv::Mat right_distortion_params( 1, 4, CV_64F, &r_cam_dist_data );
	std::cout << "Right Distortion params: " << right_distortion_params << std::endl;

	double stereo_calib_data[12] = { 0.999999, -0.001045, -0.000000,
									0.001045, 0.999999, -0.000000,
									0.000000, 0.000000, 1.000000,
									-5.520739, -0.031516, -0.051285};
	cv::Mat stereo_mat(4, 3, CV_64F, &stereo_calib_data );
	std::cout << "Stereo Calibration: " << stereo_mat << std::endl;

	// Rectify images
	cv::Mat R1(3, 3, CV_64F);
	cv::Mat R2(3, 3, CV_64F);
	cv::Mat P1(3, 4, CV_64F);
	cv::Mat P2(3, 4, CV_64F);
	cv::Mat Q(4, 4, CV_64F);

	/* Q matrix is of the following form:
		Q = [ 1 0 0 -cx; 0 1 0 -cy; 0 0  0 f; 0 0 -1/Tx (cx-c'x)/Tx ]; // where Tx is the baseline
	 */
	cv::stereoRectify(left_intrinsics, left_distortion_params, right_intrinsics, right_distortion_params, cv::Size(width, height), 
						stereo_mat(cv::Rect(0, 0, 3, 3)), stereo_mat.row(3).t(), R1, R2, P1, P2, Q, 0);

	std::cout << "t: " << stereo_mat.row(3).t() << std::endl;

	std::cout << "P1: " << P1 << std::endl;
	std::cout << "P2: " << P2 << std::endl;
	std::cout << "Q: " << Q << std::endl;

	cv::Mat mapxL( height, width, CV_32FC1 );
	cv::Mat mapyL( height, width, CV_32FC1 );
	cv::cuda::GpuMat gMapxL(height, width, CV_32FC1 );
	cv::cuda::GpuMat gMapyL(height, width, CV_32FC1 );
	
	cv::Mat mapxR( height, width, CV_32FC1 );
	cv::Mat mapyR( height, width, CV_32FC1 );
	cv::cuda::GpuMat gMapxR(height, width, CV_32FC1 );
	cv::cuda::GpuMat gMapyR(height, width, CV_32FC1 );

	cv::initUndistortRectifyMap(left_intrinsics, left_distortion_params, R1, P1, cv::Size(width, height), CV_32FC1 , mapxL, mapyL);
	gMapxL.upload(mapxL); gMapyL.upload( mapyL ); // Upload maps to GPU memory
	cv::initUndistortRectifyMap(right_intrinsics, right_distortion_params, R2, P2, cv::Size(width, height), CV_32FC1 , mapxR, mapyR);
	gMapxR.upload( mapxR ); gMapyR.upload( mapyR ); // Upload maps to GPU memory

	cv::cuda::GpuMat gImgL(height, width, imgL.type(), imgL.channels() );
	cv::cuda::GpuMat gImgR(height, width, imgL.type(), imgL.channels() );

	cv::cuda::GpuMat gImgL_rectified(height, width, imgL.type(), imgL.channels() );
	cv::cuda::GpuMat gImgR_rectified(height, width, imgL.type(), imgL.channels() );
	cv::Mat imgL_rectified(height, width, imgL.type(), imgL.channels() );
	cv::Mat imgR_rectified(height, width, imgL.type(), imgL.channels() );


	/*cv::VideoCapture capL, capR;
	if(!capL.open("../Data/f5_dynamic_deint_L.avi"))
	{
		std::cout << "Cannot open the left input video" << std::endl;
		return -1;
	}

	if(!capR.open("../Data/f5_dynamic_deint_R.avi"))
	{
		std::cout << "Cannot open the right input video" << std::endl;
		return -1;
	}

	cv::Mat imgL, imgR;
	capL.read( imgL );
	capR.read( imgR );

	cv::resize( imgL, imgL, cv::Size(320, 240),CV_INTER_LINEAR);
	cv::resize( imgR, imgR, cv::Size(320, 240),CV_INTER_LINEAR);
	cv::imwrite("../Data/lap-heart1.png", imgL);
	cv::imwrite("../Data/lap-heart2.png", imgR);  */ 
		
	//cv::imshow("Test", imgL);
	//cv::waitKey(0);


	//std::string ground_truth_file = "../Data/tsukuba/disp2.pgm";
	//get_ground_truth(ground_truth_file.c_str());

	

	/*cv::Mat temp(480/2, 640/2, imgL.type());
	cv::resize(imgL, temp, cv::Size(320, 240),CV_INTER_LINEAR);
	cv::imwrite("../Data/teddy/im-dL.png", temp);
	cv::resize(imgR, temp, cv::Size(320, 240),CV_INTER_LINEAR);
	cv::imwrite("../Data/teddy/im-dR.png", temp); */



	COCV::Settings _stereo_sttings;
	_stereo_sttings.width = width;
	_stereo_sttings.height = height;
	_stereo_sttings.d_min = d_min;
	_stereo_sttings.d_max = d_max;
	_stereo_sttings.alpha = 10;
	_stereo_sttings.beta = 1.0;
	_stereo_sttings.theta = 10.0;
	_stereo_sttings.lambda = 0.5;
	_stereo_sttings.radius = radius;
	_stereo_sttings.eps = 0.1;
	_stereo_sttings.theta_gamma = 1e-6;
	_stereo_sttings.gamma = 1.f;
	_stereo_sttings.n_itr = 50;

	CostVolume::Settings CV_Settings;
	CV_Settings.width = width;
	CV_Settings.height = height;
	CV_Settings.radius = radius;
	CV_Settings.d_max = d_max;
	CV_Settings.d_min = d_min;
	CV_Settings.color_th = 7;
	CV_Settings.gradient_th = 4;
	CV_Settings.alpha = 0.9;
	CV_Settings.type = 0;


	//compute_cost_volume(&imgL, &imgR);

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
	
	// Create CL context
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
	CV.init( CV_Settings, CostVolume::Staging::O);

	std::vector<unsigned int> v6;
	v6.push_back(0);
	clutils::CLEnvInfo<1> infoCOCV(0, 0, 0, v6, 0);
	COCV cocv(clEnv, infoCOCV);
	cocv.get( COCV::Memory::D_COST_IN ) = CV.get ( CostVolume::Memory::D_OUT );
	cocv.get( COCV::Memory::D_IMG_IN )  = I1.get( GrayscaleFilter::Memory::D_OUT );
	cocv.init(_stereo_sttings, COCV::Staging::O );


	// Start timing.
	auto t_start = std::chrono::high_resolution_clock::now();

	// Copy image data to device
/*	gImgL.upload( imgL );
	gImgR.upload( imgR );

	// Rectified RAW images
	cv::cuda::remap( gImgL, gImgL_rectified, gMapxL, gMapyL, cv::INTER_CUBIC);
	//cv::cuda::remap( gImgR, gImgR_rectified, gMapxR, gMapyR, cv::INTER_LANCZOS4);

	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Elapsed time  : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
              << " ms." << std::endl; */

	cv::remap(imgL, imgL_rectified, mapxL, mapyR, cv::INTER_LANCZOS4);
	cv::remap( imgR, imgR_rectified, mapxR, mapyR, cv::INTER_LANCZOS4);

	I1.write (GrayscaleFilter::Memory::D_IN, (void*)imgL_rectified.datastart);
	I2.write (GrayscaleFilter::Memory::D_IN, (void*)imgR_rectified.datastart);
        
	// Execute kernels	   
	cl::Event eventL, eventR, cv_event;
    std::vector<cl::Event> waitListL (1), waitListR (1), cvList (1);
    I1.run (nullptr, &eventL); waitListL[0] = eventL;
	I2.run(nullptr, &eventR); waitListR[0] = eventR;
	GradF_L.run( &waitListL );
	GradF_R.run( &waitListR );
	CV.run (nullptr, &cv_event); cvList[0] = cv_event;
	cocv.run();

	// Copy results to host	
	disparity = (cl_float *)cocv.read ();

	// End time.
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Elapsed time  : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
              << " ms." << std::endl;


	cost_out = (cl_float *)CV.read ();

	cv::imshow("Left_Image", imgL);
	cv::moveWindow("Left_Image", 0, 0);
	cv::imshow("Right_Image", imgR);
	cv::moveWindow("Right_Image", width, 0);

	cv::Mat disparity_img_f(height, width, CV_32FC1, disparity);
	double min, max;
	cv::minMaxIdx( disparity_img_f, &min, &max);
	cv::normalize(disparity_img_f, disparity_img_f, 0, 1, CV_MINMAX );
	cv::imshow("WTA_Disparity", disparity_img_f );
	cv::moveWindow("WTA_Disparity", 3*width, 0);
	cv::imwrite( "Disparity_map.png", disparity_img_f);

	int cost_slice_num = 0;
	cost_slice_mem = new float[ width*height ];
	
	cv::namedWindow("Cost");
	cv::createTrackbar("Slice_No", "Cost", &cost_slice_num, (d_max-d_min), on_trackbar, (void*)cost_out );
	on_trackbar(0, (void*)cost_out);

	/* For debugging */
/*	cl_float *debug_out = (cl_float *) GradF_L.read( GradientFilter::Memory::H_Y_OUT);
	
	cv::Mat temp(height, width, CV_32FC1, debug_out);
	double min, max;
	cv::minMaxIdx( temp, &min, &max);

	cv::imshow("Debug", temp); */

#ifdef VISUALIZE_POINTCLOUD

	// Reproject points to 3D
	// The output of the COCV is a normalized disparity. Rescale this first. 
	cv::Mat disparity_img( disparity_img_f.size(), disparity_img_f.type(), disparity_img_f.channels() );
	disparity_img = d_max*disparity_img_f;
	cv::minMaxIdx( disparity_img, &min, &max);

	cv::Mat_<cv::Vec3f> recons3D( disparity_img.rows, disparity_img.cols);
	cv::reprojectImageTo3D(disparity_img, recons3D, Q, false, CV_32F );
	cv::minMaxIdx( recons3D, &min, &max);


	// Create a point cloud
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud( new pcl::PointCloud<pcl::PointXYZRGB> );
	// Convert cv::Mat to PointCloud
	MatToPointCloud(recons3D, imgL, point_cloud);
	
	//write PointCloud to file
	std::cout << " Writing point cloud to file.. " << std::endl;
	 
	if( !pcl::io::savePLYFileASCII( "cloud.ply", *point_cloud ) )
		std::cerr << "Failed to write point cloud to file " << std::endl;


#endif

	cv::waitKey(0); 

	// Release memory
	delete cost_slice_mem;

	return 0;
}

void on_trackbar(int i , void* data)
{
	cost = cv::Mat(height, width, CV_32FC1);
	memcpy(cost.data, cost_out + width*height*i, sizeof(cl_float)*width*height);	

	// output cost needs to be rescaled to 0-255 range.
	double min, max;
	cv::minMaxIdx( cost, &min, &max);
	cv::Mat temp(height, width, CV_8UC1);
	cost.convertTo( temp, CV_8UC1, 255/(max-min)); 
	cv::imshow("Cost", cost );
	cv::moveWindow("Cost", 2*width, 0);
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
					cost_cv_debug[ abs(d-d_min)*img1->rows*img1->cols + y*img1->cols + x ] = (1-0.9)*color_cost + 0.9*grad_cost;
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

/*
void MatToPointCloud(cv::Mat &recons3D, cv::Mat &rgb, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud)
{
	float px, py, pz;
	uchar pr, pg, pb;

	for(int i=0; i<recons3D.rows; i++)
	{
		float *recons_ptr = recons3D.ptr<float>(i);
		uchar  *rgb_ptr    = rgb.ptr<uchar>(i);

		for(int j=0; j<recons3D.cols; j++)
		{

			px = recons_ptr[3*j];
			py = recons_ptr[3*j+1];
			pz = recons_ptr[3*j+2];

			//std::cout << "px: " << px << ", py: " << py << ", pz: " << pz << std::endl;

			// Set RGB info
			pb = rgb_ptr[3*j];
			pg = rgb_ptr[3*j+1];
			pr = rgb_ptr[3*j+2];

			// Insert info into point cloud structure
			pcl::PointXYZRGB point;
			point.x = px;
			point.y = py;
			point.z = pz;
			uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
              static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
			point.rgb = *reinterpret_cast<float*>(&rgb);
			pCloud->push_back( point );
		}
	}

	pCloud->width = (int) pCloud->points.size();
	pCloud->height = 1;
}
*/
