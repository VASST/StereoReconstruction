% main script. 

addpath ./miccai2013/
addpath('C:\Libs\build\prost\matlab\Release')
addpath('C:\Libs\src\prost\matlab')



% clear screen 
clc
close all

gpu_enable = false;
if(gpuDeviceCount > 0)
    gpu_enable = true;
    
    gpu = gpuDevice(1);
    setenv('CUDA_CACHE_MAXSIZE','536870912')
end

if(~exist(['HuberL1CVPrecond_mex.',mexext]))
   % compile HuberL1CVPrecond_mex.cu
   compile_HuberL1CVPrecond_mex
end

% Matching method
method = 'miccai2013';


if(gpu_enable)
    
    % Read-in input images
%     left_img_file = './images/im-dL.png';
%     right_img_file = './images/im-dR.png';
%     
%     left_img = im2single(rgb2gray(imread(left_img_file)));
%     right_img = im2single(rgb2gray(imread(right_img_file)));
    
    % Read-in video
    left_frames = read_video_file('./images/f7_dynamic_deint_L.avi');
    right_frames = read_video_file('./images/f7_dynamic_deint_R.avi');
    ground_truth_prefix = './images/f7_dynamic_deint/disparityMap_';
    
    % Set camera params
    left_intrinsics = [391.656525, 0.000000, 165.964371; 
                       0.000000, 426.835144, 154.498138;  
                       0.000000, 0.000000, 1.000000];
    left_distortions = [-0.196312 0.129540 0.004356 0.006236];
    
    right_intrinsics = [390.376862 0.000000 190.896454;
                        0.000000 426.228882 145.071411;
                        0.000000 0.000000 1.000000];
    right_distortions = [-0.205824 0.186125 0.015374 0.003660];
    
    stereo_cam_rot = [0.999999 -0.001045 -0.000000;
                      0.001045 0.999999 -0.000000;
                      0.000000 0.000000 1.000000];
                  
    stereo_cam_trans = [-5.520739 -0.031516 -0.051285];
    
    left_cam_params = cameraParameters('IntrinsicMatrix', left_intrinsics', ...
                                        'RadialDistortion', left_distortions(1:2), ...
                                        'TangentialDistortion', left_distortions(3:4));
   
    right_cam_params = cameraParameters('IntrinsicMatrix', right_intrinsics', ...
                                        'RadialDistortion', right_distortions(1:2), ...
                                        'TangentialDistortion', right_distortions(3:4));
    
    stereo_params = stereoParameters(left_cam_params,...
                                    right_cam_params,...
                                    stereo_cam_rot, ...
                                    stereo_cam_trans');   
                                
                                
    frame_no = 10;
    I1 = im2single(rgb2gray(left_frames(:,:,:,frame_no)));
    I2 = im2single(rgb2gray((right_frames(:,:,:,frame_no))));
    
    % Rectify images
    disp('Rectifying image...');
    [left_img, right_img] = rectifyStereoImages(I1, I2, stereo_params, 'OutputView', 'valid');
    figure, imshowpair(left_img, right_img, 'montage');
    title('Input Images')
    figure, imshowpair(left_img, right_img, 'falsecolor', 'ColorChannels', 'red-cyan');
                                  
    
    width = size(left_img, 2);
    height = size(left_img, 1);
    true_disparity = read_ground_truth_disparity([ground_truth_prefix, num2str(frame_no-1),...
                                                    '.txt'], ...
                                                      size(I1,2), size(I1,1));
                                                  
        
    % Cost volume parameters
    CostVolumeParams = struct('min_disp', uint8(0), ...
                             'max_disp', uint8(32), ...
                             'method', 'zncc', ...
                             'win_r', uint8(4), ...
                             'ref_left', true);
                         
if(strcmp(method, 'miccai2013'))
%% Chang et. al, MICCAI 2013
    disp('Running Chang et. al., MICCAI 2013..')
    s = sprintf('Size of the cost volume %dx%dx%d', size(left_img, 1), ...
                                                        size(left_img, 2), ...
                                                          CostVolumeParams.max_disp - CostVolumeParams.min_disp);
    disp(s);

    [disparity, disparity_color, err] = StereoReconstHuberL1(left_img, right_img, CostVolumeParams);
    
    % Plot results
    figure;
    imshow(mat2gray(disparity));
    
    figure, imshow(disparity_color);  
    
    figure, imshow(mat2gray(true_disparity))
    title('Truth')    
        
    figure;
    plot(err, 'g');    
    grid on;
    legend('HuberL1+Cost-Volume');
    xlabel('Iterations');
    ylabel('Energy function');

elseif(strcmp(method, 'sublabel_lifting')) 
%% Functional Lifting

    if(exist('cost_volume.mat'))
        load('cost_volume.mat');
        
        
    if ~exist('stereo_cost')            
    %% compute matching cost
    ndisps = 64;
    stereo_cost = compute_stereo_cost(...
        'images/im-dL.png', ...
        'images/im-dR.png', ndisps, 1, 1);
    end
     
        %% setup parameters
        L = 32;
        gamma = linspace(0, 1, L)'; 
        lmb = 0.3;

        %% solve problem and display result
        tic;
        [u_unlifted] = sublabel_lifting_convex(double(cost_volume), gamma, lmb);
        toc;

        [ny, nx, ~] = size(cost_volume);  
        u_unlifted = min(max(u_unlifted, 0), 1);
        num_colors = 65536;
        cmap = jet(num_colors);
        cmap_index = 1 + round(u_unlifted * (num_colors - 1));
        image_rgb = reshape(cmap(cmap_index,:),ny,nx,3);
        imshow(image_rgb)
        
%         cmap_index = 1 + round(true_disparity*(num_colors-1));
%         true_map = reshape(cmap(cmap_index,:),ny,nx,3);
%         imshowpair(true_disparity, reshape(u_unlifted, ny, nx), 'montage');       
        
        
    else
        disp('Cost volume does not exist');
    end
end
    

%% Release GPU memory
    reset(gpu);
    
else
    disp('Error: No GPU was found.')
end