% main script. 

addpath ./miccai2013/
addpath('C:\Libs\build\prost\matlab\Release')
addpath('C:\Libs\src\prost\matlab')



% clear screen 
clc

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
    left_img_file = './images/lap-heart1.png';
    right_img_file = './images/lap-heart2.png';
    
    % Read-in video
    left_frames = read_video_file('./images/f7_dynamic_deint_L.avi');
    right_frames = read_video_file('./images/f7_dynamic_deint_R.avi');
    ground_truth_prefix = './images/f7_dynamic_deint/disparityMap_';
    
    %left_img = im2single(rgb2gray(imread(left_img_file)));
    %right_img = im2single(rgb2gray(imread(right_img_file))); 
    frame_no = 1;
    left_img = im2single(rgb2gray(left_frames(:,:,:,frame_no)));
    right_img = im2single(rgb2gray((right_frames(:,:,:,frame_no))));
    width = size(left_img, 2);
    height = size(left_img, 1);
    true_disparity = read_ground_truth_disparity([ground_truth_prefix, num2str(frame_no-1),...
                                                    '.txt'], ...
                                                      width, height);
        
    % Cost volume parameters
    CostVolumeParams = struct('min_disp', uint8(5), ...
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

    PrimalDualParams = struct('num_itr', uint32(100), ...
                              'alpha', single(0.5), ...
                              'beta', single(1.0), ...
                              'epsilon', single(0.1), ...
                              'lambda', single(0.5), ...
                              'aux_theta', single(10), ...
                              'aux_theta_gamma', single(1e-6));
    tic 
    [d, primal, dual, primal_step, dual_step, errors_precond, cost] =  HuberL1CVPrecond_mex(left_img, right_img, CostVolumeParams, PrimalDualParams);
    t = toc;
    
    % Fetch output from GPU
    opt_disp = gather(primal);   
    opt_disp = (opt_disp-min(min(opt_disp)))/(max(max(opt_disp)) - min(min(opt_disp)));
    
    num_colors = 65536;
    cmap = jet(num_colors);
    cmap_index = 1 + round(opt_disp * (num_colors - 1));
    image_rgb = reshape(cmap(cmap_index,:),size(opt_disp,1),size(opt_disp,2),3);
    imshow(image_rgb); 
    
    err = gather(errors_precond);
    
    % Save the cost volume
    cost_volume = gather(cost);
    save('cost_volume.mat', 'cost_volume');
    
        
    figure;
    imshow(opt_disp);
        
    figure;
    plot(err, 'g');    
    grid on;
    legend('HuberL1+Cost-Volume');
    xlabel('Iterations');
    ylabel('Energy function');
    
    s = sprintf('Elapsed time %fs', t);
    disp(s)

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
        L = 16;
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
        imshow(image_rgb);       
        
        
    else
        disp('Cost volume does not exist');
    end
end
    

%% Release GPU memory
    reset(gpu);
    
else
    disp('Error: No GPU was found.')
end