% main script. 

addpath ./miccai2013/


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


if(gpu_enable)
    
    % Read-in input images
    left_img_file = './images/im-dL.png';
    right_img_file = './images/im-dR.png';
    
    left_img = im2single(rgb2gray(imread(left_img_file)));
    right_img = im2single(rgb2gray(imread(right_img_file)));                    
        
    % Cost volume parameters
    CostVolumeParams = struct('min_disp', uint8(0), ...
                             'max_disp', uint8(64), ...
                             'method', 'zncc', ...
                             'win_r', uint8(4), ...
                             'ref_left', true);
                         
%% Chang et. al, MICCAI 2013
    disp('Running Chang et. al., MICCAI 2013..')
    s = sprintf('Size of the cost volume %dx%dx%d', size(left_img, 1), ...
                                                        size(left_img, 2), ...
                                                          CostVolumeParams.max_disp - CostVolumeParams.min_disp);
    disp(s);

    PrimalDualParams = struct('num_itr', uint32(100), ...
                              'alpha', single(10.0), ...
                              'beta', single(1.0), ...
                              'epsilon', single(0.1), ...
                              'lambda', single(1e-3), ...
                              'aux_theta', single(10), ...
                              'aux_theta_gamma', single(1e-6));
    tic 
    [d, primal, dual, primal_step, dual_step, errors_precond, vol] =  HuberL1CVPrecond_mex(left_img, right_img, CostVolumeParams, PrimalDualParams);
    t = toc;
    
    % Fetch output from GPU
    opt_disp = gather(primal);   
    opt_disp = (opt_disp-min(min(opt_disp)))/(max(max(opt_disp)) - min(min(opt_disp)));
    
    err = gather(errors_precond);
    
        
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

    
%% Functional Lifting

    %TODO

    

%% Release GPU memory
    reset(gpu);
    rmpath('gpu')
    
else
    disp('Error: No GPU was found.')
end