% main script. 

addpath ./miccai2013/


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
    
    left_img_file = './images/im-dL.png';
    right_img_file = './images/im-dR.png';
    
    left_img = im2single(rgb2gray(imread(left_img_file)));
    right_img = im2single(rgb2gray(imread(right_img_file)));                    
        
    CostVoumeParams = struct('min_disp', uint8(0), ...
                             'max_disp', uint8(64), ...
                             'method', 'zncc', ...
                             'win_r', uint8(4), ...
                             'ref_left', true);

    PrimalDualParams = struct('num_itr', uint32(100), ...
                              'alpha', single(10.0), ...
                              'beta', single(1.0), ...
                              'epsilon', single(0.1), ...
                              'lambda', single(1e-3), ...
                              'aux_theta', single(10), ...
                              'aux_theta_gamma', single(1e-6));
    [d, primal, dual, primal_step, dual_step, errors_precond] =  HuberL1CVPrecond_mex(left_img, right_img, CostVoumeParams, PrimalDualParams);
    
    opt_disp = gather(primal);   
    opt_disp = (opt_disp-min(min(opt_disp)))/(max(max(opt_disp)) - min(min(opt_disp)));
        
    figure;
    imshow(opt_disp);
        
    figure;
    plot(gather(errors_precond), 'g');    
    grid on;
    legend('HuberL1+Cost-Volume');
    xlabel('Iterations');
    ylabel('Energy function');
    
    % Release GPU memory
    reset(gpu);
    rmpath('gpu')
end