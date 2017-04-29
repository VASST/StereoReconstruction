function [disparity, disparity_color, err] = StereoReconstHuberL1(left_img, right_img,  CostVolumeParams)

PrimalDualParams = struct('num_itr', uint32(150), ...
                              'alpha', single(10.0), ...
                              'beta', single(1.0), ...
                              'epsilon', single(0.1), ...
                              'lambda', single(1e-3), ...
                              'aux_theta', single(10), ...
                              'aux_theta_gamma', single(1e-6));
    tic 
    [d, primal, dual, primal_step, dual_step, errors_precond, aux, cost] =  HuberL1CVPrecond_mex(left_img, right_img, CostVolumeParams, PrimalDualParams);
    t = toc;
    
    err = gather(errors_precond);
    
    % Save the cost volume
    cost_volume = gather(cost);
    save('cost_volume.mat', 'cost_volume');
    
    % Fetch output from GPU
    opt_disp = gather(primal);   
    opt_disp = (opt_disp-min(min(opt_disp)))/(max(max(opt_disp)) - min(min(opt_disp)));
    diff_disp = repmat((CostVolumeParams.max_disp - CostVolumeParams.min_disp), size(opt_disp,1), size(opt_disp,2));
    min_disp  = repmat(CostVolumeParams.min_disp, size(opt_disp,1), size(opt_disp,2));
    disparity = opt_disp.*single(diff_disp) + single(min_disp);
       
    % Plot
    num_colors = 65536;
    cmap = jet(num_colors);
    [nx, ny] = size(left_img);
    cmap_index = 1 + round(reshape(opt_disp, 1, nx*ny)* (num_colors - 1));
    disparity_color = reshape(cmap(cmap_index,:),size(opt_disp,1),size(opt_disp,2),3);    
    
    s = sprintf('Elapsed time %fs', t);
    disp(s)
    
end