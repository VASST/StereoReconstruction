function [cost_volume] = compute_stereo_ZNCC_cost(path_im0, path_im1, ndisps, sz, dscl1, dscl2)
% computes a stereo matching cost between two images given by
% path_im0 and path_im1. ndisps denotes the number of disparities,
% sz, patch size
% dscl1 is a factor by which to downscale the original image, and
% dscl2 is a factor by which to downscale the dataterm. dscl2 is
% essentially a "patch-size" for the dataterm computation.
    
    % read stereo images and convert to gray
    im_0 = rgb2gray(imresize(double(imread(path_im0)) / 255, (1/dscl1)));
    im_1 = rgb2gray(imresize(double(imread(path_im1)) / 255, (1/dscl1)));
    [ny, nx, nc] = size(im_0);

    dt_alpha = 0.1;
    dt_beta = 0.1;

    % compute full dataterm
    f = zeros(ny, nx, ndisps);
    for d=1:ndisps
        
        patch_L = im_0(x, y, :)
       
        % compute ZNCC
        f(:, :, d) = min(dt_alpha, sum_dx) + min(dt_beta, sum_dy);
    end

    % downscale dataterm
    ny2 = floor(ny / dscl2);
    nx2 = floor(nx / dscl2);
    cost_volume = zeros(ny2, nx2, ndisps);

    for i=1:dscl2
        for j=1:dscl2
            cost_volume = cost_volume + f(i:dscl2:end, j:dscl2:end, :);
        end
    end
    cost_volume = cost_volume / (dscl2 * dscl2);

end
