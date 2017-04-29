function [disparity, disparity_color] = StereoReconst_Sublabel_Lifting(cost_volume,...
                                                                         CostVolumeParams)
                %% setup parameters
L = 8;
gamma = linspace(0, 1, L)'; 
lmb = 0.9;
                
tic;
[u_unlifted] = sublabel_lifting_convex(double(cost_volume), gamma, lmb);
toc;

[ny, nx, ~] = size(cost_volume);  
opt_disp = reshape(min(max(u_unlifted, 0), 1), ny, nx);

opt_disp = (opt_disp-min(min(opt_disp)))/(max(max(opt_disp)) - min(min(opt_disp)));
diff_disp = repmat((CostVolumeParams.max_disp - CostVolumeParams.min_disp), size(opt_disp,1), size(opt_disp,2));
min_disp  = repmat(CostVolumeParams.min_disp, size(opt_disp,1), size(opt_disp,2));
disparity = opt_disp.*single(diff_disp) + single(min_disp);

num_colors = 65536;
cmap = jet(num_colors);
cmap_index = 1 + round(opt_disp * (num_colors - 1));
disparity_color = reshape(cmap(cmap_index,:),ny,nx,3);

end
                                                                        