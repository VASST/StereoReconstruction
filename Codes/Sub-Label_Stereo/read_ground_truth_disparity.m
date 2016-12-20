function [out] = read_ground_truth_disparity(filename, width, height)

    saved_name = [filename(1:end-4), '_saved_frame_disparity', '.mat'];

    if ~exist(saved_name)
        fid = fopen(filename, 'r');
        C = textscan(fid, '%f');
        c = C{1};        
        out = reshape(c, width, height)';
    else

    end

end