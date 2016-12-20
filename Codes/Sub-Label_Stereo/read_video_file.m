function [frames] = read_video_file(filename)

    saved_name = [filename(1:end-4), '_saved_frame', '.mat'];
    if ~exist(saved_name)
        vReader = VideoReader(filename);
        frames  = read(vReader);
        save(saved_name, 'frames', '-v7.3' );    
    else
        s = sprintf('Loading files: %s', saved_name);
        disp(s);
        load(saved_name, 'frames');     
    end

end