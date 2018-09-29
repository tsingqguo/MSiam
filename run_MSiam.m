function [ results ] = run_MSiam(seq, res_path, bSaveImage,varargin)
    addpath(genpath('./tracking'));
    addpath(genpath('./utils'));
    startup;
    
    %% Parameters that should have no effect on the result.
    params.video = seq.name(1:end-2);
    params.framepaths = seq.s_frames;
    params.init_rect = seq.init_rect;
    params.visualization = bSaveImage;
    params.gpus = 1;
    
    %% Parameters that should be recorded.
%     params = vl_argparse(params, varargin);
    
    [rects,fps] = tracker(params); 
    results.type   = 'rect';
    results.res    = rects;
    results.fps    = fps;
    
end