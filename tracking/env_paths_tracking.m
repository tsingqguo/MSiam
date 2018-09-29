function p = env_paths_tracking(p)

    p.net_base_path = '/home/fan/Desktop/Object_Tracking/tracker_benchmark_v1.0/trackers/MSiam/model/';%';'./model/';
    p.seq_base_path = '/home/fan/Desktop/Object_Tracking/Dataset/OTB/';
    p.seq_vot_base_path = '/path/to/VOT/evaluation/sequences/'; % (optional)
    p.stats_path = '/path/to/ILSVRC15-VID/stats.mat'; % (optional)

end
