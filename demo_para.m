% Use for run feature extraction


for player=21:35
    for track=1:8
        opts = get_opts(player, track);
        compute_L1_tracklets3D(opts);
        compute_L2_trajectories3D(opts);
    end
end