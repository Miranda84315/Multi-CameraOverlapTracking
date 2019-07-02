% Use for run feature extraction


for player=1:1
    for track=1:2
        opts = get_opts(player, track);
        %create_experiment_dir(opts);
        %compute_L0_features(opts);
        compute_L1_tracklets3D(opts);
        compute_L2_trajectories3D(opts);
    end
end