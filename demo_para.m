% Use for run feature extraction

player_num = [22, 29, 31, 32, 34, 35];
for player=1:6
    for track=1:8
        opts = get_opts(player_num(player), track);
        create_experiment_dir(opts);
        compute_L0_features(opts);
        compute_L1_tracklets3D(opts);
        compute_L2_trajectories3D(opts);
        %pause;
    end
end