%% Options
opts = get_opts();
create_experiment_dir(opts);

%% Run Tracker

% opts.visualize = true;
opts.sequence = 1; % trainval-mini

% compute feature
%compute_L0_features(opts);

% Tracklets
opts.optimization = 'KL';
compute_L1_tracklets(opts);
compute_L1_tracklets3D(opts);

% Single-camera trajectories
opts.optimization = 'KL';
opts.trajectories.appearance_groups = 1;
compute_L2_trajectories(opts);
compute_L2_trajectories3D(opts);

