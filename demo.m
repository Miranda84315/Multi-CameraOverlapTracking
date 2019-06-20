%% Options
opts = get_opts();
create_experiment_dir(opts);

%% Run Tracker
player = 15;
track= 6;
opts = get_opts(player, track);
% opts.visualize = true;

% compute feature
compute_L0_features(opts);

% Tracklets
%compute_L1_tracklets(opts);
compute_L1_tracklets3D(opts);

% Single-camera trajectories
%compute_L2_trajectories(opts);
compute_L2_trajectories3D(opts);
opts.eval_dir = 'L2-trajectories';
evaluate(opts);
