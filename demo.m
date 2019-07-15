%% Options
opts = get_opts();
create_experiment_dir(opts);

%% Run Tracker
player = 25;
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

fix_gt;
my_evaluate(opts);
[TP_temp, FN_temp, FP_temp, IDSW_temp, MOTA_temp] = gt_demo(opts);


command = strcat('C:/Users/Owner/Anaconda3/envs/tensorflow/python.exe UI/show_3D.py' , ...
    sprintf(' --track %s', num2str(opts.track)), ...
    sprintf(' --player %s', num2str(opts.player)));
system(command);
