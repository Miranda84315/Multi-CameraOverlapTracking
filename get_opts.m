function opts = get_opts(player, track)

addpath(genpath('src'))
opts = [];
opts.player =player;
opts.track = track;

endFrame   = frame_info(opts.player, opts.track);

opts.dataset_path    = 'D:/Code/MultiCamOverlap/dataset/';
opts.gurobi_path     = 'C:/gurobi800/win64/matlab';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_alpha';
opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_openpose';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_reconnect';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cluster1';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cluster2';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cluster3';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam2';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam124';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam134';

%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam234';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam123';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam12';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam13';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam23';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam24';
%opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments_cam34';

%opts.experiment_dir = 'experiments';
if opts.player <10
    opts.experiment_name = ['Player0', num2str(opts.player), '/track', num2str(opts.track)];
else
    opts.experiment_name = ['Player', num2str(opts.player), '/track', num2str(opts.track)];
end
opts.sequence = 1;

opts.reader = DataVideoReader(opts.dataset_path, opts.experiment_name);
%opts.reader = DataVideoReader(opts.dataset_path);

% General settings
opts.eval_dir = 'L2-trajectories';
opts.visualize = false;
opts.image_width = 360;%1920;
opts.image_height = 288;%1080;
opts.current_camera = -1;
opts.world = 0;
opts.ROIs = getROIs();
opts.minimum_trajectory_length = 10;
opts.optimization = 'KL'; 
opts.use_groupping = 1;
opts.num_cam = 4;
opts.sequence = 1;
opts.sequence_names = {'trainval'};
opts.sequence_intervals = {1:endFrame};
opts.start_frames = [1, 1, 1, 1];
opts.render_threshold = 0.05;
opts.load_tracklets = 1;
opts.load_trajectories = 1;

% Tracklets
tracklets = [];
tracklets.window_width = 5;
tracklets.min_length = 1;
tracklets.alpha = 1;
tracklets.beta = 0.02;
tracklets.cluster_coeff = 0.75;
tracklets.nearest_neighbors = 8;
tracklets.speed_limit = 80000;
tracklets.threshold = 14;

% Trajectories
trajectories = [];
trajectories.appearance_groups = 1; % determined automatically when zero
trajectories.alpha = 1;
trajectories.beta = 0.01;
trajectories.window_width = 10;
trajectories.overlap = 5;
trajectories.speed_limit = 80000;
trajectories.indifference_time = 100;
trajectories.threshold = 14;
trajectories.finalnfirst=200;

% Identities
identities = [];
identities.window_width = 5000;
identities.appearance_groups = 0; % determined automatically when zero
identities.alpha = 1;
identities.beta = 0.01;
identities.overlap = 150;
identities.speed_limit = 30;
identities.indifference_time = 150;
identities.threshold = 8;
identities.extract_images = true;

% CNN model
net = [];
net.train_set = 'data/duke_train.csv';
net.image_root = 'D:/Code/AF_tracking/DukeMTMC-reID';
net.model_name = 'resnet_v1_50';
net.initial_checkpoint = 'resnet_v1_50.ckpt';
net.experiment_root = 'D:/Code/TrackingSystem/experiments/demo/demo_weighted_triplet';
net.embedding_dim = 128;
net.batch_p = 18;
net.batch_k = 4;
net.pre_crop_height = 288;
net.pre_crop_width = 144;
net.input_width = 128;
net.input_height = 256;
net.margin = 'soft';
net.metric = 'euclidean';
net.loss = 'weighted_triplet';
net.learning_rate = 0.0003;
net.train_iterations = 25000;
net.decay_start_iteration = 15000;
net.gpu_device = 0;
net.augment = true;
net.resume = false;
net.checkpoint_frequency = 1000;
net.hard_pool_size = 0;

opts.tracklets = tracklets;
opts.trajectories = trajectories;
opts.identities = identities;
opts.net = net;


