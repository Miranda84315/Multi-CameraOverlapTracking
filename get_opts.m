function opts = get_opts()

addpath(genpath('src'))

opts = [];
opts.dataset_path    = 'D:/Code/MultiCamOverlap/dataset/';
opts.gurobi_path     = 'C:/gurobi800/win64/matlab';
opts.experiment_root = 'D:/Code/MultiCamOverlap/experiments';
opts.experiment_name = 'demo';

opts.reader = DataVideoReader(opts.dataset_path);

% General settings
opts.eval_dir = 'L3-identities';
opts.visualize = false;
opts.image_width = 1920;
opts.image_height = 1080;
opts.current_camera = -1;
opts.world = 0;
opts.ROIs = getROIs();
opts.minimum_trajectory_length = 10;
opts.optimization = 'KL'; 
opts.use_groupping = 1;
opts.num_cam = 4;
opts.sequence = 1;
opts.sequence_names = {'trainval'};
opts.sequence_intervals = {1:225};
opts.start_frames = [1, 1, 1, 1];
opts.render_threshold = 0.05;
opts.load_tracklets = 1;
opts.load_trajectories = 1;

% Tracklets
tracklets = [];
tracklets.window_width = 15;
tracklets.min_length = 5;
tracklets.alpha = 1;
tracklets.beta = 0.02;
tracklets.cluster_coeff = 0.75;
tracklets.nearest_neighbors = 8;
tracklets.speed_limit = 20;
tracklets.threshold = 8;

% Trajectories
trajectories = [];
trajectories.appearance_groups = 0; % determined automatically when zero
trajectories.alpha = 1;
trajectories.beta = 0.01;
trajectories.window_width = 70;
trajectories.overlap = 35;
trajectories.speed_limit = 200;
trajectories.indifference_time = 100;
trajectories.threshold = 8;

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

