function compute_L2_trajectories3D(opts)
% Computes single-camera trajectories from tracklets
% Initialize
% -- load tracklets from L1-tracklet.mat
% -- trajectoriesFromTracklets include detection start/endFrame and
% -- segmentStart/End
load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets_%s.mat',opts.sequence_names{opts.sequence})));
trajectoriesFromTracklets = trackletsToTrajectories(tracklets,1:length(tracklets));

opts.current_camera = 1;
sequence_interval = opts.sequence_intervals{opts.sequence};
% -- use 1 second long windows ,overlap 50 % to product 10second long
% -- single camera trajecories
startFrame = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) - opts.trajectories.window_width);
endFrame   = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) + opts.trajectories.window_width);

trajectories = trajectoriesFromTracklets; 

while startFrame <= global2local(opts.start_frames(opts.current_camera), sequence_interval(end))
    % Display loop state
    clc; fprintf('Window %d...%d\n', startFrame, endFrame);
    % Compute trajectories in current time window
    trajectories = createTrajectories3D_new( opts, trajectories, startFrame, endFrame);
    % Update loop range
    startFrame = endFrame   - opts.trajectories.overlap;
    endFrame   = startFrame + opts.trajectories.window_width;
end

% Convert trajectories 
for iCam = 1:opts.num_cam
    filename = sprintf('%s/%s/L0-features/features%d.mat',opts.experiment_root,opts.experiment_name,iCam)
    features_temp   = load(filename);
    detections_temp = load(fullfile(opts.dataset_path, 'alpha_pose', opts.experiment_name, sprintf('cam%d.mat',iCam)));
    data{iCam,1} = double(features_temp.features');
    data{iCam,2} = detections_temp.detections;
end
clear detections_temp features_temp iCam trajectoriesFromTracklets startFrame endFrame

% -- delete short trajectories
remove_trajectories = removeShortTrajectories(trajectories, opts.minimum_trajectory_length);
reconnect_trajectories = reconnectTrajectories_new(opts, remove_trajectories);

% old
%reconnect_trajectories = reconnectTrajectories(trajectories);
%new_trajectories = removeShortTrajectories(reconnect_trajectories, opts.minimum_trajectory_length);
%new_trajectories = reconnectTrajectories_segment(new_trajectories);

% -- get all data from trajectories include id frame xy point
trackerOutputRaw = trajectoriesToTop(reconnect_trajectories);
% Interpolate missing detections
trackerOutputFilled = fillTrajectories(trackerOutputRaw);
% Remove spurius tracks
trackerOutputRemoved = trackerOutputFilled; %fillandSmoothTrajectories(trackerOutputFilled, opts.num_cam);
% Make identities 1-indexed
% --rename the id, because we remove the short tracklet
[~, ~, ic] = unique(trackerOutputRemoved(:,2));
trackerOutputRemoved(:,2) = ic;
trackerOutput = sortrows(trackerOutputRemoved,[2 1]);

%% Save trajectories
fprintf('Saving results\n');
fileOutput = trackerOutput(:, [1:20]);
filename_save = sprintf('%s/%s/L2-trajectories/L2_cam.mat',opts.experiment_root, opts.experiment_name);
save(filename_save,'fileOutput');

dlmwrite(sprintf('%s/%s/L2-trajectories/cam_%s.txt', ...
    opts.experiment_root, ...
    opts.experiment_name, ...
    opts.sequence_names{opts.sequence}), ...
    fileOutput, 'delimiter', ' ', 'precision', 6);

%% run result

command = strcat('C:/Users/Owner/Anaconda3/envs/tensorflow/python.exe UI/show3Dvideo.py' , ...
    sprintf(' --track %s', num2str(opts.track)), ...
    sprintf(' --player %s', num2str(opts.player)));
system(command);
end