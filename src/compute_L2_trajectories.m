function compute_L2_trajectories(opts)
% Computes single-camera trajectories from tracklets
for iCam = 1:4

    % Initialize
    % -- load tracklets from L1-tracklet.mat
    % -- trajectoriesFromTracklets include detection start/endFrame and
    % -- segmentStart/End
    %iCam =2;
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    trajectoriesFromTracklets = trackletsToTrajectories(tracklets,1:length(tracklets));
    
    opts.current_camera = iCam;
    sequence_interval = opts.sequence_intervals{opts.sequence};
    % -- use 1 second long windows ,overlap 50 % to product 10second long
    % -- single camera trajecories
    startFrame = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) - opts.trajectories.window_width);
    endFrame   = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) + opts.trajectories.window_width);

    trajectories = trajectoriesFromTracklets; 
    
    while startFrame <= global2local(opts.start_frames(opts.current_camera), sequence_interval(end))
        % Display loop state

        clc; fprintf('Cam: %d - Window %d...%d\n', iCam, startFrame, endFrame);

        % Compute trajectories in current time window
        trajectories = createTrajectories( opts, trajectories, startFrame, endFrame);

        % Update loop range
        startFrame = endFrame   - opts.trajectories.overlap;
        endFrame   = startFrame + opts.trajectories.window_width;
    end

    % Convert trajectories 
    % -- get all data from trajectories include id frame xy point
    trackerOutputRaw = trajectoriesToTop(trajectories);
    % Interpolate missing detections
    trackerOutputFilled = fillTrajectories(trackerOutputRaw);
    % Remove spurius tracks
    % -- I think the opt.minimum_trajectory_length should be more large
    trackerOutputRemoved = removeShortTracks(trackerOutputFilled, opts.minimum_trajectory_length);
    % Make identities 1-indexed
    % --rename the id, because we remove the short tracklet
    [~, ~, ic] = unique(trackerOutputRemoved(:,2));
    trackerOutputRemoved(:,2) = ic;
    trackerOutput = sortrows(trackerOutputRemoved,[2 1]);

    %% Save trajectories
    fprintf('Saving results\n');
    fileOutput = trackerOutput(:, [1:6]);
    filename_save = sprintf('%s/%s/L2-trajectories/L2_cam%d.mat',opts.experiment_root, opts.experiment_name, iCam);
    save(filename_save,'fileOutput');

    dlmwrite(sprintf('%s/%s/L2-trajectories/cam%d_%s.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        fileOutput, 'delimiter', ' ', 'precision', 6);


end
end