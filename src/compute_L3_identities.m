function compute_L3_identities(opts)
% Computes multi-camera trajectories from single-camera trajectories

trajectories = loadL2trajectories(opts);
trajectories = getTrajectoryFeatures(opts, trajectories);
filename = sprintf('%s/%s/L3-identities/L2trajectories_%s.mat',opts.experiment_root, opts.experiment_name, opts.sequence_names{opts.sequence});
save(filename,'trajectories');
%  load(filename);
identities = trajectories;

for k = 1:length(identities)
    % -- get the global frame
    % -- get the start/end Frame
    identities(k).trajectories(1).data(:,end+1) = local2global(opts.start_frames(identities(k).trajectories(1).camera) ,identities(k).trajectories(1).data(:,1));
    identities(k).trajectories(1).startFrame = identities(k).trajectories(1).data(1,9);
    identities(k).startFrame = identities(k).trajectories(1).startFrame;
    identities(k).trajectories(1).endFrame = identities(k).trajectories(1).data(end,9);
    identities(k).endFrame   = identities(k).trajectories(1).endFrame;
end
identities = sortStruct(identities,'startFrame');

global_interval = opts.sequence_intervals{opts.sequence};
startFrame = global_interval(1);
endFrame = global_interval(1) + opts.identities.window_width - 1;

while startFrame <= global_interval(end)
    fprintf('Window %d...%d\n', startFrame, endFrame);
    
    identities = linkIdentities(opts, identities, startFrame, endFrame);
    
    % advance sliding temporal window
    startFrame = endFrame   - opts.identities.window_width/2;
    endFrame   = startFrame + opts.identities.window_width;
end
%%
fprintf('Saving results\n');
trackerOutputL3 = identities2mat(identities);
for iCam = 1:opts.num_cam
    cam_data = trackerOutputL3(trackerOutputL3(:,1) == iCam,2:end);
    dlmwrite(sprintf('%s/%s/L3-identities/cam%d_%s.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        cam_data, 'delimiter', ' ', 'precision', 6);
end

% -- to save trackerOutputL3
filename_save = sprintf('%s/%s/L3-identities/L3Final_trajectories.mat',opts.experiment_root, opts.experiment_name);
save(filename_save,'trackerOutputL3');
load(filename_save);