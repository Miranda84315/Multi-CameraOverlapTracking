function compute_L0_features(opts)
% Computes features for the input poses
pose = 'alpha_pose';
%pose = 'detections';
%pose = 'openpose';
for iCam = 1:4

    % Load poses
    % -- iI change the file path and delete the original part about openpose(1, 18) to detections(1, 6) --
    load(fullfile(opts.dataset_path, pose, opts.experiment_name, sprintf('cam%d.mat',iCam)));
    % Compute feature embeddings
    features = embed_detections(opts, detections);
    
    % Save features
    % h5write(sprintf('%s/%s/L0-features/features%d.h5',opts.experiment_root,opts.experiment_name,iCam),'/emb', features);
    filename_save = sprintf('%s/%s/L0-features/features%d.mat',opts.experiment_root,opts.experiment_name,iCam);
    save(filename_save,'features');
end
