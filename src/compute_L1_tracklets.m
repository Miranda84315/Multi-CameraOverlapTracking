function compute_L1_tracklets(opts)
% Computes tracklets for all cameras

for iCam = 1:4
    
    opts.current_camera = iCam;
    
    sequence_window   = opts.sequence_intervals{opts.sequence};
    % -- because my detection had already global to local, so we don't need
    % global2local
    start_frame       = global2local(opts.start_frames(opts.current_camera), sequence_window(1));
    end_frame         = global2local(opts.start_frames(opts.current_camera), sequence_window(end));
    % --start_frame = sequence_window(1);
    % --end_frame = sequence_window(end);
    
    % Load OpenPose detections for current camera
    % -- line 15 will load variable [detections], contains [n x 56] array by
    % -- openpose detector
    load(fullfile(opts.dataset_path, 'detections/No3', sprintf('cam%d.mat',iCam)));
    detections = double(detections);
    % Load features for all detections
    % -- line 20 will load from ...src\triplet-reid\experiments\demo\L0-features
    % -- and is a openpose's detector pre-computed features [128 x n] array
    filename = sprintf('%s/%s/L0-features/features%d.mat',opts.experiment_root,opts.experiment_name,iCam)
    features_temp   = load(filename);
    features = features_temp.features;
    % features   = h5read(sprintf('%s/%s/L0-features/features%d.mat',opts.experiment_root,opts.experiment_name,iCam),'/emb');
   
    features   = double(features');
    all_dets   = detections;
    appearance = cell(size(all_dets,1),1);
    frames     = cell(size(all_dets,1),1);
    % -- line 28 will let feature [n*128]_double array -> become [n*1]_cell
    % -- line 29 only use the detections' frame number
    for k = 1:length(frames)
        appearance{k} = features(k,:);
        frames{k} = detections(k,2);
    end
    
    % Compute tracklets for every 1-second interval
    tracklets = struct([]);
    
    % -- each tracklets.windows_width(50 frame) to compute
    for window_start_frame   = start_frame : opts.tracklets.window_width : end_frame
        fprintf('%d/%d\n', window_start_frame, end_frame);
        %window_start_frame = window_start_frame + opts.tracklets.window_width ;
        
        % Retrieve detections in current window
        window_end_frame     = window_start_frame + opts.tracklets.window_width - 1;
        window_frames        = window_start_frame : window_end_frame;
        % -- line 44 find all_dets's frame is the same in window_frame and return
        % -- there index 
        window_inds          = find(ismember(all_dets(:,2),window_frames));
        detections_in_window = all_dets(window_inds,:);
        % -- openpose is [camera, frame, kpx1, kpy1, kpc1, ..., kpx18, kpy18, kpc18]
        % -- (5:3:end) can find kpc1,kpc2,...kpc18 kpc is confidence
        % -- num_visible is count the number large than threshold
        %detections_conf      = sum(detections_in_window(:,5:3:end),2);
        %num_visible          = sum(detections_in_window(:,5:3:end)> opts.render_threshold, 2);
        detections_conf = ones(size(detections_in_window,1),1);
        num_visible  = ones(size(detections_in_window,1),1);
        
        % Use only valid detections
        % -- getValidDetections calucate boundingbox and detections_in_window is repleace
        % -- detections_in_window[:,3:end] to [:,3:6(boundingbox)]
        [valid, detections_in_window] = getValidDetections(detections_in_window, detections_conf, num_visible, opts, iCam);
        detections_in_window          = detections_in_window(valid,:);
        % detections_in_window(:,7:end) = [];
        detections_in_window(:,[1 2]) = detections_in_window(:,[2 1]);
        filteredDetections = detections_in_window;
        filteredFeatures = [];
        filteredFeatures.appearance = appearance(window_inds(valid));
        
        % Compute tracklets in current window
        % Then add them to the list of all tracklets
        tracklets = createTracklets(opts, filteredDetections, filteredFeatures, window_start_frame, window_end_frame, tracklets);
    end
    
    % Save tracklets
    % -- tracklets info from 'smoothTracklet.m'
    save(sprintf('%s/%s/L1-tracklets/tracklets%d_%s.mat', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        'tracklets');
    
    % Clean up
    clear all_dets appearance detections features frames
    
end
