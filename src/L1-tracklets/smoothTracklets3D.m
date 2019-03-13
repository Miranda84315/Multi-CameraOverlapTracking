function smoothedTracklets = smoothTracklets3D( num_cam, tracklets, segmentStart, segmentInterval, featuresAppearance, minTrackletLength, currentInterval, data )
% This function smooths given tracklets by fitting a low degree polynomial 
% in their spatial location
%{
num_cam = opts.num_cam;
tracklets=trackletsToSmooth;
segmentStart=startFrame;
segmentInterval=params.window_width;
featuresAppearance=featuresAppearance;
minTrackletLength=params.min_length;
currentInterval=currentInterval;
%}

trackletIDs          = unique(tracklets(:, 8));
numTracklets         = length(trackletIDs);
smoothedTracklets    = struct([]);

for i = 1:numTracklets

    mask = tracklets(:, 8)==trackletIDs(i);
    detections = tracklets(mask,:);
    
    % Reject tracklets of short length
    start = min(detections(:,1));
    finish = max(detections(:,1));
    
    if (size(detections,1) < minTrackletLength) || (finish - start < minTrackletLength)
        continue;
    end

    intervalLength = finish-start + 1;
    
    datapoints = linspace(start, finish, intervalLength);
    frames     = detections(:,1);
    
    % currentTracklet = [frame, label, x_3d, y_3d, cam1_left, cam1_top,  cam1_right, cam1_bottom... , cam4_bottom]
    currentTracklet      = zeros(intervalLength, 20);
    currentTracklet(:,2) = ones(intervalLength,1) .* trackletIDs(i);
    currentTracklet(:,1) = [start : finish];
    currentTracklet(:, 5:end) = -1;
    
    % fill data with original detection
    for c = 1:num_cam
        index_in_detection = detections(:, c + 3);
        for k=1:length(index_in_detection)
            % if == -1: no detection in this camera
            if index_in_detection(k) ~= -1
                real_k = frames(k);
                ind = (currentTracklet(:, 1) == real_k);
                currentTracklet(ind, 4*c+1:4*c+4) = data{c, 2}(index_in_detection(k), 3:6);
            end
        end
    end
    
    % Fit xworld, yworld
    for k = 3:4
        points    = detections(:,k-1);
        p         = polyfit(frames,points,1);
        newpoints = polyval(p, datapoints);
        currentTracklet(:,k) = newpoints';
    end
    
    % Fit left, top, right, bottom
    %{
   [!!!] if use this, result is not good!
    for k = 5:20
        real_ind = frames - start + 1; 
        points    = currentTracklet(real_ind, k);
        p         = polyfit(frames, points, 1);
        newpoints = polyval(p, datapoints);
        currentTracklet(:,k) = newpoints';
    end
    %}
    
    % Compute appearance features
    medianFeature = cell(1, num_cam);
    for c=1:num_cam
        medianFeature{1, c} = median(cell2mat(featuresAppearance(mask, c)));
        if (size(cell2mat(featuresAppearance(mask, c)), 1) == 1)
            medianFeature{1, c} = cell2mat(featuresAppearance(mask, c));
        end
        if isnan(medianFeature{1, c})
            medianFeature{1, c} = [];
        end
    end
    
    centers          = currentTracklet(:, [3:4]);
    centerPoint      = median(centers); % assumes more then one detection per tracklet
    centerPointWorld = 1;% median(currentTracklet(:,[7,8]));
    
    % Add to tracklet list
    smoothedTracklets(end+1).feature       = medianFeature; 
    smoothedTracklets(end).center          = centerPoint;
    smoothedTracklets(end).centerWorld     = centerPointWorld;
    smoothedTracklets(end).data            = currentTracklet;
    smoothedTracklets(end).features        = featuresAppearance(mask, :);
    smoothedTracklets(end).realdata        = detections;
    smoothedTracklets(end).mask            = mask;
    smoothedTracklets(end).startFrame      = start;
    smoothedTracklets(end).endFrame        = finish;
    smoothedTracklets(end).interval        = currentInterval;
    smoothedTracklets(end).segmentStart    = segmentStart;
    smoothedTracklets(end).segmentInterval = segmentInterval;
    smoothedTracklets(end).segmentEnd      = segmentStart + segmentInterval - 1;
    
    assert(~isempty(currentTracklet));
end



