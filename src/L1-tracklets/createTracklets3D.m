function  tracklets = createTracklets3D(opts, originalDetections, allFeatures, startFrame, endFrame, tracklets, data)
% CREATETRACKLETS This function creates short tracks composed of several detections.
%   In the first stage our method groups detections into space-time groups.
%   In the second stage a Binary Integer Program is solved for every space-time
%   group.
%{
originalDetections = filteredDetections;
allFeatures = filteredFeatures;
startFrame = window_start_frame;
endFrame = window_end_frame;
tracklets = tracklets;
%}
%% DIVIDE DETECTIONS IN SPATIAL GROUPS
% Initialize variables
params          = opts.tracklets;
totalLabels     = 0; currentInterval = 0;

% Find detections for the current frame interval
currentDetectionsIDX    = intervalSearch(originalDetections(:,1), startFrame, endFrame);

% Skip if no more than 1 detection are present in the scene
if length(currentDetectionsIDX) < 2, return; end


% Compute bounding box centeres
detectionCenters        = originalDetections(currentDetectionsIDX, 2:3); 
detectionFrames         = originalDetections(currentDetectionsIDX, 1);

% Estimate velocities
estimatedVelocity       = estimateVelocities3D(originalDetections, startFrame, endFrame, params.nearest_neighbors, params.speed_limit);

% Spatial groupping
% -- first use spatial to separate some group and then line 65 use KL algo
% -- to spearate more detailed 
spatialGroupIDs         = getSpatialGroupIDs(opts.use_groupping, currentDetectionsIDX, detectionCenters, params);

% Show window detections
% show_frame = global2local(opts.start_frames(opts.current_camera), startFrame)
if opts.visualize, trackletsVisualizePart1; end
%% SOLVE A GRAPH PARTITIONING PROBLEM FOR EACH SPATIAL GROUP
fprintf('Creating tracklets: solving space-time groups ');
for spatialGroupID = 1 : max(spatialGroupIDs)
    
    elements = find(spatialGroupIDs == spatialGroupID);
    spatialGroupObservations        = currentDetectionsIDX(elements);
    
    % Create an appearance affinity matrix and a motion affinity matrix
    appearanceCorrelation           = getAppearanceSubMatrix3D(opts.num_cam, spatialGroupObservations, allFeatures, params.threshold);
    spatialGroupDetectionCenters    = detectionCenters(elements,:);
    spatialGroupDetectionFrames     = detectionFrames(elements,:);
    spatialGroupEstimatedVelocity   = estimatedVelocity(elements,:);
    % -- impMatrix is record the impossible matrix(velocity>speed.limit)
    %[motionCorrelation, impMatrix]  = motionAffinity(spatialGroupDetectionCenters,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity,params.speed_limit, params.beta);
    [motionCorrelation, impMatrix]  = motionAffinity_new(spatialGroupDetectionCenters, spatialGroupDetectionFrames, params.speed_limit);
    
    
    % Combine affinities into correlations
    % -- discount is the paper Optimization part e^(-bt)
    % -- decay correlations to zero as the time lag t between observation increases.
    intervalDistance                = pdist2(spatialGroupDetectionFrames,spatialGroupDetectionFrames);
    discountMatrix                  = min(1, -log(intervalDistance/params.window_width));
%     correlationMatrix               = motionCorrelation + appearanceCorrelation; 
%     correlationMatrix               = correlationMatrix .* discountMatrix;
    %correlationMatrix               = motionCorrelation .* discountMatrix + appearanceCorrelation -1; 
    correlationMatrix               = motionCorrelation + appearanceCorrelation -1; 
    correlationMatrix(impMatrix==1) = -inf;
    
    % Show spatial grouping and correlations
    %if opts.visualize, trackletsVisualizePart2; end
    
    % Solve the graph partitioning problem
    fprintf('%d ',spatialGroupID);
    
    if strcmp(opts.optimization,'AL-ICM')
        labels = AL_ICM(sparse(correlationMatrix));
    elseif strcmp(opts.optimization,'KL')
        labels = KernighanLin(correlationMatrix);
    elseif strcmp(opts.optimization,'BIPCC')
        initialSolution = KernighanLin(correlationMatrix);
        labels = BIPCC(correlationMatrix, initialSolution);
    end
    
    labels      = labels + totalLabels;
    totalLabels = max(labels);
    identities  = labels;
    originalDetections(spatialGroupObservations, 8) = identities;
    
    % Show clustered detections
    if opts.visualize, trackletsVisualizePart3; end
end
fprintf('\n');

%% FINALIZE TRACKLETS
% Fit a low degree polynomial to include missing detections and smooth the tracklet
trackletsToSmooth  = originalDetections(currentDetectionsIDX,:);
featuresAppearance = allFeatures.appearance(currentDetectionsIDX,:);
smoothedTracklets  = smoothTracklets3D(opts.num_cam, trackletsToSmooth, startFrame, params.window_width, featuresAppearance, params.min_length, currentInterval, data);

% Assign IDs to all tracklets
for i = 1:length(smoothedTracklets)
    smoothedTracklets(i).id = i;
    smoothedTracklets(i).ids = i;
end

% Attach new tracklets to the ones already discovered from this batch of detections
if ~isempty(smoothedTracklets)
    ids = 1 : length(smoothedTracklets); 
    tracklets = [tracklets, smoothedTracklets];
end

% Show generated tracklets in window
%if opts.visualize, trackletsVisualizePart4; end

if ~isempty(tracklets)
    tracklets = nestedSortStruct(tracklets,{'startFrame','endFrame'});
end


end