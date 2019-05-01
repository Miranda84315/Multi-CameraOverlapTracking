function outputTrajectories = createTrajectories3D_new( opts, inputTrajectories, startTime, endTime)
% CREATETRAJECTORIES partitions a set of tracklets into trajectories.
%   The third stage uses appearance grouping to reduce problem complexity;
%   the fourth stage solves the graph partitioning problem for each
%   appearance group.
%{
inputTrajectories = trajectories;
startTime = startFrame;
endTime = endFrame;
%}

% find current, old, and future tracklets
% -- findTrajectoriesInWindow is to find the index in the time window
% -- and use currentTrajectoriesInd to select the index of trajectories which we need info 
currentTrajectoriesInd    = findTrajectoriesInWindow(inputTrajectories, startTime, endTime);
currentTrajectories       = inputTrajectories(currentTrajectoriesInd);

% safety check
if length(currentTrajectories) <= 1
    outputTrajectories = inputTrajectories;
    return;
end

% select tracklets that will be selected in association. For previously
% computed trajectories we select only the last three tracklets.
inAssociation = []; tracklets = []; trackletLabels = [];
for i = 1 : length(currentTrajectories)
   for k = 1 : length(currentTrajectories(i).tracklets) 
       tracklets        = [tracklets; currentTrajectories(i).tracklets(k)]; %#ok
       trackletLabels   = [trackletLabels; i]; %#ok

       inAssociation(length(trackletLabels)) = true; %#ok
       if k ~=length(currentTrajectories(i).tracklets) 
           inAssociation(length(trackletLabels)) = false; %#ok
       end
       % Use the last five values
   end
end
inAssociation = logical(inAssociation);

% show all tracklets
%if opts.visualize, trajectoriesVisualizePart1; end

% ----- my method
%distanceMatrix = getFirstandFinal(opts, tracklets(inAssociation), trackletLabels(inAssociation));


% solve the graph partitioning problem for each appearance group
result = solveInGroups3D_new(opts, tracklets(inAssociation), trackletLabels(inAssociation));

% merge back solution. Tracklets that were associated are now merged back
% with the rest of the tracklets that were sharing the same trajectory
labels = trackletLabels; labels(inAssociation) = result.labels;
count = 1;
for i = 1 : length(inAssociation)
   if inAssociation(i) > 0
      labels(trackletLabels == trackletLabels(i)) = result.labels(count);
      count = count + 1;
   end
end

% merge co-identified tracklets to extended tracklets
% -- each identity each struct
newTrajectories = trackletsToTrajectories(tracklets, labels);
%checkTrajectories = checkFromEdge3D(newTrajectories, startTime);
smoothTrajectories = recomputeTrajectories(newTrajectories, opts.num_cam);

outputTrajectories = inputTrajectories;
outputTrajectories(currentTrajectoriesInd) = [];
outputTrajectories = [smoothTrajectories; outputTrajectories];
%outputTrajectories = [outputTrajectories; smoothTrajectories];

% show merged tracklets in window 
if opts.visualize, trajectoriesVisualizePart3; end

end


