function trajectoriesInWindow = findTrajectoriesInWindow(trajectories, startTime, endTime)
trajectoriesInWindow = [];

if isempty(trajectories), return; end

trajectoryStartFrame = [trajectories.segmentStart]; %cell2mat({trajectories.startFrame});
trajectoryEndFrame   = [trajectories.segmentEnd]; % cell2mat({trajectories.endFrame});
trajectoriesInWindow  = find( (trajectoryEndFrame >= startTime) .* (trajectoryStartFrame < endTime) );

