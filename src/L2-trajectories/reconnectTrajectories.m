function [ reconnect_trajectories ] = reconnectTrajectories(new_trajectories)
%This function removes short tracks that have not been associated with any
%trajectory. Those are likely to be false positives.


endFrame = [];
startFrame = [];
for i= 1:length(new_trajectories)
    endFrame = [endFrame; new_trajectories(i).endFrame];
    startFrame = [startFrame; new_trajectories(i).startFrame];
end

% find each endFrame - startFrame
% if their frame difference == 1, then combined two trajectory together.
frame_diff = pdist2(endFrame, startFrame, @(frame1, frame2) (frame1 - frame2));
[trajectory1, trajectory2] = find(frame_diff==1);

% combined each two trajectory.
if ~isempty(trajectory1 | trajectory2)
    for i=1:length(trajectory1)
        
    end
end
