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
[traj1, traj2] = find((frame_diff==1) | (frame_diff ==0));

% combined each two trajectory.
if ~isempty(traj1 | traj2) %|| (new_trajectories(traj1).segmentStart ~= new_trajectories(traj2).segmentStart)
    for i=1:length(traj1)
        new_trajectories(traj1(i)).tracklets = [new_trajectories(traj1(i)).tracklets; new_trajectories(traj2(i)).tracklets];
        new_trajectories(traj1(i)).startFrame = min(new_trajectories(traj1(i)).startFrame, new_trajectories(traj2(i)).startFrame);
        new_trajectories(traj1(i)).endFrame = max(new_trajectories(traj1(i)).endFrame, new_trajectories(traj2(i)).endFrame);
        new_trajectories(traj1(i)).segmentStart = min(new_trajectories(traj1(i)).segmentStart, new_trajectories(traj2(i)).segmentStart);
        new_trajectories(traj1(i)).segmentEnd = max(new_trajectories(traj1(i)).segmentEnd, new_trajectories(traj2(i)).segmentEnd);
    end
    new_trajectories(traj2) = [];
    reconnect_trajectories = new_trajectories;
else
    reconnect_trajectories = new_trajectories;
end
        

