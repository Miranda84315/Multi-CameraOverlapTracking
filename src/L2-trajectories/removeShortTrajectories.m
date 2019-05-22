function [ new_trajectories ] = removeShortTrajectories(trajectories, cutoffLength )
%This function removes short tracks that have not been associated with any
%trajectory. Those are likely to be false positives.



isShort = [];
for i= 1:length(trajectories)
    if (trajectories(i).endFrame - trajectories(i).startFrame) < cutoffLength
        isShort = [isShort; i];
    end
end
trajectories(isShort) =[];

new_trajectories = trajectories;

