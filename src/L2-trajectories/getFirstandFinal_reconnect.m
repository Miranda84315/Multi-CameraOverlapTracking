function [spacetimeAffinity, distanceMatrix] = getFirstandFinal_reconnect(opts, tracklets_window, labels)
%{
tracklets_window = tracklets_space;
labels = [1, 2];
%}

FirstFinalData = zeros(length(tracklets_window), 4);
for i=1:size(tracklets_window, 1)
    FirstFinalData(i, 1:2) = tracklets_window(i).center;
    FirstFinalData(i, 3:4) = tracklets_window(i).center;
end

startX          = repmat(FirstFinalData(:, 1), 1, length(tracklets_window));
startY          = repmat(FirstFinalData(:, 2), 1, length(tracklets_window));
endX            = repmat(FirstFinalData(:, 3), 1, length(tracklets_window));
endY            = repmat(FirstFinalData(:, 4), 1, length(tracklets_window));


xDiff           = endX - startX';
yDiff           = endY - startY';

distanceMatrix  = sqrt(xDiff.^2 + yDiff.^2);
spacetimeAffinity  =1 ./ (1 + distanceMatrix/200 );

params = opts.trajectories;
sameLabels  = pdist2(labels, labels) == 0;

% old method
%Frame     = [tracklets_window.segmentStart]';
%frameDifference = pdist2(Frame, Frame, @(frame1, frame2) (frame1 - frame2)/15);

% add New frame Difference Method
Frame_total = cell(length(tracklets_window),1);
for i=1:length(tracklets_window)
    %Frame_total{i} = [tracklets_window(i).startFrame:tracklets_window(i).endFrame;];
    Frame_total{i} = [tracklets_window(i).realdata(:, 1)];
end
frameDifference = zeros(length(tracklets_window));
for i=1:length(tracklets_window)
    for j = 1:length(tracklets_window)
        doubleData=intersect(Frame_total{i}, Frame_total{j});
        if isempty(doubleData)
            frameDifference(i, j) = 1;
        end
    end
end

[~, impossibilityMatrix, ~] = getSpaceTimeAffinity_new(tracklets_window, params.beta, params.speed_limit, params.indifference_time);


end