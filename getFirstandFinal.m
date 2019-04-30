function FirstFinalData = getFirstandFinal(opts, tracklets_window, labels)
%{
tracklets_window = tracklets(inAssociation);
labels = trackletLabels(inAssociation);
%}

FirstFinalData = zeros(length(tracklets_window), 4);
for i=1:size(tracklets_window, 1)
    FirstFinalData(i, 1:2) = tracklets_window(i).realdata(1, 2:3);
    FirstFinalData(i, 3:4) = tracklets_window(i).realdata(end, 2:3);
end

startX          = repmat(FirstFinalData(:, 1), 1, length(tracklets_window));
startY          = repmat(FirstFinalData(:, 2), 1, length(tracklets_window));
endX            = repmat(FirstFinalData(:, 3), 1, length(tracklets_window));
endY            = repmat(FirstFinalData(:, 4), 1, length(tracklets_window));


xDiff           = endX - startX';
yDiff           = endY - startY';

distanceMatrix  = sqrt(xDiff.^2 + yDiff.^2);
distanceMatrix  =1 ./ (1 + distanceMatrix/100 );

params = opts.trajectories;
sameLabels  = pdist2(labels, labels) == 0;

Frame     = [tracklets_window.segmentStart]';
frameDifference = pdist2(Frame, Frame, @(frame1, frame2) (frame1 - frame2)/15);

[~, impossibilityMatrix, ~] = getSpaceTimeAffinity_new(tracklets_window, params.beta, params.speed_limit, params.indifference_time);
    
distanceMatrix(frameDifference  ~= 1) = -inf;
distanceMatrix(sameLabels) = 1;



end