function [stAffinity, impossibilityMatrix, indiffMatrix] = getLocationAffinity(tracklets, indifferenceLimit)
threshold=1000;
numTracklets = length(tracklets);

[~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(tracklets);

centerFrame     = round(mean(intervals,2));
frameDifference = pdist2(centerFrame, centerFrame, @(frame1, frame2) frame1 - frame2);
overlapping     = pdist2(intervals,intervals, @overlapTest);
centers         = 0.5 * (endpoint + startpoint);
centersDistance = pdist2(centers,centers);

merging         = (centersDistance < 5) & overlapping;

% build impossibility matrix
impossibilityMatrix = zeros(numTracklets);
impossibilityMatrix(overlapping == 1 & merging ~=1) = 1;

% compute indifference matrix
timeDifference  = frameDifference .* (frameDifference > 0);
timeDifference  = timeDifference + timeDifference';
indiffMatrix    = 1 - sigmf(timeDifference,[0.1 indifferenceLimit/2]);

% compute space-time affinities
newcentersDistance = (threshold - centersDistance)/ threshold;
stAffinity      = newcentersDistance; %1 ./ (1 + centersDistance);

end

function overlap = overlapTest(interval1, interval2)

duration1       = interval1(2) - interval1(1);
duration2       = interval2(:,2) - interval2(:, 1);

i1              = repmat(interval1,size(interval2,1), 1);
unionMin        = min([i1, interval2], [], 2);
unionMax        = max([i1, interval2], [], 2);

overlap         = double(duration1 + duration2 - unionMax + unionMin >= 0);

end






