function [motionScores, impossibilityMatrix] = motionAffinity_new(detectionCenters, detectionFrames, speedLimit)
% This function computes the motion affinities given a set of detections.
%{
detectionCenters=spatialGroupDetectionCenters;
detectionFrames=spatialGroupDetectionFrames;

speedLimit=params.speed_limit;

%}
% A simple motion prediction is performed from a source detection to
% a target detection to compute the prediction error.

numDetections       = size(detectionCenters,1);
impossibilityMatrix = zeros(length(detectionFrames));

frameDifference = pdist2(detectionFrames, detectionFrames);
% -- for calucate ,to repeat mat numDetection times
%velocityX       = repmat(estimatedVelocity(:,1), 1, numDetections );
%velocityY       = repmat(estimatedVelocity(:,2), 1, numDetections );
centerX         = repmat(detectionCenters(:,1), 1, numDetections );
centerY         = repmat(detectionCenters(:,2), 1, numDetections );

% Check if speed limit is violated 
xDiff = centerX - centerX';
yDiff = centerY - centerY';
distanceMatrix = sqrt(xDiff.^2 + yDiff.^2);

% -- if the maxRequiredSpeedMatrix > speedLimit then is inf , and
% -- impossibilityMatrix = 1
maxRequiredSpeedMatrix = distanceMatrix ./ abs(frameDifference);
impossibilityMatrix(maxRequiredSpeedMatrix > speedLimit) = 1;

motionScores = 1 ./ (1 + distanceMatrix/100 );
%motionScores = 1 - 0.005*distanceMatrix;

