function [motionScores, impossibilityMatrix] = motionAffinity(detectionCenters,detectionFrames,estimatedVelocity, speedLimit, beta)
% This function computes the motion affinities given a set of detections.
% A simple motion prediction is performed from a source detection to
% a target detection to compute the prediction error.

numDetections       = size(detectionCenters,1);
impossibilityMatrix = zeros(length(detectionFrames));

frameDifference = pdist2(detectionFrames, detectionFrames);
% -- for calucate ,to repeat mat numDetection times
velocityX       = repmat(estimatedVelocity(:,1), 1, numDetections );
velocityY       = repmat(estimatedVelocity(:,2), 1, numDetections );
centerX         = repmat(detectionCenters(:,1), 1, numDetections );
centerY         = repmat(detectionCenters(:,2), 1, numDetections );

% -- from forward to back , to cal the error using linear model
errorXForward = centerX + velocityX.*frameDifference - centerX';
errorYForward = centerY + velocityY.*frameDifference - centerY';

% -- from back to forward , to cal the error using linear model
errorXBackward = centerX' + velocityX' .* -frameDifference' - centerX;
errorYBackward = centerY' + velocityY' .* -frameDifference' - centerY;

errorForward  = sqrt( errorXForward.^2  + errorYForward.^2);
errorBackward = sqrt( errorXBackward.^2 + errorYBackward.^2);

% Only upper triangular part is valid
% -- because errorForward(i, j) only cal the i<j case , the i>j case is not valid 
% -- so using triu(x) + triu(x)' to product the symmetry matrix
predictionError = min(errorForward, errorBackward);
predictionError = triu(predictionError) + triu(predictionError)';

% Check if speed limit is violated 
xDiff = centerX - centerX';
yDiff = centerY - centerY';
distanceMatrix = sqrt(xDiff.^2 + yDiff.^2);

% -- if the maxRequiredSpeedMatrix > speedLimit then is inf , and
% -- impossibilityMatrix = 1
maxRequiredSpeedMatrix = distanceMatrix ./ abs(frameDifference);
predictionError(maxRequiredSpeedMatrix > speedLimit) = inf;
impossibilityMatrix(maxRequiredSpeedMatrix > speedLimit) = 1;

motionScores = 1 - beta*predictionError;

