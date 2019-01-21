function [ correlation ] = getAppearanceSubMatrix(observations, featureVectors, threshold )
%{
observations =spatialGroupObservations;
featureVectors = allFeatures;
threshold = params.threshold;
%}
features = cell2mat(featureVectors.appearance(observations));
dist = pdist2(features, features);
% -- as paper's fomula w_ij = ta - d(x_i,x_j)/ta
correlation = (threshold - dist)/ threshold;




