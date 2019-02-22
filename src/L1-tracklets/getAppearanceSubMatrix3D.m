function [ correlation ] = getAppearanceSubMatrix3D(num_cam, observations, featureVectors, threshold )
%{
observations =spatialGroupObservations;
featureVectors = allFeatures;
threshold = params.threshold;
%}
dist = zeros(length(observations));
for i = 1:length(observations)
    for j = i + 1:length(observations)
        features_1 = [];
        features_2 = [];
        for c = 1:num_cam
            features_1 = [features_1; featureVectors.appearance{observations(i), c}];
            features_2 = [features_2; featureVectors.appearance{observations(j), c}];
        end
        dist_temp = min(min(pdist2(features_1, features_2)));
        dist(i, j) = dist_temp;
        dist(j, i) = dist_temp;
    end
end
% -- as paper's fomula w_ij = ta - d(x_i,x_j)/ta
correlation = (threshold - dist)/ threshold;




