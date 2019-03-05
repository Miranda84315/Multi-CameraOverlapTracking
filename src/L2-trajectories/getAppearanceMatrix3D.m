function [ appearanceMatrix ] = getAppearanceMatrix3D(featureVectors, threshold )
%{
threshold = params.threshold;
featureVectors = featureVectors(indices);
%}
% Computes the appearance affinity matrix
dist = zeros(length(featureVectors));
for i = 1:length(featureVectors)
    for j = i + 1:length(featureVectors)
        features_1 = [];
        features_2 = [];
        for c = 1:num_cam
            features_1 = [features_1; featureVectors{i, c}];
            features_2 = [features_2; featureVectors{j, c}];
        end
        dist_temp = min(min(pdist2(features_1, features_2)));
        dist(i, j) = dist_temp;
        dist(j, i) = dist_temp;
    end
end


features = double(cell2mat(featureVectors'));
dist = pdist2(features, features);
appearanceMatrix = (threshold - dist)/ threshold;


