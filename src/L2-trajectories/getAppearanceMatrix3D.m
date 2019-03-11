function [ appearanceMatrix ] = getAppearanceMatrix3D(num_cam, featureVectors, threshold )
%{
threshold = params.threshold;
featureVectors = featureVectors(indices);
num_cam = opts.num_cam;
%}
% Computes the appearance affinity matrix
dist = zeros(length(featureVectors));
for i = 1:length(featureVectors)
    for j = i + 1:length(featureVectors)
        features_1 = [];
        features_2 = [];
        for c = 1:num_cam
            if(~isempty(featureVectors{1, i}{1, c}))
                features_1 = [features_1; featureVectors{1, i}{1, c}];
            end
            if(~isempty(featureVectors{1, j}{1, c}))
                features_2 = [features_2; featureVectors{1, j}{1, c}];
            end
        end
        dist_temp = min(min(pdist2(features_1, features_2)));
        dist(i, j) = dist_temp;
        dist(j, i) = dist_temp;
    end
end

appearanceMatrix = (threshold - dist)/ threshold;


