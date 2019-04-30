function result = solveInGroups3D(opts, tracklets, labels)
%{
tracklets = tracklets(inAssociation);
labels = trackletLabels(inAssociation);
%}
global trajectorySolverTime;

params = opts.trajectories;
if length(tracklets) < params.appearance_groups
    params.appearance_groups = 1;
end

featureVectors      = {tracklets.feature};
% adaptive number of appearance groups
% fixed number of appearance groups
%{
all_feature = [];
for i = 1:size(featureVectors, 2)
    if(cell2mat(featureVectors{i}'))
        %-----�o�̦��jbug , �p�G�O[] �n���ư� 
    all_feature = [all_feature; cell2mat(featureVectors{i}')];
    end
%}
appearanceGroups    = ones(size(tracklets, 1), 1);
% solve separately for each appearance group
allGroups = unique(appearanceGroups);

result_appearance = cell(1, length(allGroups));
for i = 1 : length(allGroups)
    
    fprintf('merging tracklets in appearance group %d\n',i);
    group       = allGroups(i);
    indices     = find(appearanceGroups == group);
    % -- sameLabel is a only symmetry is 1 else is 0  
    sameLabels  = pdist2(labels(indices), labels(indices)) == 0;
    
    % compute appearance and spacetime scores
    appearanceAffinity = getAppearanceMatrix3D(opts.num_cam, featureVectors(indices), params.threshold);
    [spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinity_new(tracklets(indices), params.beta, params.speed_limit, params.indifference_time);
    
    % compute the correlation matrix
    %correlationMatrix = (appearanceAffinity + spacetimeAffinity)/2;
    %correlationMatrix = correlationMatrix .* indifferenceMatrix;
    correlationMatrix =(appearanceAffinity + spacetimeAffinity - 1);
    correlationMatrix = correlationMatrix .* indifferenceMatrix;
    
    correlationMatrix(impossibilityMatrix == 1) = -inf;
    correlationMatrix(sameLabels) = 1;
    %correlationMatrix(correlationMatrix > 0.60) = 1;
    
    
    % show appearance group tracklets
    if opts.visualize, trajectoriesVisualizePart2; end
    
    % solve the optimization problem
    solutionTime = tic;
    if strcmp(opts.optimization,'AL-ICM')
        result_appearance{i}.labels  = AL_ICM(sparse(correlationMatrix));
    elseif strcmp(opts.optimization,'KL')
        result_appearance{i}.labels  = KernighanLin(correlationMatrix);
    elseif strcmp(opts.optimization,'BIPCC')
        initialSolution = KernighanLin(correlationMatrix);
        result_appearance{i}.labels  = BIPCC(correlationMatrix, initialSolution);
    end
    
    trajectorySolutionTime = toc(solutionTime);
    trajectorySolverTime = trajectorySolverTime + trajectorySolutionTime;
    
    result_appearance{i}.observations = indices;
end


% collect independent solutions from each appearance group
result.labels       = [];
result.observations = [];

for i = 1:numel(unique(appearanceGroups))
    result = mergeResults(result, result_appearance{i});
end

[~,id]              = sort(result.observations);
result.observations = result.observations(id);
result.labels       = result.labels(id);


