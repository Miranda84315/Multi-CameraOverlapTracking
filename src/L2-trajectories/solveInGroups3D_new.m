function result = solveInGroups3D_new(opts, tracklets, labels)
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
    [~, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinity_new(tracklets(indices), params.beta, params.speed_limit, params.indifference_time);
    [spacetimeAffinity, distanceMatrix] = getFirstandFinal(opts, tracklets(indices), labels(indices));

    % compute the correlation matrix
    correlationMatrix =(appearanceAffinity + spacetimeAffinity) - 1;
    % new idea
    %correlationMatrix =distanceMatrix + (1- appearanceAffinity);
    
    correlationMatrix(impossibilityMatrix == 1) = -inf;
    correlationMatrix(sameLabels) = 1;
    
    % my design clustering
    correlationMatrix(sameLabels) = 0;
    for j=1:length(correlationMatrix)
        [~, correlationMatrix(j, j)] = max(correlationMatrix(j,:));
    end
    
    sizeCorrelation = length(correlationMatrix);
    for j=1:sizeCorrelation
        index_1 = j;    % self
        index_2 = correlationMatrix(j, j);  %max index
        index_3 = correlationMatrix(correlationMatrix(j, j), correlationMatrix(j, j)); %max index id
        if index_1 == index_3
            correlationMatrix(j, sizeCorrelation+1) = min(index_1, index_2);
        elseif  correlationMatrix(index_1, index_2) > 0.2
            correlationMatrix(j, sizeCorrelation+1) = min(index_1, index_2);
        else
            correlationMatrix(j, sizeCorrelation+1) = index_1;
        end
    end
    
    correlationLabel = cell(1, 1);
    m=1;
    for j=1:length(correlationMatrix)-1
        temp_label = [correlationMatrix(j, j), correlationMatrix(j, length(correlationMatrix))];
        flag = 0;
        for k=1:length(correlationLabel)
            if ~isempty(intersect(correlationLabel{k}, temp_label))
                correlationLabel{k} = union(correlationLabel{k}, temp_label);
                flag = 1;
                break
            end
        end
        if flag ==0
            correlationLabel{m} = temp_label;
            m = m+1;
        end
    end
    
    labelResult = zeros(length(correlationMatrix)-1, 1);
    for j=1:length(correlationLabel)
        index = correlationLabel{j};
        labelResult(index)=j;
    end
    
    result_appearance{i}.labels = labelResult;
    %result_appearance{i}.labels = correlationMatrix(:, sizeCorrelation+1);
    
    % show appearance group tracklets
    %if opts.visualize, trajectoriesVisualizePart2; end
    
    % solve the optimization problem
    solutionTime = tic;
    %{
    if strcmp(opts.optimization,'AL-ICM')
        result_appearance{i}.labels  = AL_ICM(sparse(correlationMatrix));
    elseif strcmp(opts.optimization,'KL')
        result_appearance{i}.labels  = KernighanLin(correlationMatrix);
    elseif strcmp(opts.optimization,'BIPCC')
        initialSolution = KernighanLin(correlationMatrix);
        result_appearance{i}.labels  = BIPCC(correlationMatrix, initialSolution);
    end
    %}
    
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

