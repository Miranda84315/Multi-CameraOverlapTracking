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
        %-----這裡有大bug , 如果是[] 要怎麼排除 
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
    
    %correlationMatrix(impossibilityMatrix == 1) = -inf;
    correlationMatrix(sameLabels) = 1;
    
    % my design clustering
    correlationMatrix(sameLabels) = 0;
    for j=1:length(correlationMatrix)
        [~, correlationMatrix(j, j)] = max(correlationMatrix(j,:));
        %if correlationMatrix(j, correlationMatrix(j, j)) < 0.05
         %   correlationMatrix(j, j) = j;
        %end
    end
    
    % find which one is small
    for j=1:length(correlationMatrix)
        for k=1:length(correlationMatrix)
            if (correlationMatrix(j, j) == correlationMatrix(k,k)) && (j~=k)
                ind = correlationMatrix(j, j);
                max_value = max(correlationMatrix(j, ind), correlationMatrix(k, ind));
                if max_value == correlationMatrix(j, ind)
                    ind_temp = [k; ind];
                    matrix_temp = correlationMatrix(k, :);
                    matrix_temp(ind_temp) = -inf;
                    [~, correlationMatrix(k, k)] = max(matrix_temp);
                else
                    ind_temp = [j; ind];
                    matrix_temp = correlationMatrix(j, :);
                    matrix_temp(ind_temp) = -inf;
                    [~, correlationMatrix(j, j)] = max(matrix_temp);
                end
            end
        end
    end
    
    correlationLabel = cell(1, 1);
    m=1;
    for j=1:length(correlationMatrix)
        temp_label = [correlationMatrix(j, j), j];
        % flag : wheather j is already clustering 
        % flag_lonely : wheather j is lonely one, not clustering with other.
        flag = 0;
        flag_lonely = 0;
        if correlationMatrix(correlationMatrix(j, j), j) >=0.1
            for k=1:length(correlationLabel)
                if ~isempty(intersect(correlationLabel{k}, temp_label)) 
                    flag = 1;
                    % check if their have -inf in correlationMatrix
                    % if have -inf ,
                    % it means they should not correlation together.
                    for r= 1:length(correlationLabel{k})
                        if correlationMatrix(correlationLabel{k}(r), j) == -inf
                            flag=0;
                            flag_lonely = 1;
                        end
                    end
                    % correlationLabel{k} and j have intersect, so we
                    % clustering they by union set
                    if flag == 1
                        correlationLabel{k} = union(correlationLabel{k}, temp_label);
                    end
                    break
                end
            end
        end
        % clustering by j and j's 
        if flag ==0 && flag_lonely == 0 && correlationMatrix(correlationMatrix(j, j), j) >=0.1
            correlationLabel{m} = temp_label;
            m = m+1;
        else
            flag_lonely = 1;
        end
        % check is real real not in correlationLabel
        for k=1:length(correlationLabel)
            if ~isempty(intersect(correlationLabel{k}, j)) 
                flag_lonely = 0;
            end
        end
        % if j are flag_lonely, clustering by j-self length(correlationMatrix)
        if flag ==0 && flag_lonely == 1
            correlationLabel{m} = [j];
            m = m+1;
        end
    end
    
    
    
    
    
    % assign label result to each tracklet
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


