function [ reconnect_trajectories ] = reconnectTrajectories_new(opts, new_trajectories)
%This function removes short tracks that have not been associated with any
%trajectory. Those are likely to be false positives.
% new_trajectories = remove_trajectories;
params = opts.trajectories;

startFrame = [new_trajectories.startFrame]';
endFrame =[new_trajectories.endFrame]';

% find total start and end frame
start_video = min(startFrame);
end_video = max(endFrame);

% find index which is not complete trajectory
ind_reconnect = find(startFrame~=start_video | endFrame~=end_video);

startFrame = startFrame(ind_reconnect);
endFrame = endFrame(ind_reconnect);

% find their frame difference
% traj1 -> first, traj2 -> after
frame_diff = pdist2(endFrame, startFrame, @(frame1, frame2) (frame1 - frame2));
[traj1, traj2] = find(abs(frame_diff) < 20);
traj1 = ind_reconnect(traj1);
traj2 = ind_reconnect(traj2);
ind_repeat = (traj1 ~= traj2);
traj1 = traj1(ind_repeat);
traj2 = traj2(ind_repeat);

% calculate their appearance and motion 
if ~isempty(traj1 | traj2)
    correlationValue = [];
    for i=1:length(traj1)
        featureVectors      = {new_trajectories(traj1(i)).tracklets(end).feature, new_trajectories(traj2(i)).tracklets(1).feature};
        tracklets_space = [new_trajectories(traj1(i)).tracklets(end); new_trajectories(traj2(i)).tracklets(1)];
        
        appearanceAffinity = getAppearanceMatrix3D(opts.num_cam, featureVectors, params.threshold);
        [spacetimeAffinity, distanceMatrix] = getFirstandFinal_reconnect(opts, tracklets_space, [1, 2]);
        
        correlationMatrix =(appearanceAffinity + spacetimeAffinity) - 0.9;
        correlationValue = [correlationValue; correlationMatrix(1, 2)];
    end
    [correlationValue, ind] = sort(correlationValue, 'descend');
    traj1 = traj1(ind);
    traj2 = traj2(ind);
    
    % clustering result
    correlationLabel = cell(1, 1);
    m=1;
    for i= 1:length(correlationValue)
        flag = 0;
        if correlationValue(i) > 0
            temp_label = [traj1(i), traj2(i)];
            for k=1:length(correlationLabel)
                if ~isempty(intersect(correlationLabel{k}, temp_label)) 
                    double = 0;
                    flag = 1;
                    count = 0;
                    for r= 1:length(correlationLabel)
                        if ~isempty(intersect(correlationLabel{r}, temp_label)) 
                            count = count + 1;
                            compare = setxor(correlationLabel{r}, temp_label);
                            if length(compare) == 2
                                real_1 = find(ind_reconnect == compare(1));
                                real_2 = find(ind_reconnect == compare(2));
                                if length(intersect(startFrame(real_1):endFrame(real_1), startFrame(real_2):endFrame(real_2)))>20
                                    double = 1;
                                end
                            end
                        end
                    end
                    if count > 1
                        double = 1;
                    end
                    % correlationLabel{k} and j have intersect, so we
                    % clustering they by union set
                    if flag == 1 && double == 0
                        correlationLabel{k} = union(correlationLabel{k}, temp_label);
                    end
                    break
                end
            end
            % clustering by j and j's 
            if flag ==0 
                correlationLabel{m} = temp_label;
                m = m+1;
            end
        end
    end
    
    ind_delete = [];
    for i=1:length(correlationLabel)
        ind_connect = [correlationLabel{i}];
        ind_delete = [ind_delete, ind_connect(2:end)];
        
        for j=2:length(ind_connect)
            new_trajectories(ind_connect(1)).tracklets = [new_trajectories(ind_connect(1)).tracklets; new_trajectories(ind_connect(j)).tracklets];
            new_trajectories(ind_connect(1)).startFrame = min(new_trajectories(ind_connect(1)).startFrame, new_trajectories(ind_connect(j)).startFrame);
            new_trajectories(ind_connect(1)).endFrame = max(new_trajectories(ind_connect(1)).endFrame, new_trajectories(ind_connect(j)).endFrame);
            new_trajectories(ind_connect(1)).segmentStart = min(new_trajectories(ind_connect(1)).segmentStart, new_trajectories(ind_connect(j)).segmentStart);
            new_trajectories(ind_connect(1)).segmentEnd = max(new_trajectories(ind_connect(1)).segmentEnd, new_trajectories(ind_connect(j)).segmentEnd);
        end
    end
    new_trajectories(ind_delete) = [];
end
reconnect_trajectories = new_trajectories;
