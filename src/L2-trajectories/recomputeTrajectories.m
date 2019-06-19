function newTrajectories = recomputeTrajectories( newTrajectories, num_cam )
%RECOMPUTETRAJECTORIES Summary of this function goes here
%   Detailed explanation goes here
% num_cam = opts.num_cam;
% newTrajectories = newTrajectories;

segmentLength = 5;%15;

for i = 1:length(newTrajectories)

    segmentStart = newTrajectories(i).segmentStart;
    segmentEnd = newTrajectories(i).segmentEnd;
    
    numSegments = (segmentEnd + 1 - segmentStart) / segmentLength;
    
    alldata = {newTrajectories(i).tracklets(:).data};
    alldata = cell2mat(alldata');
    alldata = sortrows(alldata,2);
    [~, uniqueRows] = unique(alldata(:,1));
    
    alldata = alldata(uniqueRows,:);
    dataFrames = alldata(:,1);
    
    frames = segmentStart:segmentEnd;
    interestingFrames = round([min(dataFrames):segmentLength:frames(end), max(dataFrames)]);
    %interestingFrames = round([min(dataFrames), frames(1) + segmentLength/2:segmentLength:frames(end),  max(dataFrames)]);
        
    keyData = alldata(ismember(dataFrames,interestingFrames),:);
    
%     for k = size(keyData,1)-1:-1:1
%         
%         while keyData(k,2) == keyData(k+1,2)
%             keyData(k+1,:) = [];
%         end
%         
%     end
    
    keyData(:,2) = -1;
    % only smooth 3D_x, 3Dy
    %newData = alldata(:, 1:4);
    newData = fillTrajectories(keyData(:, 1:4));
    newData = sortrows(newData, 1);
    
    for k=1:size(newData, 1)
        frame = newData(k, 1);
        ind = find(alldata(:, 1)== frame);
        if isempty(ind)
            newData(k, 5:20) = -1;
        else
            newData(k, 5:20) = alldata(ind, 5:20);
        end
    end
        
        
    
    %ind = alldata(:,1) == newData(:,1);
    %newData(ind, 5:20) = alldata(ind, 5:20);
    % add cam1~cam4's original data
    
    
    newTrajectory = newTrajectories(i);
    sampleTracklet = newTrajectories(i).tracklets(1);
    newTrajectory.tracklets = [];
    
    realdata = {newTrajectories(i).tracklets(:).realdata};
    realdata = cell2mat(realdata');
    
    for k = 1:numSegments
       
        tracklet = sampleTracklet;
        tracklet.segmentStart = segmentStart + (k-1)*segmentLength;
        tracklet.segmentEnd   = tracklet.segmentStart + segmentLength - 1;
        
        trackletFrames = tracklet.segmentStart:tracklet.segmentEnd;
        
        
        rows = ismember(newData(:,1), trackletFrames);
        rows2 = ismember(realdata(:,1), trackletFrames);
        
        tracklet.data = newData(rows,:);
        tracklet.center =  [median(newData(rows,3)),median(newData(rows,4))] ;
        tracklet.realdata =  realdata(rows2,:);
        
        tracklet.startFrame = min(tracklet.data(:,1));
        tracklet.endFrame = max(tracklet.data(:,1));
%         if isempty(tracklet.data)
%             tracklet.startFrame = min(tracklet.realdata(:,1));
%             tracklet.endFrame = max(tracklet.realdata(:,1));
%         end
        newTrajectory.startFrame = min(newTrajectory.startFrame, tracklet.startFrame);
        newTrajectory.endFrame = max(newTrajectory.endFrame, tracklet.endFrame);
        
        newTrajectory.tracklets = [newTrajectory.tracklets; tracklet];
        
    end
    
    newTrajectories(i) = newTrajectory;
    

end

