function checkTrajectories = checkFromEdge( newTrajectories,startTime )
%Every trajectories need to come from edge (expect start frame)

checkTrajectories = [];
for i = 1:length(newTrajectories)

    if(newTrajectories(i).segmentStart<=15 ||  newTrajectories(i).segmentStart< startTime)
        checkTrajectories = [checkTrajectories; newTrajectories(i)];
    else
        [a, ind] = min([newTrajectories(i).tracklets.startFrame]);
        data = newTrajectories(i).tracklets(ind).realdata;  
        min_frame = min(3, size(data, 1));
        fromEdge = 0;
        for k =1:min_frame
            fromEdge = fromEdge + IsEdge(data(k, 8:9));
        end
        if(fromEdge>=2)
            checkTrajectories = [checkTrajectories; newTrajectories(i)];
        end
    end
end

