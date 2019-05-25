function output = fillandSmoothTrajectories( detections, num_cam )
    %{
    detections = trackerOutputFilled;
    num_cam = opts.num_cam;
    %}
    % 1. fill -1 data
    detections = sortrows(detections,[2 1]);
    output = [];
    personIDs = unique(detections(:, 2));
    count = 0;
    max_frame = max(detections(:, 1)) - min(detections(:, 1)) + 1;

    for i = 1 : length(personIDs)
        personID = personIDs(i);
        temp_size = size(detections(detections(:, 2) ==personID, 1:4), 1);
        cam_detection = zeros(temp_size, 20);
        cam_detection(:, 1:4) = detections(detections(:, 2) ==personID, 1:4);
        cam_detection = sortrows(cam_detection, 1);
        for cam=1:num_cam
            index = cam*4 + 1;
            % filter camera's result and delete -1's data
            relevantDetections = detections(detections(:, 2) == personID, [1:2, index:index+3]);
            index_empty = relevantDetections(:, 3)~=-1;
            relevantDetections = relevantDetections(relevantDetections(:, 3)~=-1, :);

            % find missFrame
            startFrame = min( relevantDetections(:, 1));
            endFrame = max( relevantDetections(:, 1));
            missingFrames = setdiff([startFrame:endFrame]', relevantDetections(:, 1));

            if isempty(missingFrames)
                %cam_detection(:, index:index+3) = relevantDetections(:, 3:6);
                cam_detection(index_empty, index:index+3) = relevantDetections(:, 3:6);
                continue;
            end

            % start to fill data
            % find frame is only big then 1-> find the start index
            frameDiff = diff(missingFrames') > 1;
            startInd = find([1, frameDiff]);
            endInd = find([frameDiff, 1]);

            for k = 1:length(startInd)
                % initialization miss detection
                interpolatedDetections = zeros( missingFrames(endInd(k)) - missingFrames(startInd(k)) + 1, 6);
                interpolatedDetections(:,2) = personID;
                interpolatedDetections(:,1) = [missingFrames(startInd(k)):missingFrames(endInd(k))]';

                % find prev and post detection from this startInd and endInd
                preDetection = relevantDetections(relevantDetections(:, 1) == missingFrames(startInd(k)) - 1, :);
                postDetection = relevantDetections(relevantDetections(:, 1) == missingFrames(endInd(k)) + 1, :);

                for j = 3:size(relevantDetections, 2)
                    interpolatedDetections(:, j) = linspace(preDetection(j), postDetection(j), size(interpolatedDetections, 1));
                end

            relevantDetections = [relevantDetections; interpolatedDetections];
            end

            relevantDetections = sortrows(relevantDetections, 1);
            % if zero is in the final detection, can not be fill, so we use
            % this to fill -1
            if (size(relevantDetections, 1)~= size(cam_detection, 1))
                relevantDetections(size(relevantDetections, 1):size(cam_detection, 1), :) = -1;
            end
            cam_detection(:, index:index+3) = relevantDetections(:, 3:6);
        end
        output = [output; cam_detection];
    end
end