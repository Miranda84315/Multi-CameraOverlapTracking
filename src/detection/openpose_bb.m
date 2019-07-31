
dataset_path = 'D:/Code/MultiCamOverlap/dataset/openpose';
for player = 19:40
    for track = 1:8
        if player <10
            root = ['Player0', num2str(player), '/track', num2str(track)];
        else
            root = ['Player', num2str(player), '/track', num2str(track)];
        end
        
        for icam = 1:4
            load(fullfile(dataset_path, root, sprintf('cam%d_pose.mat', icam)));
            detections_openpose = detections;
            detections = [];
            for i=1:length(detections_openpose)
                pose = detections_openpose(i, 3:end)
                bb = pose2bb( pose, 0.05);
                [newbb, newpose] = scale_bb(bb, pose,1.25);
                feet_x = round(newbb(1) + newbb(3)/2);
                feet_y = round(newbb(2) + newbb(4));
                
                temp_detection = [detections_openpose(i,1:2), newbb, 1, feet_x, feet_y];
                detections = [detections; temp_detection];
            end
            save(fullfile(dataset_path, root, sprintf('cam%d.mat', icam)), 'detections');
        end
    end
end