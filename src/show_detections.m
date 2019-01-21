% Demo visualizing OpenPose detections
opts = get_opts();

cam = 4;

load(fullfile(opts.dataset_path,'detections', sprintf('cam%d.mat',cam)));


%% 
for frame = 122900:123030
    % disappear -> 122911
    % start -> 123029
    img = opts.reader.getFrame(cam, frame);
    poses = detections(detections(:,1) == cam & detections(:,2) == frame,3:end);

    % Transform poses into boxes
    bboxes = poses(:,1:4);
    bboxes()
    img = insertObjectAnnotation(img,'rectangle',bboxes, ones(size(bboxes,1),1));
    figure, imshow(img);
    close all
end

