% Demo visualizing OpenPose detections
opts = get_opts(, 6);

cam = 4;
frame = 1;
load(fullfile(opts.dataset_path, 'alpha_pose', opts.experiment_name, sprintf('cam%d.mat',cam)));
img = opts.reader.getFrame(cam, frame);
poses = detections(detections(:,1) == cam & detections(:,2) == frame,3:end);
bboxes = poses(:,1:4);
bboxes()
img = insertObjectAnnotation(img,'rectangle',bboxes, ones(size(bboxes,1),1));
figure, imshow(img);

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

