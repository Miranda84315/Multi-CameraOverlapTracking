% Demo visualizing OpenPose detections
opts = get_opts();

cam = 2;
frame = 188;

%% 

img = opts.reader.getFrame(cam, frame);
poses = data{cam, 2}(data{cam, 2}(:, 2) == frame, :);
bboxes = poses(:, 3:6);
img = insertObjectAnnotation(img,'rectangle',bboxes, 1:size( bboxes, 1), 'LineWidth', 3, 'FontSize', 18);
figure, imshow(img);