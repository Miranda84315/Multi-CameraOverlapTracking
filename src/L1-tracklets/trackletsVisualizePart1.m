%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHOW DETECTIONS IN WINDOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
color = ['r'; 'g'; 'b';'y'; 'm'];
h = figure(2);
clf('reset');
Img = cv.imread('C:\Users\Owner\Pictures\basketballCourt.png');
%imshow(opts.reader.getFrame(opts.current_camera,startFrame));
imshow(Img);
pause(1);
hold on;
scatter(detectionCenters(:,1),detectionCenters(:,2), color(2),'fill');
for group=1 : max(spatialGroupIDs)
    elements = find(spatialGroupIDs == group);
    scatter(detectionCenters(elements, 1),detectionCenters(elements, 2), color(group),'fill');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%