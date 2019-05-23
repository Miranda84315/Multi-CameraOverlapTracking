classdef DataVideoReader < handle
% Example use
%
% reader = DataVideoReader('D:/Code/MultiCamOverlap/dataset/');
% camera = 4;
% frame = 20; 
% figure, imshow(reader.getFrame(camera, frame));

    properties
        NumCameras = 4;
        NumFrames =  [449, 449, 449, 449];
        PartFrames = [449, 449, 449, 449];
        MaxPart = [0, 0, 0, 0];
        DatasetPath = '';
        VideoName = '';
        CurrentCamera = 1;
        CurrentPart = 0;
        PrevCamera = 1;
        PrevFrame = 0;
        PrevPart = 0;
        Video = [];
        lastFrame = [];
    end
    methods
        function obj = DataVideoReader(datasetPath, experimentname)
            obj.DatasetPath = datasetPath;
            %obj.Video = cv.VideoCapture(sprintf('%svideos/Player05/track3/cam%d.avi',obj.DatasetPath,obj.CurrentCamera), 'API','FFMPEG');
            obj.VideoName = experimentname;
            obj.Video = cv.VideoCapture(sprintf('%svideos/%s/cam%d.avi',obj.DatasetPath, obj.VideoName, obj.CurrentCamera), 'API','FFMPEG');
        end
        
        function img = getFrame(obj, iCam, iFrame)
            % DukeMTMC frames are 1-indexed
            assert(iFrame > 0 && iFrame <= obj.NumFrames(iCam),'Frame out of range');
            
            ksum = 0;
            for k = 0:obj.MaxPart(iCam)
               ksumprev = ksum;
               ksum = ksum + obj.PartFrames(k+1,iCam);
               if iFrame <= ksum
                  currentFrame = iFrame - 1 - ksumprev;
                  iPart = k;
                  break;
               end
            end
            
            if iPart ~= obj.CurrentPart || iCam ~= obj.CurrentCamera
                obj.CurrentCamera = iCam;
                obj.CurrentPart = iPart;
                obj.PrevFrame = -1;
                %obj.Video = cv.VideoCapture(sprintf('%svideos/Player05/track3/cam%d.avi',obj.DatasetPath,obj.CurrentCamera), 'API','FFMPEG');
                obj.Video = cv.VideoCapture(sprintf('%svideos/%s/cam%d.avi',obj.DatasetPath, obj.VideoName, obj.CurrentCamera), 'API','FFMPEG');
            end
            
            if currentFrame ~= obj.PrevFrame + 1
                obj.Video.PosFrames = currentFrame;
                
                if obj.Video.PosFrames ~= currentFrame
                    back_frame = max(currentFrame - 31, 0); % Keyframes every 30 frames
                    obj.Video.PosFrames =  back_frame;
                    while obj.Video.PosFrames < currentFrame
                        obj.Video.read;
                        back_frame = back_frame + 1;
                    end
                end

            end
            assert(obj.Video.PosFrames == currentFrame)
            img = obj.Video.read;
            
            % Keep track of last read
            obj.PrevCamera = iCam;
            obj.PrevFrame = currentFrame;
            obj.PrevPart = iPart;
        end
        
    end
end

