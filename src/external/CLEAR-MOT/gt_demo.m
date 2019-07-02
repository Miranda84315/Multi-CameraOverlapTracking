%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2013 - MICC - Media Integration and Communication Center,
% University of Florence. 
% Iacopo Masi and Giuseppe Lisanti  <masi,lisanti> @dsi.unifi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
% groundtruth and results are examples. Ricreate these two structures if
% you wanto to use it in your own multi-target tracker.

generateData

gt = load(fullfile(opts.dataset_path,'ground_truth', opts.experiment_name, 'gt_data_3D.mat'));
gt_3D=gt.gt_3D;

resDir = [opts.experiment_root, filesep, opts.experiment_name, filesep, opts.eval_dir];
resFilename = fullfile(resDir, 'cam_trainval.txt');
resdata = dlmread(resFilename);
resdata = resdata(:, [1, 2, 3, 4]);

max_frame = max(resdata(:, 1));
max_id = max(resdata(:, 2));
res = {};
for frame= 1:max_frame
    res(frame).trackerData.idxTracks = [1:max_id];
    temp = resdata(resdata(:, 1)== frame, [2, 3, 4]);
    for id=1:max_id
        res(frame).trackerData.target(id).bbox = temp(temp(:, 1) == id,[2, 3]);
    end
end

gt = {};
for frame= 1:max_frame
    gt{frame} = gt_3D(gt_3D(:, 1)== frame, [2, 3, 4]);
end


VOCscore = 0.01;
dispON  = true;
ClearMOT = evaluateMOT(gt,res,VOCscore,dispON);
