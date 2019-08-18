%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2013 - MICC - Media Integration and Communication Center,
% University of Florence. 
% Iacopo Masi and Giuseppe Lisanti  <masi,lisanti> @dsi.unifi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [TP, FN, FP, IDSW, MOTA] = gt_demo(opts)

    gt = load(fullfile(opts.dataset_path,'ground_truth', opts.experiment_name, 'gt_data_3D.mat'));
    gt_3D=gt.gt_3D;

    resDir = [opts.experiment_root, filesep, opts.experiment_name, filesep, opts.eval_dir];
    resFilename = fullfile(resDir, 'cam_trainval.txt');
    resdata = dlmread(resFilename);
    resdata = resdata(:, [1, 2, 3, 4]);

    max_frame = min(max(resdata(:, 1)), max(gt_3D(:, 1)));
    max_id = max(resdata(:, 2));
    res = {};
    for frame= 1:max_frame
        res(frame).trackerData.idxTracks = [1:max_id];
        temp = resdata(resdata(:, 1)== frame, [2, 3, 4]);
        for id=1:max_id
            res(frame).trackerData.target(id).bbox = temp(temp(:, 1) == id,[2, 3]);
            if isempty(temp(temp(:, 1) == id,[2, 3]))
                res(frame).trackerData.target(id).bbox = [-1, -1];
            end
        end
    end

    gt = {};
    for frame= 1:max_frame
        gt{frame} = gt_3D(gt_3D(:, 1)== frame, [2, 3, 4]);
    end


    VOCscore = 0.5;
    dispON  = false;
    ClearMOT = evaluateMOT(gt, res, VOCscore, dispON);
    
    TP = ClearMOT.TP;
    FN = ClearMOT.FN;
    FP = ClearMOT.FP;
    IDSW = ClearMOT.IDSW;
    MOTA = ClearMOT.MOTA;
    
end
