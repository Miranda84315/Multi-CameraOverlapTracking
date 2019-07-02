function [allMets, metsBenchmark, metsMultiCam] = my_evaluate(opts)

evalNames   = {'trainval'};
seqMap      = sprintf('Tracking-%s.txt', evalNames{opts.sequence});
eval_folder = [opts.experiment_root, filesep, opts.experiment_name, filesep, opts.eval_dir];
gt_folder   = [opts.dataset_path, 'ground_truth/', opts.experiment_name];
[allMets, metsBenchmark, metsMultiCam] = my_evaluateTracking(opts, seqMap, eval_folder, gt_folder, 'DukeMTMCT',opts.dataset_path);
