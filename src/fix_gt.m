% fix gt by frame different

frame_diff = [10, 26, 39, 0];
gt = load(fullfile(opts.dataset_path,'ground_truth', opts.experiment_name, 'gt_data.mat'));
gt = gt.gt;
% align frame
for icam=1:4
    ind = find(gt(:, 2) == icam);
    gt(ind, 1) = gt(ind, 1) - frame_diff(icam);
end

% only save frame>0
save_ind = find(gt(:, 1)>0);
gt = gt(save_ind, :);

% save to gt_data_fix.mat
filename_save = fullfile(opts.dataset_path,'ground_truth', opts.experiment_name, 'gt_data_fix.mat');
save(filename_save,'gt');

% run label_3D.py
command = strcat('C:/Users/Owner/Anaconda3/envs/tensorflow/python.exe src/labelGT/label_3D.py' , ...
    sprintf(' --track %s', num2str(opts.track)), ...
    sprintf(' --player %s', num2str(opts.player)), ...
    sprintf(' --day %s', '0412'));
system(command);

