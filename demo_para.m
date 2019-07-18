% Use for run feature extraction

TP = zeros(40, 8);
FN = zeros(40, 8);
FP = zeros(40, 8);
IDSW = zeros(40, 8);
MOTA = zeros(40, 8);
single = cell(40, 9);
player_spec = [4,22, 18];

for player=1:40
    for track=1:8
        if ~((player==4 & ismember(track, [5:8])) | player==18 )
            opts = get_opts(player, track);
            create_experiment_dir(opts);
            %compute_L0_features(opts);
            %compute_L1_tracklets3D_cam(opts);
            %compute_L2_trajectories3D(opts);
            %fix_gt;
            [TP_temp, FN_temp, FP_temp, IDSW_temp, MOTA_temp] = gt_demo(opts);
            TP(player, track) = TP_temp;
            FN(player, track) = FN_temp;
            FP(player, track) = FP_temp;
            IDSW(player, track) = IDSW_temp;
            MOTA(player, track) = MOTA_temp;

            [allMets, metsBenchmark, metsMultiCam] = my_evaluate(opts);
            single_temp = reshape([allMets.m], 17,4)';
            single_temp = single_temp(:, [1, 2, 3, 6, 8, 10:16]);
            single_temp = [single_temp; metsBenchmark(:, [1, 2, 3, 6, 8, 10:16])];
            single{player, track} = single_temp;
        end
    end
    
    if ~ismember(player, player_spec)
        for icam=1:5
            avg_temp = [];
            for track=1:8
                avg_temp = [avg_temp; single{player, track}(icam, :)];
            end
            single{player, 9}(icam, [1:4, 11:12]) = mean(avg_temp(:, [1:4, 11:12]));
            single{player, 9}(icam, [5:10]) = sum(avg_temp(:, [5:10]));
        end
        
    elseif (player == 4)
        for icam=1:5
            avg_temp = [];
            for track=1:4
                avg_temp = [avg_temp; single{player, track}(icam, :)];
            end
            single{player, 9}(icam, [1:4, 11:12]) = mean(avg_temp(:, [1:4, 11:12]));
            single{player, 9}(icam, [5:10]) = sum(avg_temp(:, [5:10]));
        end
        
    elseif (player == 22)
        for icam=1:5
            avg_temp = [];
            for track=2:8
                avg_temp = [avg_temp; single{player, track}(icam, :)];
            end
            single{player, 9}(icam, [1:4, 11:12]) = mean(avg_temp(:, [1:4, 11:12]));
            single{player, 9}(icam, [5:10]) = sum(avg_temp(:, [5:10]));
        end
    end
end

single_result = [];
for icam=1:5
    avg_temp = [];
    for player = 1:40
        if ~isempty(single{player, 9})
            avg_temp = [avg_temp; single{player, 9}(icam, :)];
        end
    end
    single_result(icam, [1:4, 11:12]) = mean(avg_temp(:, [1:4, 11:12]));
    single_result(icam, [5:10]) = sum(avg_temp(:, [5:10]));
end

for player=1:40
    if player==4
        MOTA(player, 9) = mean(MOTA(player, 1:4));
    elseif player == 22
        MOTA(player, 9) = mean(MOTA(player, 2:8));
    else
        MOTA(player, 9) = mean(MOTA(player, 1:8));
    end
end


fprintf('MOTA = %f \n', mean(mean(MOTA)));
fprintf('TP = %d \n', sum(sum(TP)));
fprintf('FP = %d \n', sum(sum(FP)));
fprintf('FN = %d \n', sum(sum(FN)));
fprintf('IDSW = %d \n', sum(sum(IDSW)));

    
filename_temp = fullfile(opts.experiment_root, '/single_result.mat')
save(filename_temp, 'single_result');
filename_temp = fullfile(opts.experiment_root, '/single.mat')
save(filename_temp, 'single');
filename_temp = fullfile(opts.experiment_root, '/MOTA.mat')
save(filename_temp, 'MOTA');
filename_temp = fullfile(opts.experiment_root, '/TP.mat')
save(filename_temp, 'TP');
filename_temp = fullfile(opts.experiment_root, '/FP.mat')
save(filename_temp, 'FP');
filename_temp = fullfile(opts.experiment_root, '/FN.mat')
save(filename_temp, 'FN');
filename_temp = fullfile(opts.experiment_root, '/IDSW.mat')
save(filename_temp, 'IDSW');


