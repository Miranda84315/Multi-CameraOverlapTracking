filename = 'D:/Code/MultiCamOverlap/dataset/ground_truth_temp/';

for player=11:20
    for track=1:8
        if player <10
            experiment_name = ['Player0', num2str(player), '/track', num2str(track)];
        else
            experiment_name = ['Player', num2str(player), '/track', num2str(track)];
        end
        filename_temp =fullfile( filename,experiment_name, '/gt_data.mat');
        load(filename_temp);
        
        s = double(max(gt(:, 1)));
        index = find(gt(:,1)== s);
        x1 = zeros(size(index));
        x1(1:end-1)=index(2:end);
        x1 - index;

        a = sum(gt(:,1) == s);
        b = sum(gt(:,1) == 1);

        n = size(gt);
        c = n(1)/s;
        if c~=20
            fprintf('player = %d, track= %d\n', player, track);
        end
    end
end
            

filename = 'D:/Code/MultiCamOverlap/dataset/ground_truth_temp/';

player = 10;
track = 5;
if player <10
    experiment_name = ['Player0', num2str(player), '/track', num2str(track)];
else
    experiment_name = ['Player', num2str(player), '/track', num2str(track)];
end
filename_temp =fullfile( filename,experiment_name, '/gt_data.mat');
load(filename_temp);

s = max(gt(:, 1));

for player_temp = 1:5
    for icam = 1:4
        max_frame = max(gt(gt(:, 2) ==icam & gt(:, 3) ==player_temp , 1) );
        last_value = gt(gt(:, 2) ==icam & gt(:, 3) ==player_temp & gt(:, 1) == max_frame, 2:7);
        fprintf(' --> max_frame = %d\n', max_frame);
        if max_frame~=s
            new_append = [max_frame+1:s]';
            last_value = repmat(last_value, s- max_frame, 1);
            new_append(:, 2:7) = last_value;
            gt = [gt; new_append];
        end
    end
    gt = sortrows(gt, [3,2,1]);
end
save(filename_temp, 'gt');