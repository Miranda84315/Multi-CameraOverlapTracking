function [allMets, metsBenchmark, metsMultiCam] = my_evaluateTracking(opts, seqmap, resDir, gtDataDir, benchmark, dataset_path)
%{
seqmap = seqMap;
resDir = eval_folder;
gtDataDir = gt_folder;
benchmark = 'Tracking';
dataset_path = opts.dataset_path;
%}
% Input:
% - seqmap
% Sequence map (e.g. `c2-train.txt` contains a list of all sequences to be 
% evaluated in a single run. These files are inside the ./seqmaps folder.
%
% - resDir
% The folder containing the tracking results. Each one should be saved in a
% separate .txt file with the name of the respective sequence (see ./res/data)
%
% - gtDataDir
% The folder containing the ground truth files.
%
% - benchmark
% The name of the benchmark, e.g. 'MOT15', 'MOT16', 'MOT17', 'DukeMTMCT'
%
% Output:
% - allMets
% Scores for each sequence
% 
% - metsBenchmark
% Aggregate score over all sequences
%
% - metsMultiCam
% Scores for multi-camera evaluation

% addpath(genpath('.'));
warning off;

% Benchmark specific properties
world = 0;
threshold = 0.5;
multicam = 1;


% Read sequence list
sequenceListFile = fullfile('seqmaps',seqmap);
allSequences = parseSequences2(sequenceListFile);
%fprintf('Sequences: \n');
disp(allSequences')
gtMat = [];
resMat = [];

% Evaluate sequences individually
allMets = [];
metsBenchmark = [];
metsMultiCam = [];

for ind = 1:4
    % load gt 
    gt = load(fullfile(dataset_path,'ground_truth', opts.experiment_name, 'gt_data_fix.mat'));
    gtdata = gt.gt;
    testInterval = [1: max(gtdata(:, 1))];

    cam = ind;
    startTimes = [1, 1, 1, 1];
    filter = gtdata(:, 2) == cam & ismember(gtdata(:, 1) + startTimes(cam) - 1, testInterval);
    gtdata = gtdata(filter,:);
    gtdata(:,[1 2]) = gtdata(:,[2 1]);
    gtdata = gtdata(:,2:end);
    
    gtdata = sortrows(gtdata,[1 2]);
    gtdata = double(gtdata);
    gtMat{ind} = gtdata;
    
% Parse result
    % Duke data format
    sequenceName = allSequences{ind};
    resFilename = fullfile(resDir, 'cam_trainval.txt');
    s = dir(resFilename);
    if exist(resFilename,'file') && s.bytes ~= 0
        resdata = dlmread(resFilename);
    else
        resdata = zeros(0,9);
    end
    cam = ind;
    resdata = resdata(:, [1 2 3 4 1+ind*4 : 4+ind*4]);

    % Filter rows by frame interval
    startTimes = [1, 1, 1, 1];
    resdata(~ismember(resdata(:,1) + startTimes(cam) - 1, testInterval),:) = [];
    resdata(:, [3 4]) = [];
    resdata = sortrows(resdata,[1 2]);
    resMat{ind} = resdata;
    
    % Sanity check
    frameIdPairs = resMat{ind}(:,1:2);
    [u,I,~] = unique(frameIdPairs, 'rows', 'first');
    hasDuplicates = size(u,1) < size(frameIdPairs,1);
    %{
if hasDuplicates
        ixDupRows = setdiff(1:size(frameIdPairs,1), I);
        dupFrameIdExample = frameIdPairs(ixDupRows(1),:);
        rows = find(ismember(frameIdPairs, dupFrameIdExample, 'rows'));
        
        errorMessage = sprintf('Invalid submission: Found duplicate ID/Frame pairs in sequence %s.\nInstance:\n', sequenceName);
        errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(1),:)), sprintf('\n')];
        errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(2),:)), sprintf('\n')];
        assert(~hasDuplicates, errorMessage);
    end
    %}
    
    % Evaluate sequence
    [metsCLEAR, mInf, additionalInfo] = CLEAR_MOT_HUN(gtMat{ind}, resMat{ind}, threshold, world);
    metsID = IDmeasures(gtMat{ind}, resMat{ind}, threshold, world);
    mets = [metsID.IDF1, metsID.IDP, metsID.IDR, metsCLEAR];
    allMets(ind).name = sequenceName;
    allMets(ind).m    = mets;
    allMets(ind).IDmeasures = metsID;
    allMets(ind).additionalInfo = additionalInfo;
    %fprintf('%s\n', sequenceName); printMetrics(mets); fprintf('\n');
    evalFile = fullfile(resDir, sprintf('eval_%s.txt',sequenceName));
    dlmwrite(evalFile, mets);
    
end

% Overall scores
metsBenchmark = evaluateBenchmark(allMets, world);
%fprintf('\n');
%fprintf(' ********************* Your %s Results *********************\n', benchmark);
printMetrics(metsBenchmark);
evalFile = fullfile(resDir, 'eval.txt');
dlmwrite(evalFile, metsBenchmark);

% Multicam scores
%     multicam = 0;

if multicam
    
    metsMultiCam = evaluateMultiCam(gtMat, resMat, threshold, world);
    %fprintf('\n');
    %fprintf(' ********************* Your %s MultiCam Results *********************\n', benchmark);
    %fprintf('IDF1   IDP    IDR\n');
    %fprintf('%.2f  %.2f  %.2f\n', metsMultiCam(1), metsMultiCam(2), metsMultiCam(3));
    
    evalFile = fullfile(resDir, 'eval_mc.txt');
    dlmwrite(evalFile, metsMultiCam);
    
end