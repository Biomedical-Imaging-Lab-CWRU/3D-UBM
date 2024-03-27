clc; clear all; clear all;
%to appease reviewers
% I think the idea is to compare how models trained on one reader performs
% on other reader. I am not sure if the results ever made it to the paper. 
% or just used in reviewer notes.


vol_names_reader1 = {'2019-12-14-F58', '2019-12-14-M67', 'Eye1'};
vol_names_reader2 = {'Eye2', 'Eye3'};


for i= 1 : 2
    
    imDir = fullfile(['..\data\data_reviewer\cb_3_class_radial\images']);
    pxDir = fullfile(['..\data\data_reviewer\cb_3_class_radial\gt']);
    
    classNames = ["CB", "CP", "background"];
    labelIDs =  [2 1 0];
    
    imds = imageDatastore(imDir);
    pxds = pixelLabelDatastore(pxDir, classNames, labelIDs);
    
    
    all_files = imds.Files;
    all_gt = pxds.Files;
    
    if i==1
        idx_tr = find(~contains(all_files,vol_names_reader1));
        idx_va = find(contains(all_files,vol_names_reader1));
    else
        idx_tr = find(~contains(all_files,vol_names_reader2));
        idx_va = find(contains(all_files,vol_names_reader2));
    end
    
    imds_tr = subset(imds,idx_tr);
    pxds_tr = subset(pxds,idx_tr);
    imds_va = subset(imds,idx_va);
    pxds_va = subset(pxds,idx_va);
    
    augmenter = imageDataAugmenter('RandRotation',[-10 10],'RandXReflection',true,'RandScale',[0.7 1.3],'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
    pximds_tr = pixelLabelImageDatastore(imds_tr, pxds_tr,'DataAugmentation',augmenter, 'OutputSize', [256 352],'ColorPreprocessing','gray2rgb');
    pximds_va = pixelLabelImageDatastore(imds_va, pxds_va, 'OutputSize', [256 352],'ColorPreprocessing','gray2rgb');
    
    %% Net
    imageSize = [256 352];
    numClasses = numel(classNames);
    lgraph = deeplabv3plusLayers(imageSize,numClasses,'resnet18');
    
    %     tbl = countEachLabel(pxds_tr);
    %     imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
    %     classWeights = median(imageFreq) ./ imageFreq;
    %     pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'ClassWeights', classWeights);
    %     lgraph = removeLayers(lgraph,'classification');
    %     lgraph = addLayers(lgraph, pxLayer);
    %     lgraph = connectLayers(lgraph,'softmax-out','labels');
    
    
    dcLayer = dicePixelClassificationLayer('Name','diceLayer');
    lgraph = removeLayers(lgraph,'classification');
    lgraph = addLayers(lgraph, dcLayer);
    lgraph = connectLayers(lgraph,'softmax-out','diceLayer');
    
    
    batch_size = 4;
    options = trainingOptions('adam', 'InitialLearnRate', 1e-4,...
        'L2Regularization',0.0005, ...
        'MaxEpochs',50, ...
        'MiniBatchSize',batch_size, ...
        'LearnRateDropFactor',0.2, ...
        'LearnRateDropPeriod',3, ...
        'LearnRateSchedule','piecewise', ...
        'Shuffle','every-epoch', ...
        'Verbose',true,...
        'VerboseFrequency',100,...
        'Plots', 'training-progress', ...
        'ValidationData', pximds_va, ...
        'ValidationFrequency', floor(length(imds_tr.Files)/batch_size), ...
        'ExecutionEnvironment', 'gpu',...
        'OutputFcn', @(info)stopIfAccuracyNotImproving(info, 5));
    
    
    
    [Trained_Net, info] = trainNetwork(pximds_tr, lgraph, options);
    
    %
    tempdir = fullfile(['..\data\data_reviewer\cb_3_class_radial\results']);
    pxdsResults = semanticseg(pximds_va,Trained_Net,'MiniBatchSize',4,'WriteLocation',tempdir, 'NamePrefix', vol_names{i});
    
    
    save(['deeplab_cb_3_class_radial_dice_for_reviewers_' num2str(i) '.mat'], 'Trained_Net', 'info');
    
    save(['validation_cb_3_class_radial_dice_for_reviewers_' num2str(i) '.mat'], 'pximds_va');
    
    
    
end



function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end