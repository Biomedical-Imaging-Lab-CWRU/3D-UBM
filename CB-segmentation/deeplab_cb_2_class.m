clc; clear all; clear all;

vol_names = {'2018-032', '2019-047', '2019-12-14-F58', '2019-12-14-M67', 'Eye1', 'Eye2', 'Eye3'};


for i= 1 : length(vol_names)
    
    imDir = fullfile(['..\data\cb_2_class\images']);
    pxDir = fullfile(['..\data\cb_2_class\gt']);
    
    classNames = ["CB", "background"];
    labelIDs =  [1 0];
    
    imds = imageDatastore(imDir);
    pxds = pixelLabelDatastore(pxDir, classNames, labelIDs);
    
    
    all_files = imds.Files;
    all_gt = pxds.Files;
    
    idx_tr = find(~contains(all_files,vol_names{i}));
    idx_va = find(contains(all_files,vol_names{i}));
    
    
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
    
    tbl = countEachLabel(pxds_tr);
    imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
    classWeights = median(imageFreq) ./ imageFreq;    
    pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'ClassWeights', classWeights);
    lgraph = removeLayers(lgraph,'classification');
    lgraph = addLayers(lgraph, pxLayer);
    lgraph = connectLayers(lgraph,'softmax-out','labels');
    
    
    %     dcLayer = dicePixelClassificationLayer('Name','diceLayer');
    %     lgraph = removeLayers(lgraph,'classification');
    %     lgraph = addLayers(lgraph, dcLayer);
    %     lgraph = connectLayers(lgraph,'softmax-out','diceLayer');
    
    
    batch_size = 16;
    options = trainingOptions('adam', 'InitialLearnRate', 1e-4,...
        'L2Regularization',0.0005, ...
        'MaxEpochs',50, ...
        'MiniBatchSize',batch_size, ...
        'LearnRateDropFactor',0.2, ...
        'LearnRateDropPeriod',5, ...
        'LearnRateSchedule','piecewise', ...
        'Shuffle','every-epoch', ...
        'Verbose',true,...
        'VerboseFrequency',100,...
        'Plots', 'training-progress', ...
        'ValidationData', pximds_va, ...
        'ValidationFrequency', floor(length(imds_tr.Files)/batch_size), ...
        'ExecutionEnvironment', 'gpu',...
        'OutputFcn', @(info)stopIfAccuracyNotImproving(info, 10));
    
    
    
    [Trained_Net, info] = trainNetwork(pximds_tr, lgraph, options);
    
    
    tempdir = fullfile(['..\data\cb_2_class\results']);;
    pxdsResults = semanticseg(pximds_va,Trained_Net,'MiniBatchSize',4,'WriteLocation',tempdir, 'NamePrefix', vol_names{i});
    
    
    save(['deeplab_cb_2_class_en_face_dice_' vol_names{i} '.mat'], 'Trained_Net', 'info');
    
    save(['validation_cb_2_class_en_face_dice_' vol_names{i} '.mat'], 'pximds_va');
    
    
    
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