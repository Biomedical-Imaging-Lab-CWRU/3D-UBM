%%evaluate performance

clc; clear all; clear all;

vol_names = {'2019-12-14-F58', '2019-12-14-M67', 'Eye1', 'Eye2', 'Eye3'};

cnt = 1;

gt_vol_tot= [];
res_vol_tot= [];

for i= 1 : length(vol_names)
    
    imDir = fullfile(['..\data\cb_2_class_radial\images']);
    pxDir = fullfile(['..\data\cb_2_class_radial\gt']);
    
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
    
    gt_vol= [];
    res_vol= [];
    
    pxds_name = pxds_va.Files;
    [Y,ndx,dbg] = natsortfiles(pxds_name);
    
    
    for j= 1:length(pxds_va.Files)
        gt = imread(pxds_name{j});
        
        result = imresize(imread(fullfile(['..\data\cb_2_class_radial\results\' vol_names{i} '_' num2str(j) '.png'])),size(gt));
        
        %        result2= zeros(size(result));
        %        result2(result==1)=1;
        
        gt2  = 2* ones(size(gt));
        gt2(gt==1)=1;
        
        %         eval(j,:) = Evaluate2(gt2,result);
        
        %         gt_vol(:,:,j) = gt2;
        %         res_vol(:,:,j) = result;
        
        gt_vol(1:size(gt,1),1:size(gt,2),ndx(j)) = gt2;
        res_vol(1:size(gt,1),1:size(gt,2),ndx(j)) = result;
        cnt = cnt +1;
        
        
        %
        %         scores2(cnt,:) = Evaluate2(gt2,result);
        %         cnt = cnt +1;
    end
    
    %     eval = Evaluate2(gt_vol(:),res_vol(:));
    
    %     res{i,1} = vol_names{i};
    %     res{i,2} = eval;
    %     eval = Evaluate2(gt_vol(:),res_vol(:));
    
    %for 3 class confusion matrix
    %     gt_vol_tot = vertcat(gt_vol_tot, gt_vol(:));
    %     res_vol_tot = vertcat(res_vol_tot, res_vol(:));
    [c_matrix,Result,RefereceResult]= confusion.getMatrix(gt_vol(:),res_vol(:));
    
    x{i} = RefereceResult;
    %     res{i,1} = vol_names{i};
    %     res{i,2} = eval;
end


%%
try
    res_vol_pre = res_vol(:,:,361:720);
    gt_vol_pre = gt_vol(:,:,361:720);
    gt_vol_post = gt_vol(:,:,1:360);
    res_vol_post = res_vol(:,:,1:360);
    
    1+(sum(res_vol_pre(:) == 1)-sum(gt_vol_pre(:) == 1))/sum(gt_vol_pre(:) == 1)
    
    1+(sum(res_vol_post(:) == 1)-sum(gt_vol_post(:) == 1))/sum(gt_vol_post(:) == 1)
    
catch
end

%%
 for i=1:length(x)
     accuracy(i) = x{1,i}.AccuracyInTotal;
     sens(i) = x{1,i}.Sensitivity;
     spec(i) = x{1,i}.Specificity;
     preci(i) = x{1,i}.Precision;
     f1_s(i) = x{1,i}.F1_score;
 end
 
[mean(accuracy), mean(sens), mean(spec), mean(preci), mean(f1_s)]

[std(accuracy), std(sens), std(spec), std(preci), std(f1_s)]


function EVAL = Evaluate2(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by
% calculating the common performance measures: Accuracy, Sensitivity,
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
idx = (ACTUAL()==1);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL = [accuracy sensitivity specificity precision recall f_measure];
end