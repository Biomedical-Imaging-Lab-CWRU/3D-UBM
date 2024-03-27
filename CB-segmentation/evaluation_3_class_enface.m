%%evaluate performance

clc; clear all; clear all;
vol_names = {'2019-12-14-F58', '2019-12-14-M67', 'Eye1', 'Eye2', 'Eye3'};
cnt = 1;

gt_vol_tot= [];
res_vol_tot= [];

for i= 1 : length(vol_names)
    
    imDir = fullfile(['..\data\cb_3_class\images']);
    pxDir = fullfile(['..\data\cb_3_class\gt']);
    
    classNames = ["CB", "CP", "background"];
    labelIDs =  [2 1 0];
    
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
    for j= 1:length(pxds_va.Files)
        gt = imread(pxds_name{j});
        
        result = imresize(imread(fullfile(['..\data\cb_3_class\results\' vol_names{i} '_' num2str(j) '.png'])),size(gt));
        
        result2= zeros(size(result));
        result2(result==1)=2;
        result2(result==2)=1;
        
        %         gt2  = 2* ones(size(gt));
        %         gt2(gt==1)=1;
        %
        %         %         eval(j,:) = Evaluate2(gt2,result);
        %
        gt_vol(1:size(gt,1),1:size(gt,2),j) = gt;
        res_vol(1:size(gt,1),1:size(gt,2),j) = result2;
        
%         imshowpair(gt,result2,'montage');
    end
    
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

acc_tot = [];
sen_tot = [];
spec_tot = [];
f1_tot = [];
precision_tot = [];

for i = 1:length(x)
  acc_= x{i}.AccuracyOfSingle';
  sen_= x{i}.Sensitivity'  ;
  spec_= x{i}.Specificity' ;
  prec_ = x{i}.Precision';
  f1_s= x{i}.F1_score' ;
  
  
acc_tot = vertcat(acc_tot, acc_);

sen_tot = vertcat(sen_tot, sen_);

spec_tot = vertcat(spec_tot, spec_);


precision_tot = vertcat(precision_tot,prec_);

f1_tot = vertcat(f1_tot, f1_s);

%%
% [c_matrix,Result,RefereceResult]= confusion.getMatrix(gt_vol_tot,res_vol_tot);
end

mean(acc_tot)
mean(sen_tot)
mean(spec_tot)
mean(precision_tot)
mean(f1_tot)


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
