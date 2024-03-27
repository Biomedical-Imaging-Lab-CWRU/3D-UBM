%%evaluate performance

clc; clear all; clear all;

vol_names = {'2018-032', '2019-047', '2019-12-14-F58', '2019-12-14-M67', 'Eye1', 'Eye2', 'Eye3'};


for i= 2 %1 : length(vol_names)
    
    imDir = ['C:\Users\axm1287\Desktop\Desktop_09x20x2021\lab_meeting_code_ciliary_body\data\cb_3_class_radial\images'];
    pxDir = ['C:\Users\axm1287\Desktop\Desktop_09x20x2021\lab_meeting_code_ciliary_body\data\cb_3_class_radial\gt'];
    
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
    imds_name = imds_va.Files;
    for j= 1:length(pxds_va.Files)
        
        
        gt = imread(pxds_name{j});
        img = imread(imds_name{j});
        
        result = imresize(imread(['C:\Users\axm1287\Desktop\Desktop_09x20x2021\lab_meeting_code_ciliary_body\data\cb_3_class_radial\results\' vol_names{i} '_' num2str(j) '.png']),size(gt));
        
        %        result2= zeros(size(result));
        %        result2(result==1)=1;
        
        gt2  = 2* ones(size(gt));
        gt2(gt==1)=1;
        
        %         eval(j,:) = Evaluate2(gt2,result);
        
%         gt2 = abs(gt2-2);
%         result = abs(double(result)-2);
        
        
        gt_vol(:,:,j) = gt2;
        res_vol(:,:,j) = result;
        img_vol(:,:,j) = img;
        
    end
    
    niftiwrite(img_vol,['C:\Users\axm1287\Desktop\Desktop_09x20x2021\lab_meeting_code_ciliary_body\code\visualization\2-class\' vol_names{1,i} '-matlab.nii']);
    
    niftiwrite(gt_vol,['C:\Users\axm1287\Desktop\Desktop_09x20x2021\lab_meeting_code_ciliary_body\code\visualization\2-class\' vol_names{1,i} '-matlab.labels.nii']);
    
    niftiwrite(res_vol,['C:\Users\axm1287\Desktop\Desktop_09x20x2021\lab_meeting_code_ciliary_body\code\visualization\2-class\' vol_names{1,i} '-matlab.labels2.nii']);
    
    
end