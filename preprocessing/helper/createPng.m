function createPng(dataName, saveDir, dataVolume)
    
    if ~exist(fullfile(saveDir,'PNG'), 'dir')
       mkdir(fullfile(saveDir,'PNG'))
    end
     
    for i = 1:size(dataVolume,3)
        img = dataVolume(:,:,i);
        imwrite(img, fullfile(saveDir,'PNG',[dataName(1:end-4) sprintf('%03d',i-1) '.png']));
    end

end
