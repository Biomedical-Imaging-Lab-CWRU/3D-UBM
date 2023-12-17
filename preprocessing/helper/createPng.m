function createPng(dataName, dataVolume)
    directory = fullfile(dataName(1:end-4),'PNG');

    if ~exist(directory, 'dir')
       mkdir(directory)
    end
     
    for i = 1:size(dataVolume,3)
        img = vol(:,:,i);
        imwrite(img, [directory '\' dataName(1:end-4) sprintf('%03d',i-1) '.png'])
    end

end
