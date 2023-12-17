function createNifti(dataName, dataVolume, volumeInfo)

niftiwrite(dataVolume, [dataName(1:end-4) '.nii']);

convertToAmiraCompatibleNifti([dataName(1:end-4) '.nii'],volumeInfo);

end