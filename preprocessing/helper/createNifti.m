function createNifti(dataName, dataVolume, volumeInfo)


dataVolume = permute(dataVolume,[2 1 3]);
niftiwrite(dataVolume, [dataName(1:end-4) '.nii']);

convertToAmiraCompatibleNifti([dataName(1:end-4) '.nii'],volumeInfo);

end