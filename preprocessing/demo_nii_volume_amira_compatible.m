clc;clear all;close all;

%script to make a single volume Amira compatible
rawVolumeInfo = niftiinfo(fullfile(saveDir, [PATIENT, '.nii'])); % info borrowed from raw/unaligned data
convertToAmiraCompatibleNifti(fullfile(saveDir, [PATIENT, '.raw.aligned.nii']), rawVolumeInfo); %file overwritten with new info to ensure compatibility with Amira