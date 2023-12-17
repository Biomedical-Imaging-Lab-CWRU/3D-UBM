function convertToAmiraCompatibleNifti(inputFilePath,volumeInfo)

    outputFilePath = inputFilePath;
    % Convert a NIfTI volume to Amira compatible format.

    % Read the NIfTI volume
    vol_info = niftiinfo(inputFilePath);
    vol_data = niftiread(vol_info);

    % Create a copy of the NIfTI info for modification
    vol_info2 = vol_info;

    % Update the transformation information
    vol_info2.MultiplicativeScaling = 0;
    vol_info2.TransformName = 'Qform';
    vol_info2.Transform.T = eye(4);
    vol_info2.Transform.T(1, 1) = volumeInfo.xSpacing;
    vol_info2.Transform.T(2, 2) = volumeInfo.ySpacing;
    vol_info2.Transform.T(3, 3) = volumeInfo.zSpacing;

    % Update pixel dimensions
    vol_info2.PixelDimensions(1) = volumeInfo.xSpacing;
    vol_info2.PixelDimensions(2) = volumeInfo.ySpacing;
    vol_info2.PixelDimensions(3) = volumeInfo.zSpacing;

    % Set irrelevant fields to 0
    vol_info2.raw.dim(5:end) = 0;
    vol_info2.raw.pixdim(5:end) = 0;
    vol_info2.raw.scl_slope = 0;
    vol_info2.raw.qform_code = 1;
    vol_info2.raw.sform_code = 0;
    vol_info2.raw.quatern_b = 0;
    vol_info2.raw.quatern_c = 0;
    vol_info2.raw.quatern_d = 0;
    vol_info2.raw.qoffset_x = 0;
    vol_info2.raw.qoffset_y = 0;
    vol_info2.raw.qoffset_z = 0;
    vol_info2.raw.srow_x = [0, 0, 0, 0];
    vol_info2.raw.srow_y = [0, 0, 0, 0];
    vol_info2.raw.srow_z = [0, 0, 0, 0];

    % Write the modified NIfTI file
    niftiwrite(vol_data, outputFilePath, vol_info2);

    disp(['Amira compatible NIfTI file created: ' outputFilePath]);
end
