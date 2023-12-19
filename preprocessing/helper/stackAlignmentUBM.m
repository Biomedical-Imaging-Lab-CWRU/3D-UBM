function stackAlignmentUBM(PATIENT, dataDir)
    imgDir = fullfile(dataDir, PATIENT, 'PNG'); % the code expects there to be a PNG folder where .png images are saved
    saveDir = fullfile(dataDir, PATIENT); %

    optTransformix.AutoDefaultPixelValue = true;

    fileh = dir(fullfile(imgDir, '*.png'));
    N = length(fileh);
    ICENT = round(N/2);

    % Rigid registration from 1 to ICENT
    for I = 1:(ICENT-1)
        [~, filename0] = fileparts(fileh(I+1).name);
        imh0 = scimat_load(fullfile(imgDir, [filename0, '.png']));
        [~, filename] = fileparts(fileh(I).name);
        imh = scimat_load(fullfile(imgDir, [filename, '.png']));
        th2h(I) = regmatchedfilt(imh0, imh, 0);
    end

    % Load reference slice
    imref = scimat_load(fullfile(imgDir, fileh(ICENT).name));
    th2h(ICENT) = th2h(ICENT-1);
    th2h(ICENT).TransformParameters = [0 0 0];
    th2h(ICENT).Origin = scimat_index2world([1 1], imref);
    th2h(ICENT).Size = [size(imref.data, 2) size(imref.data, 1)];
    th2h(ICENT).Spacing([2 1]) = [imref.axis.spacing];

    % Rigid registration from ICENT+1 to N
    for I = (ICENT+1):N
        imh0 = scimat_load(fullfile(imgDir, fileh(I-1).name));
        imh = scimat_load(fullfile(imgDir, fileh(I).name));
        th2h(I) = regmatchedfilt(imh0, imh, 0);
    end

    disp('Initial transform done');

    % Accumulate registrations down the stack on both sides of the reference slice
    th2hAcc(ICENT) = th2h(ICENT);

    % Accumulate transforms 120->(121), 119->(120->121), ...
    for I = (ICENT-1):-1:1
        imh0 = scimat_load(fullfile(imgDir, fileh(I).name));
        th2hAcc(I) = elastix_compose_afftransf(th2hAcc(I+1), th2h(I));
        th2hAcc(I).Origin = th2h(ICENT).Origin;
        th2hAcc(I).Spacing = th2h(ICENT).Spacing;
        th2hAcc(I).Size = th2h(ICENT).Size;
        imh(I) = transformix(th2hAcc(I), imh0, optTransformix);
    end

    % Accumulate transforms (121)<-122, (121<-122)<-123,...
    for I = (ICENT+1):N
        imh0 = scimat_load(fullfile(imgDir, fileh(I).name));
        th2hAcc(I) = elastix_compose_afftransf(th2hAcc(I-1), th2h(I));
        th2hAcc(I).Origin = th2h(ICENT).Origin;
        th2hAcc(I).Spacing = th2h(ICENT).Spacing;
        th2hAcc(I).Size = th2h(ICENT).Size;
        imh(I) = transformix(th2hAcc(I), imh0, optTransformix);
    end

    disp('Accumulate registrations down the stack on both sides of the reference slice done');

    % Intra-3D-UBM correction
    opts.Grayscale = true;
    opts.AutoDefaultPixelValue = true;
    for I = 1:N
        imh(I) = scimat_load(fullfile(imgDir, fileh(I).name));
        imh(I) = transformix(th2h(I), imh(I), opts);
    end

    % Rigid transform diffusion registration of the 3D-UBM volume
    clear optReg optDiff;
    optReg.Angle = 0;
    optReg.verbose = true;
    optDiff.MaxIter = 200;
    [taux, ~, ~, imout] = transfdiffreg('regmatchedfilt', imh, optReg, optDiff);

    % Concatenate transforms
    th2h = elastix_cat(taux, th2h);

    disp('TD done');

    % Save results
    for k = 1:N
        vol_out(:, :, k) = imout(k).data';
    end

    niftiwrite(vol_out, fullfile(saveDir, [PATIENT, '.raw.aligned.nii'])); % file saved once
    rawVolumeInfo = niftiinfo(fullfile(saveDir, [PATIENT, '.nii'])); % info borrowed from raw/unaligned data
    convertToAmiraCompatibleNifti(fullfile(saveDir, [PATIENT, '.raw.aligned.nii']), rawVolumeInfo); %file overwritten with new info to ensure compatibility with Amira

    disp('Processing completed.');
end
