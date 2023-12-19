clear all;clc;close all;

% dependencies: https://github.com/vigente/gerardus/tree/master/matlab
% follow instructions for installing the dependencies

% toolbox required:
% https://github.com/vigente/gerardus/tree/master/matlab/ElastixToolbox
% https://github.com/vigente/gerardus/tree/master/matlab/FileFormatToolbox
% https://github.com/vigente/gerardus/tree/master/matlab/FiltersToolbox
% https://github.com/vigente/gerardus/tree/master/matlab/ManifoldToolbox
% https://github.com/vigente/gerardus/tree/master/matlab/RegistrationToolbox
% https://github.com/vigente/gerardus/tree/master/matlab/ThirdPartyToolbox

% code requires dataDir/PATIENT/PNG folder exists and contains series of
% UBM images.

dataDir = 'Z:\3DUltrasound\3D-UBM Eye Volumes Master\3D-UBM Eye Volumes Master\patient eyes\Dr Orge Backup\';
PATIENT = '7903689_OS_753-141220-105435';
stackAlignmentUBM(PATIENT, dataDir);