% Function to find the number of frames and header bytes in the .cin file
function [numFrames, frameHeaderBytes] = findNumFramesAndHeaderBytes(data, rowBytes, numRows, cinHeaderBytes, dataSize)
numFrames = 0;
frameHeaderBytes = 0;

% Iterate over possible frame and header combinations
for iFrame = 1:1505
    for iHeader = 500:900
        if checkFrameAndHeader(iFrame, iHeader, rowBytes, numRows, cinHeaderBytes, dataSize)
            numFrames = iFrame;
            frameHeaderBytes = iHeader;
            break
        end
        if numFrames > 0
            break
        end
    end
end
end