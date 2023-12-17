% Function to check if a frame and header combination is valid
function valid = checkFrameAndHeader(iFrame, iHeader, rowBytes, numRows, cinHeaderBytes, dataSize)
    valid = false;
    
    % Check if the combination satisfies the condition
    if iFrame * (rowBytes * numRows + iHeader) + cinHeaderBytes == dataSize
        valid = true;
    end
end