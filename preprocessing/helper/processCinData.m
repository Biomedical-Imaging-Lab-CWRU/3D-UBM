% Function to process the .cin data and return the data volume
function dataVolumeCropped = processCinData(data, rowBytes, numRows, cinHeaderBytes, croppedRowBytes)
    dataUse = data;
    dataSize = length(dataUse);

    % Find the number of frames and header bytes
    [numFrames, frameHeaderBytes] = findNumFramesAndHeaderBytes(data, rowBytes, numRows, cinHeaderBytes, dataSize);

    % Extract data after the header
    dataAfterHead = data(1, cinHeaderBytes + 1:end);
    dataLength = rowBytes * numRows;
    dataVolume = zeros(rowBytes, numRows, numFrames);

    % Iterate over frames and reconstruct the data volume
    for i = 1:numFrames
        oneFrame = (i - 1) * (dataLength + frameHeaderBytes);
        imageIndex = oneFrame + frameHeaderBytes;
        frame = dataAfterHead(imageIndex + 1 : imageIndex + dataLength);
        frame = reshape(frame, [rowBytes numRows]);
        dataVolume(:, :, i) = frame;

        % Crop the frame
        croppedFrame = frame(401:1500, :);
        dataVolumeCropped(:, :, i) = croppedFrame;

    end
end
