function createVideo(data_name, V)
    % Create a video from the data array V and save it as an MPEG-4 file.

    % Define the output video file name
    outputVideo = VideoWriter([data_name(1:end-4) '.mp4'], 'MPEG-4');

    % Open the video file for writing
    open(outputVideo);

    % Loop through the frames in V and write each frame to the video file
    for i = 1:size(V, 3)
        writeVideo(outputVideo, uint8(V(:, :, i)));
    end

    % Close the video file
    close(outputVideo);

    disp(['Video file created: ' outputVideo.Filename]);
end