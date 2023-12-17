classdef CinFileReader
    properties
        DataName
        LogFileName
        Data
        RowBytes
        NumRows
        CinHeaderBytes
        CroppedRowBytes
        IsoXyzResolution
        XSpacing
        YSpacing
        ZSpacing
        DataUse
        CinSizeBytes
        NumFrames
        FrameHeaderBytes
        DataAfterHead
        DataLength
        DataVolume
        DataVolumeCropped
    end
    
    methods
        function obj = CinFileReader(dataName, logFileName)
            obj.DataName = dataName;
            obj.LogFileName = logFileName;
            obj.RowBytes = 2048;
            obj.NumRows = 384;
            obj.CinHeaderBytes = 8;
            obj.CroppedRowBytes = 1100;
            obj.IsoXyzResolution = 0.025;
        end
        
        function readCinFile(obj)
            obj.Data = fileread(obj.DataName);
            disp('Finish reading .cin file');
        end
        
        function readLogFile(obj)
            logFile = fileread(obj.LogFileName);
            obj.XSpacing = obj.readValue(logFile, 'x_spacing');
            obj.YSpacing = obj.readValue(logFile, 'y_spacing');
            obj.ZSpacing = obj.readValue(logFile, 'z_spacing');
            disp('Finish reading log file');
        end
        
        function value = readValue(~, text, key)
            keyPattern = [key ' = '];
            startIdx = regexp(text, keyPattern, 'end');
            endIdx = regexp(text, [keyPattern '\S*']);
            value = str2num(text(startIdx:endIdx-1));
        end
        
        function processCinData(obj)
            obj.DataUse = double(uint8(obj.Data));
            obj.CinSizeBytes = length(obj.DataUse);
            
            obj.findNumFramesAndHeaderBytes();
            

            size(obj.Data)
            obj.DataAfterHead = obj.Data(1, obj.CinHeaderBytes + 1:end);
            obj.DataLength = obj.RowBytes * obj.NumRows;
            obj.DataVolume = zeros(obj.RowBytes, obj.NumRows, obj.NumFrames);
            obj.DataVolumeCropped = zeros(obj.CroppedRowBytes, obj.NumRows, obj.NumFrames);
            
            for i = 1:obj.NumFrames
                oneFrame = (i - 1) * (obj.DataLength + obj.FrameHeaderBytes);
                imageIndex = oneFrame + obj.FrameHeaderBytes;
                frame = obj.DataAfterHead(imageIndex + 1 : imageIndex + obj.DataLength);
                frame = reshape(frame, [obj.RowBytes obj.NumRows]);
                croppedFrame = frame(401:1500, :);
                obj.DataVolume(:, :, i) = frame;
                obj.DataVolumeCropped(:, :, i) = croppedFrame;
            end
        end
        
        function findNumFramesAndHeaderBytes(obj)
            obj.NumFrames = 0;
            obj.FrameHeaderBytes = 0;
            for iFrame = 1:1505
                for iHeader = 500:900
                    if obj.checkFrameAndHeader(iFrame, iHeader)
                        return;
                    end
                end
            end
        end
        
        function valid = checkFrameAndHeader(obj, iFrame, iHeader)
            if iFrame * (obj.RowBytes * obj.NumRows + iHeader) + obj.CinHeaderBytes == obj.CinSizeBytes
                obj.NumFrames = iFrame;
                obj.FrameHeaderBytes = iHeader;
                valid = true;
            else
                valid = false;
            end
        end
    end
end
