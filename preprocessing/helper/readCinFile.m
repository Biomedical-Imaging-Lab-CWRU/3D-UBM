% Function to read the .cin file and return the data
function data = readCinFile(dataName)
%% caution: do not use fileread function
fileID = fopen(dataName);
data = fread(fileID);
data = uint8(data)';
disp('finish reading cin file')
end

% Function to extract a numerical value from a log file based on a key
function value = readValue(text, key)
keyPattern = [key ' = '];
startIdx = regexp(text, keyPattern, 'end');
endIdx = regexp(text, [keyPattern '\S*']);
value = str2num(text(startIdx:endIdx-1));
end