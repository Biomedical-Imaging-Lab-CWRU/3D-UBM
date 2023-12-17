% Function to read the log file and extract spacing values
function [xSpacing, ySpacing, zSpacing] = readLogFile(logFileName)
logFile = fileread(logFileName);
x_spacing_end = regexp(logFile, 'x_spacing = ','end');
y_spacing_start = regexp(logFile, 'y_spacing = ');
y_spacing_end = regexp(logFile, 'y_spacing = ','end');
z_spacing_start = regexp(logFile, 'z_spacing = ');
z_spacing_end = regexp(logFile, 'z_spacing = ','end');
comment_start = regexp(logFile, 'comment = ');
xSpacing = str2num(logFile(x_spacing_end:y_spacing_start-1));
ySpacing = str2num(logFile(y_spacing_end:z_spacing_start-1));
zSpacing = str2num(logFile(z_spacing_end:comment_start-1));
disp('Finish reading log file');
end
