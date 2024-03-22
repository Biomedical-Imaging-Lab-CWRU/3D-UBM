# 3D-UBM
 This repository contains code for the 3D-UBM system.

 <em>preprocessing</em> contains files for pre-processing .cin and .log files, converting them to common formats, and align them using transformation diffusion approach.

**To read and convert file**
 <ol>
 <li> Create a data folder.</li>
 <li> Put *.cin and associated *.log file under data/folder_name</li>
 <li> Update dataDir to data/folder_name, cinName to *.cin, and logName to *.log</li>
 <li> run demo_read_cin_data_log_save_as_movie_nii_png.m</li>
 <li> run demo_stack_alignment.m</li>
 </ol>
