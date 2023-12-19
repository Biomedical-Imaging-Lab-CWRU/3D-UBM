# 3D-UBM
This repository contains code for the 3D-UBM system

### Prerequisites
1. MATLAB (tested on R2023a)
2. Gerardus toolbox (https://github.com/vigente/gerardus/tree/master/matlab)

### Installation
1. Install MATLAB
2. Clone/download the gerardus repository. Add ./matlab to MATLAB search path. This toolbox is required for stack alignment.

<!-- USAGE EXAMPLES -->
## Usage
1. To read .cin and .log files and convert them into .nii or mp4 or a stack of .png, run
   ```sh
   preprocessing/demo_read_cin_data_log_save_as_movie_nii_png.m
   ```
2. To convert a MATLAB-created .nii file to Amira-Avizo compatible .nii, run
   ```sh
   preprocessing/demo_nii_volume_amira_compatible.m
   ```
3. To perform stack alignment, run
   ```sh
   preprocessing/demo_stack_alignment.m
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Ahmed Tahseen Minhaz - axm1287@case.edu

Project Link: [https://github.com/tahseenb/3D-UBM](https://github.com/tahseenb/3D-UBM)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
