# 3D-UBM
This repository contains code for the 3D-UBM system. Please cite the following papers when using the repository.

<ol>
<li> Ahmed Tahseen Minhaz, Duriye Damla Sevgi, Sunwoo Kwak, Alvin Kim, Hao Wu, Richard W. Helms, Mahdi Bayat, David L. Wilson, Faruk H. Orge; Deep Learning Segmentation, Visualization, and Automated 3D Assessment of Ciliary Body in 3D Ultrasound Biomicroscopy Images. Trans. Vis. Sci. Tech. 2022;11(10):3. https://doi.org/10.1167/tvst.11.10.3.</li>

<li> Ahmed Tahseen Minhaz, Mahdi Bayat, Duriye Damla Sevgi, Haoxing Chen, Sunwoo Kwak, Richard Helms, Faruk Orge, David L. Wilson, "Deconvolution of ultrasound biomicroscopy images using generative adversarial networks to visualize and evaluate localization of ocular structures," Proc. SPIE 11602, Medical Imaging 2021: Ultrasonic Imaging and Tomography, 116020H (25 February 2021); https://doi.org/10.1117/12.2582128</li>

<li> Richard W. Helms, Ahmed Tahseen Minhaz, David L. Wilson, Faruk H. Örge; Clinical 3D Imaging of the Anterior Segment With Ultrasound Biomicroscopy. Trans. Vis. Sci. Tech. 2021;10(3):11. https://doi.org/10.1167/tvst.10.3.11.</li>
</ol>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<em>preprocessing</em> contains files for pre-processing .cin and .log files, converting them to common formats, and align them using transformation diffusion approach.

**To read and convert file**
 <ol>
 <li> Create a data folder.</li>
 <li> Put *.cin and associated *.log file under data/folder_name</li>
 <li> Update dataDir to data/folder_name, cinName to *.cin, and logName to *.log</li>
 <li> run demo_read_cin_data_log_save_as_movie_nii_png.m</li>
 <li> run demo_stack_alignment.m</li>
 </ol>
