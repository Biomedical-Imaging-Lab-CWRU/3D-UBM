<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#instructions">Instructions</a>
        <ul>
        <li><a href="#preprocessing">Preprocessing</a></li>
        <li><a href="#sdv-gan">SDV-GAN</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
# 3D-UBM
This repository contains code for the 3D-UBM system. Please cite the following papers when using the repository.

<ol>
<li> Ahmed Tahseen Minhaz, Duriye Damla Sevgi, Sunwoo Kwak, Alvin Kim, Hao Wu, Richard W. Helms, Mahdi Bayat, David L. Wilson, Faruk H. Orge; Deep Learning Segmentation, Visualization, and Automated 3D Assessment of Ciliary Body in 3D Ultrasound Biomicroscopy Images. Trans. Vis. Sci. Tech. 2022;11(10):3. https://doi.org/10.1167/tvst.11.10.3.</li>

<li> Ahmed Tahseen Minhaz, Mahdi Bayat, Duriye Damla Sevgi, Haoxing Chen, Sunwoo Kwak, Richard Helms, Faruk Orge, David L. Wilson, "Deconvolution of ultrasound biomicroscopy images using generative adversarial networks to visualize and evaluate localization of ocular structures," Proc. SPIE 11602, Medical Imaging 2021: Ultrasonic Imaging and Tomography, 116020H (25 February 2021); https://doi.org/10.1117/12.2582128</li>

<li> Richard W. Helms, Ahmed Tahseen Minhaz, David L. Wilson, Faruk H. Örge; Clinical 3D Imaging of the Anterior Segment With Ultrasound Biomicroscopy. Trans. Vis. Sci. Tech. 2021;10(3):11. https://doi.org/10.1167/tvst.10.3.11.</li>
</ol>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Instructions

Download the repository locally and follow instructions listed in README as well as the codes. The data can be accessed from our lab's server.

### Preprocessing
<em>preprocessing</em> contains files for pre-processing .cin and .log files, converting them to common formats, and align them using transformation diffusion approach.

**To read and convert file**
 <ol>
 <li> Create a data folder.</li>
 <li> Put *.cin and associated *.log file under data/folder_name</li>
 <li> Update dataDir to data/folder_name, cinName to *.cin, and logName to *.log</li>
 <li> run demo_read_cin_data_log_save_as_movie_nii_png.m</li>
 <li> run demo_stack_alignment.m</li>
 </ol>

### SDV-GAN
# Ultrasound Biomicroscopy (UBM) Image Enhancement using Pix2Pix GAN with Residual Blocks and Attention Gates

This project focuses on enhancing Ultrasound Biomicroscopy (UBM) images using a Pix2Pix Generative Adversarial Network (GAN) architecture equipped with residual blocks and attention gates.

## Overview

Ultrasound Biomicroscopy (UBM) is a non-invasive imaging technique used for high-resolution imaging of the eye's anterior segment. However, UBM images often suffer from noise and lack of clarity, which can hinder accurate analysis and diagnosis. This project aims to enhance the quality of UBM images using deep learning techniques.

## Features

- **Pix2Pix GAN Architecture**: Utilizes the Pix2Pix architecture, which learns a mapping from an input image (noisy UBM image) to an output image (enhanced UBM image).
- **Residual Blocks**: Integrates residual blocks into the generator model to facilitate better gradient flow and faster convergence during training.
- **Attention Gates**: Incorporates attention gates into the decoder blocks of the generator to selectively focus on relevant image features, improving the quality of the generated images.
- **Training and Evaluation**: Provides scripts for training the model using UBM image datasets and evaluating the performance using evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Multi-Scale SSIM (MS-SSIM).
- **Visualization**: Generates plots showcasing the model's performance over training epochs and saves the best-performing model for future use.

## Python Files

- **train.py**: Script for training the Pix2Pix GAN model using UBM image datasets.
- **test.py**: Script for evaluating the trained model on a separate validation dataset and computing evaluation metrics.
- **utils.py**: Contains utility functions used for data loading, model construction, and evaluation.

## Usage

1. **Training**: Run `train.py` script to train the Pix2Pix GAN model using the provided UBM image dataset.
    ```bash
    python train.py --dataset_path /path/to/train_dataset --val_dataset_path /path/to/val_dataset
    ```
2. **Testing**: Run `test.py` script to evaluate the trained model on a separate validation dataset and compute evaluation metrics.
    ```bash
    python test.py --dataset_path /path/to/test_dataset --model_path /path/to/trained_model
    ```

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-image
- Matplotlib

## Dataset

The UBM image dataset used for training and validation should be organized in a structured format where each image is paired with its corresponding enhanced version.

## References

- Paper: [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- Documentation: TensorFlow, NumPy, scikit-image




<p align="right">(<a href="#readme-top">back to top</a>)</p>