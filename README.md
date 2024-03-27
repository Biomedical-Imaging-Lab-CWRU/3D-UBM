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
        <li><a href="#ciliary-body-segmentation">Ciliary Body Segmentation</a></li>
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

## Preprocessing
<em>preprocessing</em> contains files for pre-processing .cin and .log files, converting them to common formats, and align them using transformation diffusion approach.

**To read and convert file**
 <ol>
 <li> Create a data folder.</li>
 <li> Put *.cin and associated *.log file under data/folder_name</li>
 <li> Update dataDir to data/folder_name, cinName to *.cin, and logName to *.log</li>
 <li> run demo_read_cin_data_log_save_as_movie_nii_png.m</li>
 <li> run demo_stack_alignment.m</li>
 </ol>

## SDV-GAN
## 3D Ultrasound Biomicroscopy (3D-UBM) Image Enhancement using Spatially varying Deconvolution GAN (SDV-GAN)

This project focuses on enhancing 3D Ultrasound Biomicroscopy (UBM) images using a custom Generative Adversarial Network (GAN) architecture equipped with residual blocks and attention gates.

### Overview

Ultrasound Biomicroscopy (UBM) is a non-invasive imaging technique used for high-resolution imaging of the eye's anterior segment. However, UBM images often suffer from blurring due to varying point-spread-function of ultrasound systems. This project aims to enhance the quality of UBM images using deep learning techniques.

### Features

- **SDV-GAN Architecture**: Utilizes the Pix2Pix architecture, which learns a mapping from an input image (noisy UBM image) to an output image (enhanced UBM image).
- **Residual Blocks**: Integrates residual blocks into the generator model to facilitate better gradient flow and faster convergence during training.
- **Attention Gates**: Incorporates attention gates into the decoder blocks of the generator to selectively focus on relevant image features, improving the quality of the generated images.
- **Training and Evaluation**: Provides scripts for training the model using UBM image datasets and evaluating the performance using evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Multi-Scale SSIM (MS-SSIM).
- **Visualization**: Generates plots showcasing the model's performance over training epochs and saves the best-performing model for future use.

### Python Files

- **deconv_GAN_train_XXX.py**: Script for training the SDV-GAN model under various network conditions using UBM image datasets.
- **deconv_GAN_test_XXX.py**: Script for evaluating the trained model on a separate validation dataset and computing evaluation metrics.
- **evaluation.ipynb**: Contains functions for evaluating the model performance on test set.

### Usage

1. **Training**: Run `deconv_GAN_train_XXX.py` script to train the SDV-GAN model using the provided UBM image dataset. The code asssumes dataset and saved models are stored in a specific directory. Please check code for details.
    ```bash
    python deconv_GAN_train_XXX.py
    ```
2. **Testing**: Run `deconv_GAN_test_XXX.py` script to evaluate the trained model on a separate validation dataset and compute evaluation metrics.
    ```bash
    python deconv_GAN_test_XXX.py
    ```

### Dependencies

- Python 3.9.18
- TensorFlow 2.6
- NumPy
- scikit-image
- Matplotlib

### Dataset

The UBM image dataset used for training and validation should be organized in a structured format where each image is paired with its corresponding enhanced version. Please see the files in our lab server.

### References

- Paper: [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- Paper: [3D ultrasound biomicroscopy image enhancement using generative adversarial networks]
- Documentation: TensorFlow, NumPy, scikit-image


## Ciliary Body Segmentation

### Overview:
The MATLAB scripts provided in this folder are designed for training and evaluating semantic segmentation models using the DeepLabv3+ architecture. These scripts are intended for binary and multi-class segmentation tasks on 3D-UBM images. Please read the [paper](https://doi.org/10.1167/tvst.11.10.3) first.

### List of Scripts:
1. `deeplab_cb_2_class_XXX_YYY_ZZZ.m`: This script trains a DeepLabv3+ model for binary semantic segmentation on 3D-UBM images, segmenting images into two classes: "CB" (ciliary body) and "background". XXX indicates enface or radial. YYY indicates 2D or 2.5D. ZZZ indicates dice loss or cross-entropy loss.
   
2. `deeplab_cb_3_class_XXX_YYY_ZZZ.m`: This script trains a DeepLabv3+ model for multi-class semantic segmentation on medical images, segmenting images into three classes: "CB" (ciliary muscle), "CP" (ciliary processes), and "background".

3. `evaluation_2_class_*.m` and `evaluation_3_class_*.m`: This script evaluates the performance of the trained segmentation models. It computes various performance metrics such as Accuracy, Sensitivity, Specificity, Precision, and F1-score on a per-volume basis.

## Usage:
1. **Data Preparation:**
   - Organize your image and ground truth (GT) data into appropriate directories.
   - Ensure that images and GT data are correctly named and correspond to each other.

2. **Training:**
   - Update the directories for images and GT data in the training scripts (`deeplab_cb_2_class_*.m` and `deeplab_cb_3_class_*.m`).
   - Customize the network architecture, training options, and parameters as needed.
   - Execute the training scripts in MATLAB environment.

3. **Evaluation:**
   - Update the directories for images, GT data, and segmentation results in the evaluation script (`evaluation_*.m`).
   - Run the evaluation script to compute performance metrics for the trained models.

## Additional Notes:
- Ensure that necessary MATLAB toolboxes (e.g., Image Processing Toolbox, Deep Learning Toolbox) are installed and configured.
- GPU support is utilized for faster training; make sure a compatible GPU and CUDA toolkit are available.
- Adjust paths, network architecture, and training parameters according to your dataset and requirements.
- Customize evaluation metrics or add additional evaluation methods as needed.

## Example Workflow:
1. Train the desired segmentation model using `deeplab_cb_2_class.m` or `deeplab_cb_3_class.m`.
2. Evaluate the trained model using `evaluation_2_class_enface.m` or `evaluation_3_class_enface.m`.
3. Visualize results of all files using `visualization_images*.m`

## Acknowledgements:
These scripts leverage MATLAB's Image Processing Toolbox and Deep Learning Toolbox for image processing and deep learning tasks. The DeepLabv3+ architecture is implemented using MATLAB's built-in functions. The code also requires [natsortfiles](https://www.mathworks.com/matlabcentral/fileexchange/47434-natural-order-filename-sort) toolbox for natural sorting.


<p align="right">(<a href="#readme-top">back to top</a>)</p>