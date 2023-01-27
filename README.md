# Image Quality Metric Index Finder
This repository contains a Python script that can be used to find the corresponding indexes between slices of two 3D images(volumes) based on a image quality metric value such as SSIM,PSNR,HFEN and NRMSE

### Usage
1. Clone the repository to your local machine.
   ```sh
   git clone https://github.com/sthamamta/Image_metric.git
   ```
2. Ensure that you have Python 3 and the necessary dependencies installed. These include numpy, opencv-python, scipy,and nibabel.
3. Run the script using the following command, replacing the placeholder values with the appropriate values for your use case.
```sh
   python find_index.py --hr_path PATH_TO_3DARRAY1 --lr_path PATH_TO_3DARRAY2 --subcript INDEX_DICT_SAVE_NAME --metric METRIC VALUE
   ```
 where METRIC_NAME can be one of ssim, psnr, hfen, or nrmse.

4. The script will output the corresponding indexes between the two images based on the specified image quality metric save it in a dictionary.

## Additional information
The script is designed to work with 3D images in the NIFTI format to create a training pair of 2D image slice between two 3d array(volume).

