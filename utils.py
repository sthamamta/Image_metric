
import nibabel as nib
import numpy as np

from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from image_quality_assessment import PSNR,SSIM,NRMSELoss,hfen_error


def get_array(path):
    array = load_array(path)

    #normalize along last axis
    for i in range(array.shape[2]):
        array[:,:,i] = (array[:,:,i]- array[:,:,i].min())/(array[:,:,i].max()-array[:,:,i].min())
        array[:,:,i] = array[:,:,i]*255.

    return array


def load_array(path):
    import nibabel as nib
    img = nib.load(path)
    affine_mat=img.affine
    hdr=img.header
    data = img.get_fdata()
    data_norm = data
    return data_norm 

def mse(hr_array, lr_array):
    return (np.square(np.subtract(hr_array,lr_array)).mean())

def pad_zeros_around(arr,factor,shape):
    y,x = shape
    rows_pad =(y-(y//factor))//2
    cols_pad =(x-(x//factor))//2
    return np.pad(arr, [(rows_pad, rows_pad), (cols_pad, cols_pad)], mode='constant',constant_values=0)

def pad_image_kspace(data,factor=2):  #function for cropping and/or padding the image in kspace
    F = np.fft.fft2(data)
    fshift = np.fft.fftshift(F)
    shape = (720,512)
    fshift= pad_zeros_around(fshift,factor=2,shape=shape)
    img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_reco = np.abs(img_reco_cropped )
    img_reco = (img_reco-img_reco.min())/(img_reco.max()-img_reco.min())
    img_reco = img_reco *255
    return img_reco


# function to handle choosing min or max depending on the metric
def find_index(hr_array, lr_array, metric='mse'):
    if metric in ['psnr', 'ssim']:
        index_match = find_index_max(hr_array, lr_array, metric)
    elif metric in ['mse', 'nrmse','hfen']:
        index_match = find_index_min(hr_array, lr_array, metric)
    return index_match


def find_index_min(hr_array, lr_array, metric='hfen'):
    hr_x,hr_y,hr_z = hr_array.shape
    lr_x, lr_y, lr_z = lr_array.shape
    index_match = {}  #lr_index: hr_index
    count = 0
    for i in range(lr_z):
        best_metric = 1436547523846
        best_index = 6000
        lr_array_i = pad_image_kspace(lr_array[:,:,i])
        # print('lr range',lr_array.min(), lr_array.max())
        # print('hr range',hr_array.min(), hr_array.max())
        for j in range(hr_z):
            metric_value = measure_metric(hr_array[:,:,j],lr_array_i, metric=metric)
            count += 1
            print("metric value", metric_value)
            if metric_value < best_metric:
                best_metric = metric_value
                best_index = j
                index_match[i]= best_index
    print('count', count)
    return index_match
    

def find_index_max(hr_array, lr_array, metric='psnr'):
    hr_x,hr_y,hr_z = hr_array.shape
    lr_x, lr_y, lr_z = lr_array.shape

    index_match = {}  #lr_index: hr_index
    count = 0
    for i in range(lr_z):
        best_metric = 0
        best_index = 6000
        lr_array_i = pad_image_kspace(lr_array[:,:,i])
        # print('lr range',lr_array.min(), lr_array.max())
        # print('hr range',hr_array.min(), hr_array.max())
        for j in range(hr_z):
            metric_value = measure_metric(hr_array[:,:,j],lr_array_i, metric=metric)
            count += 1
            if metric_value > best_metric:
                best_metric = metric_value
                best_index = j
                index_match[i]= best_index
    print('count', count)
    return index_match


def pad_lr_array(lr_array):
    for i in range(lr_array.shape[2]):
        lr_array[:,:,i] = pad_image_kspace(lr_array[:,:,i])
    return lr_array



# https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/imgproc.py
def image2tensor(image: np.ndarray, range_norm: bool=False, half: bool=False) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch
    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type
    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch
    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=True, half=False)
    """
    # Convert image data type to Tensor data type
    tensor = F.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool=False, half: bool=False,color:bool=False) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type
    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.
    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV
    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=False, half=False)
    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    if color:
      image = tensor.detach().squeeze().permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    else:
        image = tensor.detach().squeeze().mul(255).clamp(0, 255).cpu().numpy().astype("uint8") 

    return image


def measure_metric(lr_arr, hr_arr, metric='mse'):
    lr_tensor = image2tensor(lr_arr/255.).unsqueeze(dim=0)
    hr_tensor = image2tensor(hr_arr/255.).unsqueeze(dim=0)
    if metric == 'mse':
        mse = torch.nn.MSELoss()
        mse_value = mse(lr_tensor, hr_tensor)
        return mse_value.item()
    elif metric == 'psnr':
        psnr = PSNR()
        psnr = psnr.to(device='cpu', memory_format=torch.channels_last, non_blocking=True)
        psnr_value = psnr(lr_tensor, hr_tensor)
        return psnr_value.item()
    elif metric == 'ssim':
        ssim = SSIM()
        ssim = ssim.to(device='cpu', memory_format=torch.channels_last, non_blocking=True)
        ssim_value = ssim(lr_tensor, hr_tensor)
        return ssim_value.item() 
    elif metric == 'hfen':
        hfen_value = hfen_error(lr_tensor,hr_tensor)
        print(hfen_value)
        return hfen_value
    elif metric == 'nrmse':
        nrmse = NRMSELoss().to(device='cpu')
        lr_tensor = lr_tensor.to('cpu')
        hr_tensor = hr_tensor.to('cpu')
        nrmse_value =  nrmse(lr_tensor,hr_tensor)
        return nrmse_value.item()
    else:
       print("metric not implemented")
        
