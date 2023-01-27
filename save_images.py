

from typing import Any

import cv2
import numpy as np

from torchvision.transforms import functional as F
import pickle
from utils import *
import os

# F1_160 , f1_160 , F2_145,f2_145, F4_149,f4_149,  F5_153,f5_153

datapath_hr = 'OneDrive/AD_P522R_F4_149/MRI_25um/mag_sos_wn.nii'
datapath_lr = 'OneDrive/AD_P522R_F4_149/MRI_50um/mag_sos_wn.nii'
subscript = 'f4_149'


data_hr = get_array(datapath_hr)
data_lr = get_array(datapath_lr)

save_lr_dir = 'dataset/test_val/'
save_hr_dir = 'dataset/hr_label/'

if not os.path.exists(save_lr_dir):
    os.makedirs(save_lr_dir)

if not os.path.exists(save_hr_dir):
    os.makedirs(save_hr_dir)

for i in range(data_lr.shape[2]):
    print('save lr')
    if i<140: #ignore the last image whose mainly consists of blank dark images
        lr_name = 'lr_{}_z_{}.png'. format(subscript, i)
        image_upsampled = pad_image_kspace(data_lr[:,:,i])
        save_path = os.path.join(save_lr_dir, lr_name)
        cv2.imwrite(save_path, image_upsampled)

for i in range(data_hr.shape[2]):
    hr_name = 'hr_{}_z_{}.png'. format(subscript, i)
    save_path = os.path.join(save_hr_dir, hr_name)
    cv2.imwrite(save_path, data_hr[:,:,i])
