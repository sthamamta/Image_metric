
from typing import Any

import cv2
# import numpy as np

from torchvision.transforms import functional as F
import pickle
from utils import *
import argparse

# F1_160 , f1_160 , F2_145,f2_145, F4_149,f4_149,  F5_153,f5_153

parser = argparse.ArgumentParser()
parser.add_argument('--hr_path', help="path for first array", type=str, required=False, default='OneDrive/AD_P522R_F5_153/MRI_25um/mag_sos_wn.nii')
parser.add_argument('--lr_path', help="path for second array", type=str, required=False, default='OneDrive/AD_P522R_F5_153/MRI_50um/mag_sos_wn.nii')
parser.add_argument('--subscript', help="suscript to form save directory", type=str, required=False, default='f5_153')
parser.add_argument('--metric',choices=['psnr', 'ssim', 'mse','hfen'],help='image quality metric',default='psnr')
args = parser.parse_args()

# read the 3D image array
data_hr = get_array(args.hr_path)
data_lr = get_array(args.lr_path)


print(data_hr.shape)
print(data_hr[:,:,1].min(), data_hr[:,:,1].max())

print(data_lr.shape)
print(data_lr[:,:,1].min(), data_lr[:,:,1].max())


index_dict = find_index(data_hr, data_lr,args.metric)
print(index_dict)

save_name = '{}_{}_match_index.pkl'.format(args.subscript,args.metric)
a_file = open(save_name, "wb")
pickle.dump(index_dict, a_file)
a_file.close()


annotation_dict = {}
for (key,value) in index_dict.items():
    new_key = 'lr_{}_z_{}.png'. format(args.subscript, key)
    new_value = 'hr_{}_z_{}.png'.format(args.subscript,value)
    annotation_dict[new_key] = new_value

print(annotation_dict)
# saving the matched index dictionary
save_name = '{}_{}_annotation.pkl'.format(args.subscript,args.metric)
a_file = open(save_name, "wb")
pickle.dump(annotation_dict, a_file)
a_file.close()

