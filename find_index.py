
from typing import Any

import cv2
import numpy as np

from torchvision.transforms import functional as F
import pickle
from utils import *

# F1_160 , f1_160 , F2_145,f2_145, F4_149,f4_149,  F5_153,f5_153

datapath_hr = 'OneDrive/AD_P522R_F5_153/MRI_25um/mag_sos_wn.nii'
datapath_lr = 'OneDrive/AD_P522R_F5_153/MRI_50um/mag_sos_wn.nii'

subscript = 'f5_153'

#define image metric 
metric = 'mse'

# read the 3D image array
data_hr = get_array(datapath_hr)
data_lr = get_array(datapath_lr)


print(data_hr.shape)
print(data_hr[:,:,1].min(), data_hr[:,:,1].max())

print(data_lr.shape)
print(data_lr[:,:,1].min(), data_lr[:,:,1].max())


index_dict = find_index(data_hr, data_lr,metric)
print(index_dict)

save_name = '{}_{}_match_index.pkl'.format(subscript,metric)
a_file = open(save_name, "wb")
pickle.dump(index_dict, a_file)
a_file.close()


annotation_dict = {}
for (key,value) in index_dict.items():
    new_key = 'lr_{}_z_{}.png'. format(subscript, key)
    new_value = 'hr_{}_z_{}.png'.format(subscript,value)
    annotation_dict[new_key] = new_value

print(annotation_dict)
save_name = '{}_{}_annotation.pkl'.format(subscript,metric)
a_file = open(save_name, "wb")
pickle.dump(annotation_dict, a_file)
a_file.close()

