
from typing import Any

import cv2
import numpy as np

from torchvision.transforms import functional as F
import pickle
from utils import *
import pickle
import matplotlib.pyplot as plt
import os

def plot_figure(lr,hr,index_lr,index_hr,save_path):
    plt.figure(figsize = (20,10))

    plt.subplot(121), plt.imshow(lr,cmap='gray')
    plt.title("50 micron  "+str(index_lr),fontsize = 25), plt.xticks([]),plt.yticks([])

    plt.subplot(122), plt.imshow(hr,cmap='gray')
    plt.title("25 micron  "+ str(index_hr),fontsize = 25), plt.xticks([]), plt.yticks([])

    # plt.show()
    plt.savefig(save_path)


datapath_hr = 'OneDrive/AD_P522R_F2_145/MRI_25um/mag_sos_wn.nii'
data_hr = get_array(datapath_hr)

datapath_lr = 'OneDrive/AD_P522R_F2_145/MRI_50um/mag_sos_wn.nii'
data_lr = get_array(datapath_lr)
# annotation_path =  'matched_index.pkl'
annotation_path = 'mse_annotation/f2_145_mse_match_index.pkl'

with open(annotation_path,'rb') as f:
    annotation_dictionary = pickle.load(f)

save_folder = 'plots/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

    
for i in range(data_lr.shape[2]):
    lr_index = i
    hr_index = annotation_dictionary[lr_index]

    hr_image = data_hr[:,:,hr_index]
    lr_image = data_lr[:,:,lr_index]

    print(hr_image.shape)
    print(lr_image.shape)

    save_name = 'image_'+str(lr_index)+'.png'
    save_path = os.path.join(save_folder,save_name)
    plot_figure(lr=lr_image,hr=hr_image,index_lr=lr_index,index_hr=hr_index,save_path=save_path)




