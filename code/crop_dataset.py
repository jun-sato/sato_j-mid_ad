import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
import pydicom
from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.filereader import read_dicomdir
from skimage.draw import polygon
import SimpleITK as sitk
import sys
import scipy.ndimage as ndi
from skimage import morphology
from scipy import signal
from multiprocessing import Pool
from functools import partial

## セグメンテーションしたあと、特定の臓器に焦点を当てて、それ以外の空間をcropするコード。
## 新たにimage,maskそれぞれで保存する。

def crop_img_and_dilatedmask(img,mask,dilated_array):
    ##dilationしたあと、その他の白い部分を消去する。

    z_up = np.nonzero(dilated_array)[0].max() #zの上端
    z_dn = np.nonzero(dilated_array)[0].min() #zの下端
    y_up = np.nonzero(dilated_array)[1].max() #yの上端
    y_dn = np.nonzero(dilated_array)[1].min() #yの下端
    x_up = np.nonzero(dilated_array)[2].max() #xの上端
    x_dn = np.nonzero(dilated_array)[2].min() #xの下端
    print(z_dn,z_up,x_dn,x_up,y_dn,y_up)

    return img[z_dn:z_up,y_dn:y_up,x_dn:x_up], mask[z_dn:z_up,y_dn:y_up,x_dn:x_up], dilated_array[z_dn:z_up,y_dn:y_up,x_dn:x_up]


def focus_organs(organ_index:list,mask_path):
    ####organ_index
    ''' "0": "background",
        "1": "right lung",
        "2": "left lung",
        "3": "heart",
        "4": "Aorta",
        "5": "esophagus",
        "6": "inside esophagus",
        "7": "uterus",
        "8": "liver",
        "9": "gallbladder",
        "10": "stomach+duodenum",
        "11": "air inside stomach",
        "12": "others inside stomach",
        "13": "spleen",
        "14": "right renal",
        "15": "left renal",
        "16": "IVC",
        "17": "portal vein etc",
        "18": "pancreas",
        "19": "bladder",
        "20": "prostate"
        if you want to focus on left and right lungs, organ_index=[1,2]
    '''
    try:
        default_img_path = image_dir + os.path.basename(mask_path).split('.')[0] + '_0000.nii.gz'
        default_mask_path = mask_dir + os.path.basename(mask_path).split('.')[0] + '.nii.gz'
        pred_img = sitk.ReadImage(mask_path)
        default_img = sitk.ReadImage(default_img_path)
        default_mask = sitk.ReadImage(default_mask_path)
        
        new_spacing = default_img.GetSpacing()
        new_origin = default_img.GetOrigin()
        new_direction = default_img.GetDirection()

        default_array = sitk.GetArrayFromImage(default_img)   
        default_mask = sitk.GetArrayFromImage(default_mask) 
        pred_array = sitk.GetArrayFromImage(pred_img)

        #pred_array = (pred_array==1) | (pred_array==2)
        if len(organ_index) ==1:
            pred_array = pred_array==organ_index
        elif len(organ_index) == 2:
            pred_array = (pred_array==organ_index[0]) | (pred_array==organ_index[1])
        else:
            raise ValueError("target organs is less than two!")
            
        ## morphology変換でdilation
        dilated_array = morphology.binary_dilation(pred_array, morphology.ball(radius=1))

        default_array,default_mask,dilated_array = crop_img_and_dilatedmask(default_array,default_mask,dilated_array)

        default_array = default_array * dilated_array
        default_mask = default_mask * dilated_array

        ## img_preprocessed　肺野のみを抽出した画像。
        img_preprocessed = sitk.GetImageFromArray(default_array)
        img_preprocessed.SetSpacing(new_spacing)
        img_preprocessed.SetOrigin(new_origin)
        img_preprocessed.SetDirection(new_direction)

        ## delated_arrayを保存
        mask_preprocessed = sitk.GetImageFromArray(default_mask)
        mask_preprocessed.SetSpacing(new_spacing)
        mask_preprocessed.SetOrigin(new_origin)
        mask_preprocessed.SetDirection(new_direction)

        sitk.WriteImage(img_preprocessed,f'{save_imagedir}/{os.path.basename(default_img_path)}')
        sitk.WriteImage(mask_preprocessed,f'{save_maskdir}/{os.path.basename(mask_path)}')
    except Exception as e:
        print(e,mask_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')
    parser.add_argument('--maskdir', default="/mnt/hdd/LCSG_impacts/compana_impacts_ps256Tr/",
                        help='output folder of the cropped labels, (default: models)')
    parser.add_argument('--imagedir', default='/mnt/ssd/nnUNet_raw_data_base/nnUNet_raw_data/Task801_Impacts/imagesTr/',
                        help='output folder of the cropped labels, (default: models)')
    
    parser.add_argument('--save_maskdir', default="/mnt/ssd/nnUNet_raw_data_base/nnUNet_raw_data/Task802_Impactscrop/labelsTr/",
                        help='output folder of the cropped labels, (default: models)')
    parser.add_argument('--save_imagedir', default="/mnt/ssd/nnUNet_raw_data_base/nnUNet_raw_data/Task802_Impactscrop/imagesTr/",
                        help='output folder of the cropped images , (default: models)')
    parser.add_argument('--num_threads', default=20, type=int,
                        help='number of threds to process , (default: 20)')
    parser.add_argument('--organ_id', default=8, type=int,
                        help='organ label identifier. see focus_organ function.')

    args = parser.parse_args()
    print(args)

    image_prediction_dir = args.maskdir
    image_dir = args.imagedir
    mask_dir = args.maskdir
    save_maskdir = args.save_maskdir
    save_imagedir = args.save_imagedir
    organ_id = args.organ_id
    os.makedirs(save_maskdir,exist_ok=True)
    os.makedirs(save_imagedir,exist_ok=True)

    p = Pool(args.num_threads) # プロセス数を設定

    lung_seg_predict_lists = glob.glob(os.path.join(image_prediction_dir,'*nii.gz'))
    result = p.map(partial(focus_organs,[organ_id]), lung_seg_predict_lists)

    print('finish processing..')

