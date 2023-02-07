from medcam import medcam
import logging
import os
import sys
import shutil
import tempfile
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
import seaborn as sns
from monai.visualize import GradCAM ,OcclusionSensitivity
from monai.visualize.utils import blend_images
#%config InlineBackend.figure_format = 'retina'

import monai
from monai.data import DataLoader, CSVSaver,ImageDataset, PersistentDataset, pad_list_data_collate, ThreadBuffer,Dataset,decollate_batch
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    AsDiscrete,
    Activations,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    Resized
)
from monai.metrics import ROCAUCMetric
import torch.nn.functional as F
from utils import TimmModel, CLSDataset


def apply_occlusion_sensitivity(image_dir,save_imagedir,organ,segtype,seed,backbone,load_model_name,input_size):
    print('input size is ',input_size)
    # Define transforms for image
    test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="LAI"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-200,
                    a_max=400,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
                #SpatialPadd(keys=["image"],spatial_size=(256, 256, 64)),
                Resized(keys=["image"], spatial_size=input_size),
            ]
        )

    test_df = pd.read_csv(os.path.join(datadir,organ+'_dataset_test_clean.csv'))
    file_test = test_df['file']
    images_test = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in test_df['file']])
    print(images_test[0])
    orig_test = np.array([os.path.join(datadir,organ+'_seg_img',p) for p in test_df['file']])
    labels_test = test_df['human_label'].astype(int).values
    print(len(labels_test),'num of abnormal label is ',labels_test.sum())
    #le = LabelEncoder()
    #encoded_data = le.fit_transform(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images[:num], labels[:num],filenames[:num])]
    orig_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(orig_test, labels_test,file_test)]
    test_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_test, labels_test,file_test)]
    # create a testidation data loader
    num_gpu = torch.cuda.device_count()
    orig_ds = Dataset(data=orig_files, transform=test_transforms)
    orig_loader = DataLoader(orig_ds, batch_size=1, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate,)

    test_ds = CLSDataset(filenames=images_test,labels=labels_test,transform=None)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_gpu*2, pin_memory=True, collate_fn=pad_list_data_collate,)



    model = TimmModel(backbone, pretrained=False).to(device)
    num_gpu = torch.cuda.device_count()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
    up = torch.nn.Upsample(size=input_size)
            
    ##１症例ごとにocclusion sensitivityを適用する。
    for n,(orig_data,test_data) in enumerate(zip(orig_loader,test_loader)):
        test_images, test_labels,test_fname =test_data[0].to(device), test_data[1].to(device),test_data[2]
        orig_images = orig_data['image']
        print(orig_images.size()) ###(bs,ch,w,h,d)
        test_images = torch.permute(test_images, (0, 2, 3, 4, 1)) ###(bs,w,h,d,ch)
        outputs = torch.zeros((1,1,384,384,32))
        print(test_images.size(),test_fname)
        for i in range(5):
            print('---------------- fold ',i,'-------------------')
            weight_path_ = load_model_name.split('.pth')[0] +'_'+str(i)+'.pth'
            model.load_state_dict(torch.load(weight_path_))
            model.eval()
            occ_sens = OcclusionSensitivity(
                model, mask_size=[16,16,4],stride=[12,12,2], n_batch=24, verbose=True,per_channel=False
            )
            occ_map, _ = occ_sens(test_images)
            outputs = outputs+occ_map[:,:,:,:,:,0].detach().cpu()/5
            print(test_labels.to('cpu').numpy())
            #test_images = (test_images+1)/2
        outputs = up(outputs)
        outputs = torch.flip(outputs,[2,4])
        orig_images = torch.rot90(orig_images,k=1,dims=[2,3])
        print(outputs.size())
        torch.save(outputs, f'{save_imagedir}/{os.path.basename(test_fname[0])}.pt')
        blend_img_occ = blend_images(image=(orig_images+1)/2, label=outputs*3, alpha=0.5, cmap="hsv", rescale_arrays=False)
        os.makedirs(f'{save_imagedir}/{str(n)}_label_{str(test_labels)}',exist_ok=True)

        for i in range(64):
            # plot the slice 50 - 100 of image, label and blend result
            slice_index = 1 * i
            plt.figure("blend image and label", (12, 6))
            plt.subplot(1 ,3, 1)
            plt.title(f"image slice {slice_index}")
            plt.imshow(orig_images[0, 0, :, :, slice_index].cpu().detach(), cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label slice {slice_index}")
            plt.imshow(outputs[0, 0, :, :, slice_index].cpu().detach(),cmap="jet")
            plt.subplot(1, 3, 3)
            plt.title(f"occlusion blend slice {slice_index}")
            # switch the channel dim to the last dim
            plt.imshow(torch.moveaxis(blend_img_occ[0, 0, :, :, slice_index].cpu().detach(),1,1) )
            #plt.colorbar()
            #plt.show()
            plt.savefig(f'{save_imagedir}/{str(n)}_label_{str(test_labels)}/fname_{str(i)}temp.png')
            plt.close()
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')

    parser.add_argument('--datadir', default='../data/',
                        help='output folder of the cropped labels, (default: models)')
    parser.add_argument('--save_imagedir', default="../attention_maps/",
                        help='output folder of the occulusion sensitivity.')
    parser.add_argument('--organ', default='liver', type=str,
                        help='which organ to predict')
    parser.add_argument('--segtype', default="25D",
                        help='whether to use seg or square or 2.5 dimentions.')
    parser.add_argument('--seed', default=0, type=int,
                        help='random_seed.')
    parser.add_argument('--backbone', default='tf_efficientnetv2_s_in21ft1k',
                        help='backbone of 2dCNN model.')
    parser.add_argument('--load_model_name', default="weights/best_metric_model_classification3d_dict.pth",
                        help='load_model_name.')

    args = parser.parse_args()
    print(args)

    datadir = args.datadir
    save_imagedir = args.save_imagedir
    organ = args.organ
    segtype = args.segtype
    seed  = args.seed
    backbone = args.backbone
    load_model_name = args.load_model_name
    input_size = (384,384,64)
    os.makedirs(save_imagedir,exist_ok=True)
    apply_occlusion_sensitivity(datadir,save_imagedir,organ,segtype,seed,backbone,load_model_name,input_size)
    print('finish processing..')