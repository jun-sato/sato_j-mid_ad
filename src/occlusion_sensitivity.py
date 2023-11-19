import logging
import os
import sys
import shutil
import tempfile
import torch
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
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (9132, rlimit[1]))
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
from dataset import CLSDataset_eval
from models import TimmModel, TimmModelMultiHead
from transform import get_transforms
from utils import OcclusionSensitivityMap, Normalize

def custom_blend(image1, image2, alpha):
    return np.clip(image1 * (1 - alpha) + image2 * alpha, 0, 1)

def apply_occlusion_sensitivity(image_dir,save_imagedir,organ,segtype,seed,backbone,load_model_name,input_size):
    print('input size is ',input_size)

    test_transforms = Compose(
            [
                LoadImaged(keys=["image"], reader="itkreader"),
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


    organs_mapping = {
        'liver': (['嚢胞','脂肪肝','胆管拡張','SOL','変形','石灰化','pneumobilia','other_abnormality','nofinding'], '肝臓'),
        'adrenal': (['SOL','腫大','脂肪','石灰化','other_abnormality','nofinding'], '副腎'),
        'esophagus': (['mass','hernia','拡張','other_abnormality','nofinding'], '食道'),
        'gallbladder': (['SOL','腫大','変形','胆石','壁肥厚','ポリープ','other_abnormality','nofinding'], '胆嚢'),
        'kidney': (['嚢胞','SOL(including_complicated_cyst)','腫大','萎縮','変形','石灰化','other_abnormality','nofinding'], '腎臓'),
        'pancreas': (['嚢胞','SOL','腫大','萎縮','石灰化','膵管拡張/萎縮','other_abnormality','nofinding'], '膵臓'),
        'spleen': (['嚢胞','SOL','変形','石灰化','other_abnormality','nofinding'], '脾臓'),
    }

    abnormal_list, col = organs_mapping.get(organ)
    if abnormal_list is None or col is None:
        raise ValueError("please set appropriate organ")
    num_classes = len(abnormal_list)

    test_df = pd.read_csv(os.path.join(datadir,'dataset_test.csv'))

    if col=='腎臓' or col=='副腎':
        print('specified organ is bilateral')
        test_df_left = test_df.copy()
        test_df_left['file'] = test_df['file'].apply(lambda x:x.split('.')[0]+'left'+'.nii.gz')
        test_df_left[col] = test_df_left['左'+col]
        test_df['file'] = test_df['file'].apply(lambda x:x.split('.')[0]+'right'+'.nii.gz')
        test_df[col] = test_df['右'+col]
        test_df = pd.concat([test_df,test_df_left],axis=0).reset_index(drop=True)
        print(test_df[['file',col]].head())


    file_test = test_df['file']
    images_test = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in test_df['file']])
    print(images_test[0])
    orig_test = np.array([os.path.join(datadir,organ+'_seg_img',p) for p in test_df['file']])
    labels_test = test_df[col].isna().astype(int).values
    print(len(labels_test),'num of abnormal label is ',labels_test.sum())

    test_df = test_df[images_test!=None].reset_index(drop=True) ##ファイルが存在しなければ計算から除外する。
    print(test_df.shape,'before shape')

    orig_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(orig_test, labels_test,file_test)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmModel(backbone, input_size=input_size, pretrained=False,num_classes=num_classes).to(device)
    num_gpu = torch.cuda.device_count()
    model = torch.nn.DataParallel(model, device_ids=list(range(4)))

    orig_ds = Dataset(data=orig_files, transform=test_transforms)
    orig_loader = DataLoader(orig_ds, batch_size=1, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate,)

    test_ds = CLSDataset_eval(filenames=images_test,labels=labels_test,transform=None)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_gpu*2, pin_memory=True, collate_fn=pad_list_data_collate,)

    up = torch.nn.Upsample(size=input_size)
            
    ##１症例ごとにocclusion sensitivityを適用する。
    for n,(orig_data,test_data) in enumerate(zip(orig_loader,test_loader)):
        test_images, test_labels,test_fname =test_data[0].to(device), test_data[1].to(device),test_data[2]
        orig_images = orig_data['image']
        #np.save('tmp.npy',orig_images.cpu().detach().numpy())
        tmp = test_images#すぐけす
        print(orig_images.size()) ###(bs,ch,w,h,d)
        test_images = torch.permute(test_images, (0, 2, 1, 3, 4)) ###(bs,w,h,d,ch)
        outputs = torch.zeros((1,1,32,256,256))
        print(test_images.size(),test_fname)
        for i in range(5):
            print('---------------- fold ',i,'-------------------')
            weight_path_ = load_model_name.split('.pth')[0] +'_'+str(i)+'.pth'
            model.load_state_dict(torch.load(weight_path_))
            model.eval()
            avg = model(tmp)
            print(avg.size(),'avg_size',avg)
            occ_sens = OcclusionSensitivity(
                model, mask_size=[2,16,16],overlap=0.25, n_batch=16, verbose=True,activate=False
            )
            # occlusion_sensitivity = OcclusionSensitivityMap(model,mask_size=(4,16,16),stride=(2,8,8),num_classes=num_classes)
            # occ_map = occlusion_sensitivity.generate_sensitivity_map(test_images, 24)
            # occ_map = ((occ_map-occ_map.min())/(occ_map.max()-occ_map.min()))*255
            #print('try',occ_map.size())
            occ_map, most_probable_class = occ_sens(test_images)
            print(occ_map.size(),most_probable_class.shape)
            outputs = outputs+(occ_map[:,-1:,:,:,:].detach().cpu()-avg[0,-1].detach().cpu())/5

        outputs = torch.permute(outputs, (0, 1, 3, 4, 2))
        outputs = up(outputs)
        outputs = torch.flip(outputs,[2,4])
        orig_images = torch.rot90(orig_images,k=1,dims=[2,3])
        print(outputs.size(),outputs.max(),outputs.min(),orig_images.max(),orig_images.min())
        np.save(f'{save_imagedir}/{os.path.basename(test_fname[0])}.npy',outputs.cpu().detach().numpy())
        blend_img_occ = blend_images(image=orig_images, label=outputs, alpha=0.3, cmap="jet", rescale_arrays=True)
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
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')

    parser.add_argument('--datadir', default="/mnt/hdd/jmid/data/",
                        help='path to the data directory.')
    parser.add_argument('--save_imagedir', default="../attention_maps/",
                        help='output folder of the occulusion sensitivity.')
    parser.add_argument('--organ', default='liver', type=str,
                        help='which organ to predict')
    parser.add_argument('--segtype', default="25D2",
                        help='whether to use seg or square or 2.5 dimentions.')
    parser.add_argument('--seed', default=0, type=int,
                        help='random_seed.')
    parser.add_argument('--backbone', default='tf_efficientnetv2_s_in21ft1k',
                        help='backbone of 2dCNN model.')
    parser.add_argument('--weight_path', default="/mnt/hdd/jmid/data/weight.pth",
                        help='path to the weight.')

    args = parser.parse_args()
    print(args)

    datadir = args.datadir
    save_imagedir = args.save_imagedir
    organ = args.organ
    segtype = args.segtype
    seed  = args.seed
    backbone = args.backbone
    weight_path = args.weight_path
    input_size = (256,256,64)
    os.makedirs(save_imagedir,exist_ok=True)
    apply_occlusion_sensitivity(datadir,save_imagedir,organ,segtype,seed,backbone,weight_path,input_size)
    print('finish processing..')