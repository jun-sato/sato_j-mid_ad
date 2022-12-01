# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import argparse
import numpy as np
import torch
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_auc(fpr, tpr,organ,dataset = 'validation'):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, marker='o', label=organ)    
    ax.legend()
    ax.grid()
    ax.set_xlabel('FPR: False Positive Rete', fontsize = 13)
    ax.set_ylabel('TPR: True Positive Rete', fontsize = 13)
    fig.savefig(f'{outputdir}/{organ}_ROC_curve_{dataset}.jpg')
    plt.close()

def plot_confusion_matrix(target,y_pred_cutoff,organ,dataset = 'validation'):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    cm = confusion_matrix(target, y_pred_cutoff)
    sns.heatmap(cm, annot=True,fmt = '.4g', cmap='Blues', ax=ax)
    ax.set_title('cutoff (Youden index)')
    #ax.ticklabel_format(style='plain')
    fig.savefig(f'{outputdir}/{organ}_confsion_matrix_{dataset}.jpg')
    plt.close()
    return cm

def main(datadir,organ,weight_path,outputdir,num_train_imgs,num_val_imgs,seed,amp=True):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if organ=='liver':
        input_size = (320,320,64)
    else:
        input_size = (256,256,64)

    print('input size is ',input_size)
    # Define transforms for image
    val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
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
    data_df = pd.read_csv(os.path.join(datadir,organ+'_dataset.csv'))
    filenames = data_df['file']
    images = [os.path.join(datadir,organ+'_square_img',p) for p in data_df['file']]
    print(images[0])
    labels = data_df['abnormal'].astype(int)
    #le = LabelEncoder()
    #encoded_data = le.fit_transform(data)
    #train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images[:num], labels[:num],filenames[:num])]
    images_train, images_test, labels_train, labels_test, file_train, file_test = train_test_split(images, labels,filenames, shuffle=True, stratify=labels,random_state=seed,train_size=num_train_imgs)
    images_val, images_test, labels_val, labels_test, file_val, file_test = train_test_split(images_test, labels_test,file_test, shuffle=True, stratify=labels_test,random_state=seed,train_size=num_val_imgs)
    train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images_train, labels_train,file_train)]
    val_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_val, labels_val,file_val)]
    test_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_test, labels_test,file_test)]
    num = 13000
    train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images[:num], labels[:num],filenames[:num])]
    val_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images[num:-1000], labels[num:-1000],filenames[num:-1000])]
    test_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images[-1000:], labels[-1000:],filenames[-1000:])]
    
    # Represent labels in one-hot format for binary classifier training,
    # BCEWithLogitsLoss requires target to have same shape as input
    #labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    #val_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate,)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=2, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate,)

    auc_metric = ROCAUCMetric()
    cutoff_criterions = list()

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.seresnext50(spatial_dims=3,in_channels=1,num_classes=2).to(device)
    #model = monai.networks.nets.resnet18(spatial_dims=3,n_input_channels=1,num_classes=2).to(device)
    # model = monai.networks.nets.EfficientNetBN("efficientnet-b1", pretrained=False, 
    #         progress=False, spatial_dims=3, in_channels=1, num_classes=2,
    #         norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False).to(device)
    num_gpu = torch.cuda.device_count()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    model.load_state_dict(torch.load(weight_path))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        #saver = CSVSaver(output_dir="./output_eval")

        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)

        for val_data in val_loader:
            val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)

            with torch.no_grad():
                tmp = model(val_images)
                y_pred = torch.cat([y_pred, tmp], dim=0)
                ##print(y_pred>0.5,y_pred.argmax(dim=1))
                y = torch.cat([y, val_labels], dim=0)
            #saver.save_batch(tmp, val_data["image"].meta)
    
    ##metrics計算
    acc_value = torch.eq(y_pred.argmax(dim=1), y)
    acc_metric = acc_value.sum().item() / len(acc_value)
    pred =F.softmax(y_pred)[:,1].to('cpu').detach().numpy().copy()
    target = y.to('cpu').detach().numpy().copy()
    fpr, tpr, thres = roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    sng = 1 - fpr
    # Youden indexを用いたカットオフ基準
    Youden_index_candidates = tpr-fpr
    index = np.where(Youden_index_candidates==max(Youden_index_candidates))[0][0]
    cutoff = thres[index]
    print(f'{organ}, auc:{auc} ,cutoff : {cutoff}')
    ## plot auc curve
    plot_auc(fpr, tpr,organ,dataset = 'validation')

    # Youden indexによるカットオフ値による分類
    y_pred_cutoff = pred >= cutoff
    # 混同行列をヒートマップで可視化
    cm = plot_confusion_matrix(target,y_pred_cutoff,organ,dataset = 'validation')

    print('confusion matrix : \n',confusion_matrix(target,y_pred.argmax(dim=1).to('cpu').detach().numpy().copy()),
            '\n youden index : \n ',cm, '\n AUC : ',auc ,'accuracy : ',acc_metric )
    #saver.finalize()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        #saver = CSVSaver(output_dir="./output_eval")
        files = []
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        for test_data in test_loader:
            test_images, test_labels,file = test_data['image'].to(device), test_data['label'].to(device),test_data['filename']
            files += file
            with torch.no_grad():
                tmp = model(test_images)
                y_pred = torch.cat([y_pred, tmp], dim=0)
                ##print(y_pred>0.5,y_pred.argmax(dim=1))
                y = torch.cat([y, test_labels], dim=0)
            #saver.save_batch(tmp, val_data["image"].meta)
    
    acc_value = torch.eq(y_pred.argmax(dim=1), y)
    acc_metric = acc_value.sum().item() / len(acc_value)

    pred =F.softmax(y_pred)[:,1].to('cpu').detach().numpy().copy()
    target = y.to('cpu').detach().numpy().copy()
    
    fpr, tpr, thres = roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    sng = 1 - fpr

    print(f'{organ},test auc:{auc} ')
    ## plot auc curve
    plot_auc(fpr, tpr,organ,dataset = 'test')

    # Youden indexによるカットオフ値による分類
    y_pred_cutoff = pred >= cutoff
    # 混同行列をヒートマップで可視化
    cm = plot_confusion_matrix(target,y_pred_cutoff,organ,dataset = 'test')
    print('confusion matrix : \n',confusion_matrix(target,y_pred.argmax(dim=1).to('cpu').detach().numpy().copy()),
            '\n youden index : \n ',cm, '\n AUC : ',auc ,'accuracy : ',acc_metric )
    #saver.finalize()
    _ = pd.DataFrame([files,list(pred),list(target)]).T
    _.columns = ['file','prediction','target']
    _.to_csv('../result_eval/test_prediction.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')
    parser.add_argument('--organ', default="pancreas",
                        help='which organ AD model to train')
    parser.add_argument('--datadir', default="/mnt/hdd/jmid/data/",
                        help='path to the data directory.')
    parser.add_argument('--outputdir', default="../result_eval",
                        help='path to the folder in which results are saved.')
    parser.add_argument('--weight_path', default="/mnt/hdd/jmid/data/weight.pth",
                        help='path to the weight.')
    parser.add_argument('--num_train_imgs', default=13000, type=int,
                        help='number of images for training.')
    parser.add_argument('--num_val_imgs', default=13000, type=int,
                        help='number of images for validation.')
    parser.add_argument('--seed', default=0, type=int,
                        help='random_seed.')
    
    args = parser.parse_args()
    print(args)

    datadir=args.datadir
    outputdir = args.outputdir
    os.makedirs(outputdir,exist_ok=True)
    organ = args.organ
    weight_path = args.weight_path
    num_train_imgs = args.num_train_imgs
    num_val_imgs = args.num_val_imgs
    seed = args.seed
    main(datadir,organ,weight_path,outputdir,num_train_imgs,num_val_imgs,seed,amp=True)