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
import shutil
import tempfile
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
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

class L2ConstraintedNet(nn.Module):
    def __init__(self, org_model, alpha=16, num_classes=2):
        super().__init__()
        self.org_model = org_model
        self.alpha = alpha

    def forward(self, x):
        x = self.org_model(x)
        # モデルの出力をL2ノルムで割り、定数alpha倍する
        l2 = torch.sqrt((x**2).sum()) # 基本的にこの行を追加しただけ
        x = self.alpha * (x / l2)     # 基本的にこの行を追加しただけ
        return x

def calc_metrics(y_pred,y,organ,cutoff=None,dataset='validation'):
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
    if cutoff == None:
        cutoff = thres[index]
    print(f'{organ}, auc:{auc} ,cutoff : {cutoff}')
    ## plot auc curve
    plot_auc(fpr, tpr,organ,dataset = dataset)
    # Youden indexによるカットオフ値による分類
    y_pred_cutoff = pred >= cutoff
    # 混同行列をヒートマップで可視化
    cm = plot_confusion_matrix(target,y_pred_cutoff,organ,dataset = dataset)
    print('confusion matrix : \n',confusion_matrix(target,y_pred.argmax(dim=1).to('cpu').detach().numpy().copy()),
            '\n youden index : \n ',cm, '\n AUC : ',auc ,'accuracy : ',acc_metric )
    return cutoff

def main(datadir,organ,weight_path,outputdir,num_train_imgs,num_val_imgs,segtype,seed,amp=True):
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

    data_df = pd.read_csv(os.path.join(datadir,organ+'_dataset_train.csv'))
    test_df = pd.read_csv(os.path.join(datadir,organ+'_dataset_test.csv'))
    filenames = data_df['file']
    file_test = test_df['file']
    images = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in data_df['file']])
    images_test = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in test_df['file']])
    print(images[0])
    labels = data_df['abnormal'].astype(int).values
    labels_test = test_df['abnormal'].astype(int).values
    print(len(labels),'num of abnormal label is ',labels.sum())
    #le = LabelEncoder()
    #encoded_data = le.fit_transform(data)
    groups = data_df['FACILITY_CODE'].astype(str)+data_df['ACCESSION_NUMBER'].astype(str)
    cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_y_pred_val = torch.tensor([], dtype=torch.float32, device=device)
    total_y_val = torch.tensor([], dtype=torch.long, device=device)
    total_y_pred_test = torch.tensor([], dtype=torch.float32, device=device)
    total_y_test = torch.tensor([], dtype=torch.long, device=device)
    
    for n,(train_idxs, test_idxs) in enumerate(cv.split(images, labels, groups)):
        print('---------------- fold ',n,'-------------------')
        images_train,labels_train,file_train = images[train_idxs],labels[train_idxs],filenames[train_idxs]
        images_val,labels_val,file_val = images[test_idxs],labels[test_idxs],filenames[test_idxs]

        num_train_imgs = len(images_train)
        num_val_imgs = len(images_val)

        #train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images[:num], labels[:num],filenames[:num])]
        val_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_val, labels_val,file_val)]
        test_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_test, labels_test,file_test)]
        post_pred = Compose([Activations(softmax=True)])
        post_label = Compose([AsDiscrete(to_onehot=2)])
        # create a validation data loader
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=36, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate,)
        test_ds = Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=36, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate,)

        auc_metric = ROCAUCMetric()
        cutoff_criterions = list()

        # Create DenseNet121
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = monai.networks.nets.seresnext50(spatial_dims=3,in_channels=1,num_classes=2).to(device)
        #model = L2ConstraintedNet(model,alpha=16,num_classes=2).to(device)
        #model = monai.networks.nets.resnet18(spatial_dims=3,n_input_channels=1,num_classes=2).to(device)
        # model = monai.networks.nets.EfficientNetBN("efficientnet-b1", pretrained=False, 
        #         progress=False, spatial_dims=3, in_channels=1, num_classes=2,
        #         norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False).to(device)
        num_gpu = torch.cuda.device_count()
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
        weight_path_ = weight_path.split('.pth')[0] +'_'+str(n)+'.pth'
        model.load_state_dict(torch.load(weight_path_))
        model.eval()

        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            #saver = CSVSaver(output_dir="./output_eval")

            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)

            for val_data in val_loader:
                val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                outputs = model(val_images)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            total_y_pred_val = torch.cat([total_y_pred_val, y_pred], dim=0)
            total_y_val = torch.cat([total_y_val, y], dim=0)

        cutoff = calc_metrics(y_pred,y,organ,dataset='validation')

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
                outputs = model(test_images)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, test_labels], dim=0)
            total_y_pred_test = torch.cat([total_y_pred_test, y_pred], dim=0)
            total_y_test = y
        _ = calc_metrics(y_pred,y,organ,cutoff=cutoff,dataset='test')
        #saver.finalize()

    print('####-----------------overall metrics-------------------------###')
    total_y_pred_test = total_y_pred_test.view(-1,5,2).mean(axis=1)
    overall_cutoff = calc_metrics(total_y_pred_val,total_y_val,organ,dataset='total_val')
    _ = calc_metrics(total_y_pred_test,total_y_test,organ,cutoff=overall_cutoff,dataset='total_test')


    pred_df = pd.DataFrame([files,list(F.softmax(total_y_pred_test)[:,1].to('cpu').detach().numpy().copy()),
                    list(total_y_test.to('cpu').detach().numpy().copy())]).T
    pred_df.columns = ['file','prediction','target']

    pred_df['final_prediction'] = (pred_df['prediction']>cutoff).astype(int)
    groups = test_df['FACILITY_CODE'].astype(str)+test_df['ACCESSION_NUMBER'].astype(str)
    columns = ['file','prediction','final_prediction','target','io_tokens','FINDING','FINDING_JSON']
    pred_df.merge(test_df,on='file',how='left').drop_duplicates(subset='file')[columns].to_csv(f'../result_eval/{organ}_with_finding.csv',index=False)

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
    parser.add_argument('--segtype', default="square",
                        help='whether to use seg or square.')

    args = parser.parse_args()
    print(args)

    datadir=args.datadir
    outputdir = args.outputdir
    os.makedirs(outputdir,exist_ok=True)
    organ = args.organ
    weight_path = args.weight_path
    num_train_imgs = args.num_train_imgs
    num_val_imgs = args.num_val_imgs
    segtype = args.segtype
    seed = args.seed
    main(datadir,organ,weight_path,outputdir,num_train_imgs,num_val_imgs,segtype,seed,amp=True)