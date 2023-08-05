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
from monai.metrics import ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
import torch.nn.functional as F
import SimpleITK as sitk
from utils import seed_everything,seed_worker,L2ConstraintedNet,mixup,criterion
import timm
from dataset import CLSDataset_eval
from models import TimmModel, TimmModelMultiHead
from transform import get_transforms

def plot_auc(fpr, tpr,organ,dataset = 'validation',fold=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, marker='o', label=organ)    
    ax.legend()
    ax.grid()
    ax.set_xlabel('FPR: False Positive Rete', fontsize = 13)
    ax.set_ylabel('TPR: True Positive Rete', fontsize = 13)
    if fold is not None:
        fig.savefig(f'{outputdir}/{organ}_ROC_curve_{dataset}_fold{fold}.jpg')
    else:
        fig.savefig(f'{outputdir}/{organ}_ROC_curve_{dataset}.jpg')
    plt.close()

def plot_confusion_matrix(target,y_pred_cutoff,organ,dataset = 'validation',fold=None):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    cm = confusion_matrix(target, y_pred_cutoff)
    sns.heatmap(cm, annot=True,fmt = '.4g', cmap='Blues', ax=ax)
    ax.set_title('cutoff (Youden index)')
    #ax.ticklabel_format(style='plain')
    if fold is not None:
        fig.savefig(f'{outputdir}/{organ}_confsion_matrix_{dataset}_fold{fold}.jpg')
    else:
        fig.savefig(f'{outputdir}/{organ}_confsion_matrix_{dataset}.jpg')
    plt.close()
    return cm



def calc_metrics(y_pred,y,organ,cutoff=None,dataset='validation',fold=None):
    ##metrics計算
    print(y_pred,y)
    acc_value = torch.eq(y_pred>0, y)
    acc_metric = acc_value.sum().item() / len(acc_value)
    pred =F.sigmoid(y_pred).to('cpu').detach().numpy().copy()
    target = y.to('cpu').detach().numpy().copy()
    fpr, tpr, thres = roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    sng = 1 - fpr
    # Youden indexを用いたカットオフ基準
    # Youden_index_candidates = tpr-fpr
    # index = np.where(Youden_index_candidates==max(Youden_index_candidates))[0][0]
    # if cutoff == None:
    #     cutoff = thres[index]
    cutoff = 0.83
    print(f'{organ}, auc:{auc} ,cutoff : {cutoff}')
    ## plot auc curve
    plot_auc(fpr, tpr,organ,dataset = dataset,fold=fold)
    # # Youden indexによるカットオフ値による分類
    if dataset == 'test':
        for cutoff in np.arange(0,1,0.001):
            print('fdajsio')
            y_pred_cutoff = pred >= cutoff
            cm = plot_confusion_matrix(target,y_pred_cutoff,organ,dataset = dataset,fold=fold)
            print('cutoff: ',cutoff,cm)
    y_pred_cutoff = pred >= cutoff
    # 混同行列をヒートマップで可視化
    cm = plot_confusion_matrix(target,y_pred_cutoff,organ,dataset = dataset,fold=fold)
    print('confusion matrix : \n',confusion_matrix(target,(pred>0.5)),
            '\n youden index : \n ',cm, '\n AUC : ',auc ,'accuracy : ',acc_metric )
    return cutoff

def main(datadir,organ,num_classes,weight_path,outputdir,segtype,backbone,seed,amp=True):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    input_size = 256
    print('input size is ',input_size)
    # Define transforms for image
    _, val_transforms = get_transforms(input_size,seed)

    data_df = pd.read_csv(os.path.join(datadir,organ+'_dataset_train_multi_internal.csv'))
    test_df = pd.read_csv(os.path.join(datadir,organ+'_dataset_test.csv'))
    #test_df = test_df[test_df['大学名'].apply(lambda x:x in ['tokushima','ehime','kyushu','juntendo'])]
    filenames = data_df['file']
    file_test = test_df['file']
    images = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in data_df['file']])
    images_test = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in test_df['file']])

    labels = data_df['nofinding'].astype(int).values
    labels_test = test_df['肝臓'].isna().astype(int).values
    print(len(labels_test),'num of abnormal label is ',labels_test.sum())
    #le = LabelEncoder()
    #encoded_data = le.fit_transform(data)
    groups = data_df['患者ID'].astype(str)
    cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_y_pred_val = torch.tensor([], dtype=torch.float32, device=device)
    total_y_val = torch.tensor([], dtype=torch.long, device=device)
    total_y_pred_test = torch.zeros((len(labels_test)), dtype=torch.float32, device=device)
    total_y_test = torch.zeros((len(labels_test)), dtype=torch.long, device=device)
    
    for n,(train_idxs, test_idxs) in enumerate(cv.split(images, labels, groups)):
        #if n!=0:continue
        print('---------------- fold ',n,'-------------------')
        images_train,labels_train,file_train = images[train_idxs],labels[train_idxs],filenames[train_idxs]
        images_val,labels_val,file_val = images[test_idxs],labels[test_idxs],filenames[test_idxs]

        num_train_imgs = len(images_train)
        num_val_imgs = len(images_val)

        #train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images[:num], labels[:num],filenames[:num])]
        val_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_val, labels_val,file_val)]
        test_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_test, labels_test,file_test)]
        post_pred = Compose([Activations(sigmoid=True)])
        post_label = Compose([AsDiscrete(argmax=True)])
        # create a validation data loader
        num_gpu = torch.cuda.device_count()
        val_ds = CLSDataset_eval(filenames=images_val,labels=labels_val,transform=None)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=num_gpu*2, pin_memory=True, collate_fn=pad_list_data_collate,)
        test_ds = CLSDataset_eval(filenames=images_test,labels=labels_test,transform=None)
        test_loader = DataLoader(test_ds, batch_size=4, num_workers=num_gpu*2, pin_memory=True, collate_fn=pad_list_data_collate,)

        auc_metric = ROCAUCMetric()
        cutoff_criterions = list()

        # Create DenseNet121
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TimmModel(backbone, input_size=input_size, pretrained=False,num_classes=num_classes).to(device)
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
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                outputs = model(val_images) #torch.Size([4, 32])
                outputs = outputs[:,-1].view(4,32)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                if len(y) > 100:break
            y_pred = y_pred.mean(1)
            y = y.mean(1)
            total_y_pred_val = torch.cat([total_y_pred_val, y_pred], dim=0)
            total_y_val = torch.cat([total_y_val, y], dim=0)
        #print(total_y_pred_val.shape,total_y_val.shape) torch.Size([4]) torch.Size([4])
        cutoff = calc_metrics(y_pred,y,organ,dataset='validation',fold=str(n))
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            #saver = CSVSaver(output_dir="./output_eval")
            files = []
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for test_data in test_loader:
                test_images, test_labels,file = test_data[0].to(device), test_data[1].to(device),test_data[2]
                files += file
                outputs = model(test_images)
                outputs = outputs[:,-1].view(4,32)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, test_labels], dim=0)
            y_pred = y_pred.mean(1)
            y = y.mean(1)
            print(torch.eq(y_pred>0,y).sum(),'prediction')
            total_y_pred_test = total_y_pred_test + y_pred
            total_y_test = y
        _ = calc_metrics(y_pred,y,organ,cutoff=cutoff,dataset='test',fold=str(n))
        #saver.finalize()

    print('####-----------------overall metrics-------------------------###')
    total_y_pred_test = total_y_pred_test/5
    overall_cutoff = calc_metrics(total_y_pred_val,total_y_val,organ,dataset='total_val',fold=None)
    _ = calc_metrics(total_y_pred_test,total_y_test,organ,cutoff=overall_cutoff,dataset='total_test',fold=None)


    pred_df = pd.DataFrame([files,list(F.softmax(total_y_pred_test).to('cpu').detach().numpy().copy()),
                    list(total_y_test.to('cpu').detach().numpy().copy())]).T
    pred_df.columns = ['file','prediction','target']

    pred_df['final_prediction'] = (pred_df['prediction']>cutoff).astype(int)
    groups = test_df['患者ID']
    columns = ['file','prediction','final_prediction','target','io_tokens','所見','所見_JSON']
    pred_df.merge(test_df,on='file',how='left').drop_duplicates(subset='file')[columns].to_csv(f'../result_eval/{organ}_with_finding_{str(seed)}.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')
    parser.add_argument('--organ', default="pancreas",
                        help='which organ AD model to train')
    parser.add_argument('--datadir', default="/mnt/hdd/jmid/data/",
                        help='path to the data directory.')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of target classes.')
    parser.add_argument('--outputdir', default="../result_eval",
                        help='path to the folder in which results are saved.')
    parser.add_argument('--weight_path', default="/mnt/hdd/jmid/data/weight.pth",
                        help='path to the weight.')
    parser.add_argument('--backbone', default='tf_efficientnetv2_s_in21ft1k',
                        help='backbone of 2dCNN model.')
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
    num_classes = args.num_classes
    weight_path = args.weight_path
    segtype = args.segtype
    backbone = args.backbone
    seed = args.seed
    main(datadir,organ,num_classes,weight_path,outputdir,segtype,backbone,seed,amp=True)
