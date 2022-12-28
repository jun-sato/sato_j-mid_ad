import logging
import os
import sys
import shutil
import tempfile
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import time
import pathlib
import glob
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
#from pytorch_metric_learning import losses, distances, regularizers

from torch.utils.data import default_collate
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

import monai
from monai.losses import FocalLoss
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, PersistentDataset, pad_list_data_collate, ThreadBuffer,Dataset,decollate_batch

from monai.transforms import (
    OneOf,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianNoised,
    EnsureChannelFirstd,
    AsDiscrete,
    Activations,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandRotated,
    ScaleIntensityRanged,
    MaskIntensityd,
    Spacingd,
    SpatialPadd,
    Resized
)
from monai.utils import get_torch_version_tuple, set_determinism
from monai.metrics import ROCAUCMetric
from loss import LabelSmoothingCrossEntropy
from utils import seed_everything,seed_worker,L2ConstraintedNet


def get_transforms(spatial_size,seed):
    # Define transforms
    train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-100,
                    a_max=400,
                    b_min=0,
                    b_max=1.0,#2.0にしても良さそう
                    clip=True,
                ),
                RandScaleIntensityd(keys=["image"], factors = [-0.2,0.2], prob=0.4),
                RandAdjustContrastd(keys=["image"],prob=0.3,gamma=(0.75,1.25)),
                RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
                #SpatialPadd(keys=["image"],spatial_size=(224, 224, 48)),
                #MaskIntensityd(keys=["image"],mask_key="mask"),
                Resized(keys=["image"], spatial_size=spatial_size),
            ]
        )

    val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-100,
                    a_max=400,
                    b_min=0,
                    b_max=1.0,
                    clip=True,
                ),
                #SpatialPadd(keys=["image"],spatial_size=(224, 224, 48)),
                #MaskIntensityd(keys=["image"],mask_key="mask"),
                Resized(keys=["image"], spatial_size=spatial_size),
            ]
        )
    train_transforms.set_random_state(seed=seed)
    val_transforms.set_random_state(seed=seed)
    return train_transforms,val_transforms


def main(organ,num_epochs,num_classes,datadir,batch_size,save_model_name,segtype,seed=0,amp=True,use_buffer=True,metric_learning=False):
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    print_config()
    data_df = pd.read_csv(os.path.join(datadir,organ+'_dataset_train_multi'+str(num_classes)+'.csv'))
    total_filenames = data_df['file'].values
    total_images = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in data_df['file']])
    total_preds = np.array([os.path.join(datadir,organ+'_'+segtype+'_pred',p.split('.')[0][:-5]+'.nii.gz') for p in data_df['file']])

    print(total_images[0])
    total_labels = data_df.iloc[:,-num_classes:].astype(float).values 
    g = torch.Generator()
    g.manual_seed(seed)
    print(len(total_labels),'num of abnormal label is ',total_labels.sum(axis=0))
    groups = data_df['FACILITY_CODE'].astype(str)+data_df['ACCESSION_NUMBER'].astype(str)
    cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    for n,(train_idxs, test_idxs) in enumerate(cv.split(total_images, total_labels[:,0], groups)):
        print('---------------- fold ',n,'-------------------')
        print(train_idxs,test_idxs)
        if n !=4 : continue
        save_model_name_ = save_model_name.split('.pth')[0] +'_'+str(n)+'.pth'
        images_train,labels_train,file_train,mask_train = total_images[train_idxs],total_labels[train_idxs],total_filenames[train_idxs],total_preds[train_idxs]
        images_val,labels_val,file_val,mask_val = total_images[test_idxs],total_labels[test_idxs],total_filenames[test_idxs],total_preds[test_idxs]
        num_train_imgs = len(images_train)
        num_val_imgs = len(images_val)
        print(f'number of train images is {num_train_imgs}, number of validation images is {num_val_imgs}')
        train_files = [{"image": img, "label": label,'filename':filename,'mask':mask} for img, label,filename,mask in zip(images_train, labels_train,file_train,mask_train)]
        val_files = [{"image": img, "label": label, 'filename':filename,'mask':mask} for img, label,filename,mask in zip(images_val, labels_val,file_val,mask_val)]
        post_pred = Compose([Activations(sigmoid=True)])
        post_label = Compose([AsDiscrete(argmax=True)])

        num_gpu = torch.cuda.device_count()
        # create a training data loader
        if organ=='liver':
            input_size = (256,256,64)
        else:
            input_size = (256,256,64)
        train_transforms, val_transforms = get_transforms(input_size,seed)
        #persistent_cache = pathlib.Path("/mnt/ssd/", "persistent_cache")
        #persistent_cache.mkdir(parents=True, exist_ok=True)
        train_ds = Dataset(data=train_files, transform=train_transforms)#,cache_dir=persistent_cache)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_gpu*2, pin_memory=pin_memory, collate_fn=pad_list_data_collate,worker_init_fn=seed_worker,generator=g,)
        # create a validation data loader
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_gpu*2, pin_memory=pin_memory, collate_fn=pad_list_data_collate,)

        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        model = monai.networks.nets.seresnext50(spatial_dims=3,in_channels=1,num_classes=num_classes).to(device)
        #model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes).to(device)
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

        #loss_function = FocalLoss(to_onehot_y=True,reduction='mean', gamma=2).to(device)
        #loss_function_ce = torch.nn.CrossEntropyLoss().to(device)
        #loss_function_bce = LabelSmoothingCrossEntropy().to(device)
        loss_function_bce = torch.nn.BCEWithLogitsLoss().to(device)  # also works with this data


        #optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,nesterov=True, momentum=0.9, weight_decay=1e-5) # lr is min lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_train_imgs/batch_size)*80, eta_min=0.00001)
        auc_metric = ROCAUCMetric()
        ###to do : cosine annealingを追加する。
        scaler = torch.cuda.amp.GradScaler() if amp else None

        # start a typical PyTorch training
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        writer = SummaryWriter()
        max_epochs = num_epochs
        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            src = ThreadBuffer(train_loader, 1) if use_buffer else train_loader
            for batch_data in src:
                step_start = time.time()
                step += 1
                inputs, labels, filenames = batch_data['image'].to(device), batch_data['label'].to(device), batch_data['filename']
                plot_2d_or_3d_image(torch.permute(inputs,(0,1,4,2,3)),step=1,writer=writer,tag=filenames[0]+str(labels[0]),max_frames=8)
                #print(labels,filenames)
                optimizer.zero_grad()

                if amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = loss_function_bce(outputs[:,0], labels[:,0]) + 0.02*loss_function_bce(outputs[:,1:],labels[:,1:]) 
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = loss_function_bce(outputs, labels)
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
                scheduler.step()
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size},"
                    f" train_loss: {loss.item():.4f}"
                    f" step time: {(time.time() - step_start):.4f}"
                )
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()

                num_correct = 0.0
                metric_count = 0

                y_pred = torch.zeros([0, num_classes], dtype=torch.float32, device=device)
                y = torch.zeros([0,num_classes], dtype=torch.long, device=device)
                
                for n,val_data in enumerate(val_loader):
                    if n >50:break
                    val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                    with torch.no_grad():
                        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)

                print(y_pred,y_pred.size())
                acc_value = torch.eq(y_pred[:,0]>0, y[:,0])
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = y#[post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_auc = auc_result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), save_model_name_)
                    print("saved new best metric model")

                print("current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best acc: {:.4f} at epoch {} then AUC: {:.4f}".format(
                            epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch, best_auc))
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

        print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')
    parser.add_argument('--organ', default="pancreas",
                        help='which organ AD model to train')
    parser.add_argument('--num_epochs', default=5, type=int,
                        help='number of epochs to train.')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of target classes.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='number of batch size to train.')
    parser.add_argument('--seed', default=0, type=int,
                        help='random_seed.')
    parser.add_argument('--segtype', default="square",
                        help='whether to use seg or square.')
    parser.add_argument('--datadir', default="/mnt/hdd/jmid/data/",
                        help='path to the data directory.')
    parser.add_argument('--save_model_name', default="weights/best_metric_model_classification3d_dict.pth",
                        help='save_model_name.')


    args = parser.parse_args()
    print(args)
    organ = args.organ
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    datadir = args.datadir
    batch_size = args.batch_size
    segtype = args.segtype
    seed = args.seed
    save_model_name = args.save_model_name
    seed_everything(seed)
    #set_determinism(seed,use_deterministic_algorithms=True)

    main(organ,num_epochs,num_classes,datadir,batch_size,save_model_name,segtype,seed,amp=True,use_buffer=False,metric_learning=False)