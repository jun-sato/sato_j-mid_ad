import logging
import os
import sys
import shutil
import tempfile
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import time
import pathlib
import glob
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    Spacingd,
    SpatialPadd,
    Resized
)
from monai.utils import get_torch_version_tuple, set_determinism
from monai.metrics import ROCAUCMetric
#from loss import LabelSmoothingCrossEntropy

def get_transforms(spatial_size):
    # Define transforms
    train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                #OneOf([RandAdjustContrastd(keys=["image"],prob=0.3,gamma=(0.75,1.5)),
                #        RandScaleIntensityd(keys=["image"], factors = [-0.2,0.2], prob=0.3),
                #        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-200,
                    a_max=400,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
                RandRotated(keys=['image'],range_x =np.pi*0.5,range_y=np.pi*0.5, prob=0.3),
                #SpatialPadd(keys=["image"],spatial_size=(256, 256, 64)),
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
                    a_min=-200,
                    a_max=400,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
                #SpatialPadd(keys=["image"],spatial_size=(256, 256, 64)),
                Resized(keys=["image"], spatial_size=spatial_size),
            ]
        )
    return train_transforms,val_transforms


def main(organ,num_epochs,num_classes,datadir,batch_size,num_train_imgs,num_val_imgs,save_model_name,seed=0,amp=True,use_buffer=True,metric_learning=False):
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()
    data_df = pd.read_csv(os.path.join(datadir,organ+'_dataset.csv'))
    filenames = data_df['file']
    images = [os.path.join(datadir,organ+'_square_img',p) for p in data_df['file']]
    print(images[0])
    labels = data_df['abnormal'].astype(int)
    print(len(labels),'num of abnormal label is ',labels.sum())
    #le = LabelEncoder()
    #encoded_data = le.fit_transform(data)
    images_train, images_test, labels_train, labels_test, file_train, file_test = train_test_split(images, labels,filenames, shuffle=True, stratify=labels,random_state=seed,train_size=num_train_imgs)
    images_val, images_test, labels_val, labels_test, file_val, file_test = train_test_split(images_test, labels_test,file_test, shuffle=True, stratify=labels_test,random_state=seed,train_size=num_val_imgs)
    train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images_train, labels_train,file_train)]
    val_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_val, labels_val,file_val)]
    test_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images_test, labels_test,file_test)]
    # Represent labels in one-hot format for binary classifier training,
    # BCEWithLogitsLoss requires target to have same shape as input
    #labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    num_gpu = torch.cuda.device_count()
    # create a training data loader
    if organ=='liver':
        input_size = (320,320,64)
    else:
        input_size = (256,256,64)
    train_transforms, val_transforms = get_transforms(input_size)
    # train_ds = PersistentDataset(
    #     data=data_dicts[:20000], transform=train_transforms, cache_dir=persistent_cache
    # )
    #train_ds = ImageDataset(image_files=images[:20000], labels=labels[:20000], transform=train_transforms)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_gpu*2, pin_memory=pin_memory, collate_fn=pad_list_data_collate,)
    # create a validation data loader
    # val_ds = PersistentDataset(
    #     data=data_dicts[20000:], transform=val_transforms, cache_dir=persistent_cache
    # )
    val_ds = Dataset(data=val_files, transform=val_transforms)
    #val_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_gpu*2, pin_memory=pin_memory, collate_fn=pad_list_data_collate,)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    if metric_learning:
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes).to(device)
        distance = distances.CosineSimilarity()
        regularizer = regularizers.RegularFaceRegularizer()
        loss_function = losses.ArcFaceLoss(num_classes, 128, margin=28.6, scale=64,weight_regularizer=regularizer, distance=distance).to(device)

    else:
        #model = monai.networks.nets.resnet18(spatial_dims=3,n_input_channels=1,num_classes=num_classes).to(device)
        #model = monai.networks.nets.EfficientNetBN("efficientnet-b1", pretrained=False,
        #            progress=False, spatial_dims=3, in_channels=1, num_classes=num_classes,
        #            norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False).to(device)
        model = monai.networks.nets.seresnext50(spatial_dims=3,in_channels=1,num_classes=num_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    #loss_function = FocalLoss(to_onehot_y=True,reduction='mean', gamma=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    #loss_function = LabelSmoothingCrossEntropy().to(device)
    # loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data


    #optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5) # lr is min lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_train_imgs/batch_size)*num_epochs, eta_min=0.0001)
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
    set_determinism(seed=seed)

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
            #plot_2d_or_3d_image(torch.permute(inputs,(0,1,4,2,3)),step=1,writer=writer,tag=filenames[0]+str(labels[0]),max_frames=8)
            #print(labels,filenames)
            optimizer.zero_grad()

            if amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
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

            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)

            for val_data in val_loader:
                val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                with torch.no_grad():
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot

            if acc_metric > best_metric:
                best_metric = acc_metric
                best_auc = auc_result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), save_model_name)
                print("saved new best metric model")

            print("current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {} then AUC: {:.4f}".format(
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
    parser.add_argument('--num_train_imgs', default=13000, type=int,
                        help='number of images for training.')
    parser.add_argument('--num_val_imgs', default=13000, type=int,
                        help='number of images for validation.')
    parser.add_argument('--seed', default=0, type=int,
                        help='random_seed.')
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
    num_train_imgs = args.num_train_imgs
    num_val_imgs = args.num_val_imgs
    seed = args.seed
    save_model_name = args.save_model_name

    main(organ,num_epochs,num_classes,datadir,batch_size,num_train_imgs,num_val_imgs,save_model_name,seed,amp=True,use_buffer=True,metric_learning=False)
