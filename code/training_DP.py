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
import time
import pathlib
import glob

from torch.utils.data import default_collate

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, PersistentDataset, pad_list_data_collate, ThreadBuffer
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import get_torch_version_tuple, set_determinism


def get_transforms():
    # Define transforms
    train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )

    val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
            ]
        )
    return train_transforms,val_transforms

def main(organ,num_epochs,datadir,batch_size,amp=True,use_buffer=True,):
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    images = glob.glob(f'{datadir}/{organ}/*.nii.gz')
    print(f'{len(images)} images found.')
    # 2 binary labels for gender classification: man or woman
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]*3489)

    # Represent labels in one-hot format for binary classifier training,
    # BCEWithLogitsLoss requires target to have same shape as input
    labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]

    ##cacheする場所を設定する
    persistent_cache = pathlib.Path(os.getcwd(), "persistent_cache")
    persistent_cache.mkdir(parents=True, exist_ok=True)

    # # Define nifti dataset, data loader
    # check_ds = PersistentDataset(
    #     data=data_dicts[:10], transform=train_transforms, cache_dir=persistent_cache
    # )
    # check_loader = DataLoader(check_ds, batch_size=4, num_workers=2, pin_memory=pin_memory,collate_fn=pad_list_data_collate,)

    # batch= monai.utils.misc.first(check_loader)
    # im,label = batch['image'],batch['label']
    # print(type(im), im.shape, label, label.shape)

    # create a training data loader
    train_transforms, val_transforms = get_transforms()
    train_ds = PersistentDataset(
        data=data_dicts[:10000], transform=train_transforms, cache_dir=persistent_cache
    )
    #train_ds = ImageDataset(image_files=images[:10], labels=labels[:10], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_gpu*2, pin_memory=pin_memory, collate_fn=pad_list_data_collate,)
    # create a validation data loader
    val_ds = PersistentDataset(
        data=data_dicts[:10000], transform=val_transforms, cache_dir=persistent_cache
    )
    #val_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_gpu*2, pin_memory=pin_memory, collate_fn=pad_list_data_collate,)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    model = monai.networks.nets.resnet18(spatial_dims=3,n_input_channels=1,num_classes=2).to(device)
    num_gpu = torch.cuda.device_count()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    ###to do : cosine annealingを追加する。
    scaler = torch.cuda.amp.GradScaler() if amp else None

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    max_epochs = num_epochs
    set_determinism(seed=0)

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
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
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
            print(
                   f"{step}/{len(train_ds) // train_loader.batch_size},"
                   f" train_loss: {loss.item():.4f}"
                   f" step time: {(time.time() - step_start):.4f}"
            )
            epoch_len = len(train_ds) // train_loader.batch_size

            # outputs = model(inputs)
            # loss = loss_function(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # epoch_loss += loss.item()
            # epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                if amp:
                    with torch.cuda.amp.autocast():
                        val_outputs = model(val_images)
                        value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                        metric_count += len(value)
                        num_correct += value.sum().item()
                else:
                    with torch.no_grad():
                        val_outputs = model(val_images)
                        value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                        metric_count += len(value)
                        num_correct += value.sum().item()
                        
            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')
    parser.add_argument('--organ', default="pancreas",
                        help='which organ AD model to train')
    parser.add_argument('--num_epochs', default=5, type=int,
                        help='number of epochs to train.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='number of batch size to train.')
    parser.add_argument('--datadir', default="/mnt/hdd/jmid/data/",
                        help='path to the data directory.')
    

    args = parser.parse_args()
    print(args)

    organ = args.organ
    num_epochs = args.num_epochs
    datadir = args.datadir 
    batch_size = args.batch_size
    main(organ,num_epochs,datadir,batch_size,amp=True,use_buffer=True,)
