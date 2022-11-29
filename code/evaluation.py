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
from sklearn.metrics import confusion_matrix

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

def main(datadir,organ,weight_path,amp=True):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


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
                Resized(keys=["image"], spatial_size=(256, 256, 64)),
            ]
        )

    data_df = pd.read_csv(os.path.join(datadir,organ+'_dataset.csv'))
    filenames = data_df['file']
    images = [os.path.join(datadir,organ+'_square_img',p) for p in data_df['file']]
    print(images[0])
    labels = data_df['abnormal'].astype(int)
    #le = LabelEncoder()
    #encoded_data = le.fit_transform(data)
    num = 8000
    #train_files = [{"image": img, "label": label,'filename':filename} for img, label,filename in zip(images[:num], labels[:num],filenames[:num])]
    val_files = [{"image": img, "label": label, 'filename':filename} for img, label,filename in zip(images[num:], labels[num:],filenames[num:])]
    # Represent labels in one-hot format for binary classifier training,
    # BCEWithLogitsLoss requires target to have same shape as input
    #labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    #val_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate,)
    auc_metric = ROCAUCMetric()

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.seresnext50(spatial_dims=3,in_channels=1,num_classes=2).to(device)
    #model = monai.networks.nets.resnet18(spatial_dims=3,n_input_channels=1,num_classes=2).to(device)
    # model = monai.networks.nets.EfficientNetBN("efficientnet-b1", pretrained=False, 
    #          progress=False, spatial_dims=3, in_channels=1, num_classes=2,
    #          norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False).to(device)
    num_gpu = torch.cuda.device_count()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    model.load_state_dict(torch.load(weight_path))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        saver = CSVSaver(output_dir="./output_eval")

        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)

        for val_data in val_loader:
            val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)

            with torch.no_grad():
                tmp = model(val_images)
                y_pred = torch.cat([y_pred, tmp], dim=0)
                ##print(y_pred>0.5,y_pred.argmax(dim=1))
                y = torch.cat([y, val_labels], dim=0)
            saver.save_batch(tmp, val_data["image"].meta)
            

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        auc_metric.reset()
        pred =y_pred.argmax(dim=1).to('cpu').detach().numpy().copy()
        target = y.to('cpu').detach().numpy().copy()
        print(confusion_matrix(target,pred))
        del y_pred_act, y_onehot
        print('confusion matrix : ',confusion_matrix(target,pred),'AUC : ',auc_metric,auc_result,'accuracy : ',acc_metric )
        saver.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')
    parser.add_argument('--organ', default="pancreas",
                        help='which organ AD model to train')
    parser.add_argument('--datadir', default="/mnt/hdd/jmid/data/",
                        help='path to the data directory.')
    parser.add_argument('--weight_path', default="/mnt/hdd/jmid/data/weight.pth",
                        help='path to the weight.')
    
    args = parser.parse_args()
    print(args)

    datadir=args.datadir
    organ = args.organ
    weight_path = args.weight_path
    main(datadir,organ,weight_path,amp=True)