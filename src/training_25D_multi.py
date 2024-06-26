import logging
import os
import sys
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
import random
from sklearn.model_selection import StratifiedGroupKFold

import monai
from monai.config import print_config
from monai.data import ThreadDataLoader, ThreadBuffer, decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete

from monai.metrics import ROCAUCMetric
from utils import seed_everything
from transform import get_transforms
from models import TimmModel
from dataset import CLSDatasets

def train_one_epoch(epoch, num_classes, model, optimizer, scheduler, loss_function, amp, scaler, src, train_ds, train_loader, writer, device):
    print("-" * 10)
    print(f"epoch {epoch + 1}")
    model.train()
    epoch_loss = 0
    step = 0
    for n, batch_data in enumerate(src):
        step_start = time.time()
        step += 1
        inputs, labels = batch_data[0].to(device, non_blocking=True), batch_data[1].to(device, non_blocking=True)
        labels = labels.contiguous().view(-1, num_classes) 
        optimizer.zero_grad()
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss =  loss_function(outputs, labels)
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
    return epoch_loss


def validate(model, num_classes, loss_function, val_loader, device, post_pred, post_label, auc_metric):
    model.eval()

    num_correct = 0.0
    metric_count = 0
    
    y_pred = torch.zeros([0,num_classes], dtype=torch.float32, device=device)
    y = torch.zeros([0,num_classes], dtype=torch.long, device=device)
    
    for n, val_data in enumerate(val_loader):
        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        val_labels = val_labels.contiguous().view(-1, num_classes)
        with torch.no_grad():
            pred = model(val_images)
            y_pred = torch.cat([y_pred, pred], dim=0)
            y = torch.cat([y, val_labels], dim=0)
        if len(y) > 12800:break
    loss = loss_function(y_pred[:,-1].float(), y[:,-1].float())
    print(y_pred, y_pred.size(), loss)
    acc_value = torch.eq(y_pred[:,-1]>0, y[:,-1])
    acc_metric = acc_value.sum().item() / len(acc_value)
    y_onehot = y[:,-1]
    y_pred_act = [post_pred(i) for i in decollate_batch(y_pred[:,-1])]
    auc_metric(y_pred_act, y_onehot)
    auc_result = auc_metric.aggregate()
    auc_metric.reset()
    del y_pred_act, y_onehot
    return acc_metric, auc_result, loss


def main(organ,num_epochs,datadir,batch_size,save_model_name,backbone,segtype,fold,seed=0,amp=True,use_buffer=True,metric_learning=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    print_config()
    if save_model_name.split('/')[-2]=='weights_three_internal':
        path = os.path.join(datadir,organ+'_dataset_train_three_internal_final.csv')
    else:
        path = os.path.join(datadir,organ+'_dataset_train_multi_internal.csv')
    print(path,'dataset_path')
    data_df = pd.read_csv(path)
    total_filenames = data_df['file'].values
    total_images = np.array([os.path.join(datadir,organ+'_'+segtype+'_img',p) for p in data_df['file']])
    total_preds = np.array([os.path.join(datadir,organ+'_'+segtype+'_pred',p.split('.')[0][:-5]+'.nii.gz') for p in data_df['file']])

    #各臓器でのラベルの定義
    abnormal_mapping = {
        'liver': ['嚢胞','脂肪肝','胆管拡張','SOL','変形','石灰化','pneumobilia','other_abnormality','nofinding'],
        'gallbladder': ['SOL','腫大','胆石','壁肥厚','ポリープ','other_abnormality','nofinding'],
        'kidney': ['嚢胞','SOL(including_complicated_cyst)','腫大','萎縮','変形','石灰化','other_abnormality','nofinding'],
        'pancreas': ['嚢胞','SOL','腫大','萎縮','石灰化','膵管拡張/萎縮','other_abnormality','nofinding'],
        'spleen' : ['嚢胞','SOL','変形','石灰化','other_abnormality','nofinding'],
        }
    abnormal_list = abnormal_mapping.get(organ)
    if abnormal_list is None:
        raise ValueError("please set appropriate organ")
    num_classes = len(abnormal_list)
    print(total_images[0])

    input_size = (256,256,64)
    total_labels = data_df.loc[:,abnormal_list].astype(int).values 
    g = torch.Generator()
    g.manual_seed(seed)
    print(len(total_labels),'num of abnormal label is ',total_labels.sum(axis=0))
    groups = data_df['患者ID'].astype(str)
    cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    for n,(train_idxs, test_idxs) in enumerate(cv.split(total_images, total_labels[:,-1], groups)):
        print('---------------- fold ',n,'-------------------')
        if n != fold :continue
        save_model_name_ = save_model_name.split('.pth')[0] +'_'+str(n)+'.pth'
        images_train,labels_train,file_train,mask_train = total_images[train_idxs],total_labels[train_idxs],total_filenames[train_idxs],total_preds[train_idxs]
        images_val,labels_val,file_val,mask_val = total_images[test_idxs],total_labels[test_idxs],total_filenames[test_idxs],total_preds[test_idxs]
        num_train_imgs = len(images_train)
        num_val_imgs = len(images_val)
        print('sample files',images_val[:5],labels_val[:5])
        print(f'number of train images is {num_train_imgs}, number of validation images is {num_val_imgs}')
        print('total_normal_labels is ',labels_val.sum(axis=0))
        train_files = [{"image": img, "label":np.repeat(label[np.newaxis, :], input_size[2]//2, axis=0),'filename':filename,'mask':mask} for img, label,filename,mask in zip(images_train, labels_train,file_train,mask_train)]
        val_files = [{"image": img, "label": np.repeat(label[np.newaxis, :], input_size[2]//2, axis=0), 'filename':filename,'mask':mask} for img, label,filename,mask in zip(images_val, labels_val,file_val,mask_val)]
        post_pred = Compose([Activations(sigmoid=True)])
        post_label = Compose([AsDiscrete(argmax=True)])

        num_gpu = torch.cuda.device_count()
        print('input size is ',input_size)

        train_transforms, val_transforms = get_transforms(input_size,seed)
        train_ds = CLSDataset(filenames=images_train,labels=labels_train,transform=train_transforms)
        train_loader = ThreadDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_gpu*6, pin_memory=pin_memory)
        val_ds = CLSDataset(filenames=images_val,labels=labels_val,transform=None)
        val_loader = ThreadDataLoader(val_ds, batch_size=batch_size, num_workers=num_gpu*2, pin_memory=pin_memory)

        model = TimmModel(backbone, input_size=input_size,num_classes=num_classes,pretrained=True).to(device)
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

        loss_function = torch.nn.BCEWithLogitsLoss().to(device)  # also works with this data
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)#23e-5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_train_imgs/batch_size)*num_epochs, eta_min=23e-6)
        auc_metric = ROCAUCMetric()
        scaler = torch.cuda.amp.GradScaler() if amp else None

        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        best_acc = 0
        epoch_loss_values = []
        metric_values = []
        writer = SummaryWriter()
        max_epochs = num_epochs

        src = ThreadBuffer(train_loader, 1) if use_buffer else train_loader

        for epoch in range(max_epochs):
            epoch_loss = train_one_epoch(epoch, num_classes, model, optimizer, scheduler, loss_function, amp, scaler, src, train_ds, train_loader, writer, device)
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                acc_metric, auc_result, loss = validate(model, num_classes, loss_function, val_loader, device, post_pred, post_label, auc_metric)
                
                if auc_result > best_metric:
                    best_metric = auc_result
                    best_acc = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), save_model_name_)
                    print("saved new best metric model")

                print("current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best AUC: {:.4f} at epoch {} then ACC: {:.4f} valloss: {:.4f}".format(
                            epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch, best_acc,loss))
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
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
    parser.add_argument('--seed', default=1, type=int,
                        help='random_seed.')
    parser.add_argument('--fold', default=0, type=int,
                        help='cross validation fold.')
    parser.add_argument('--backbone', default='convnext_base_in22ft1k',
                        help='backbone of 2dCNN model.')
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
    datadir = args.datadir
    backbone = args.backbone
    batch_size = args.batch_size
    segtype = args.segtype
    seed = args.seed
    fold = args.fold
    save_model_name = args.save_model_name
    seed_everything(seed)

    main(organ,num_epochs,datadir,batch_size,save_model_name,backbone,segtype,fold,seed,amp=True,use_buffer=True,metric_learning=False)
