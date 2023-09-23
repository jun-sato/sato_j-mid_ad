import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.data import Dataset

class CLSDataset(Dataset):
    def __init__(self, filenames, labels, transform, num_zslices=32):

        self.files = filenames
        self.labels = labels
        self.transform = transform
        self.num_zslices = num_zslices

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.files[index]
        labels = np.array(self.labels[index])
        images = sitk.ReadImage(filepath)
        images = sitk.GetArrayFromImage(images)
        #imgs = []
        #for n in range(len(images)):
        #    image = self.transform(image = images[n].transpose(1,2,0))['image']
        #    image = image.transpose(2, 0, 1).astype(np.float32)
        #    imgs.append(image)
        #images = np.stack(images,0)
        labels  = np.repeat(labels[np.newaxis], self.num_zslices, axis=0)
        images = torch.from_numpy(images).float()
        if self.transform is not None:
            images = [self.transform({'image': image})['image'] for image in images]
            images = torch.stack(images).float()
        labels = torch.from_numpy(labels).float()
            
        return images, labels


class CLSDataset_eval(Dataset):
    def __init__(self, filenames, labels, transform, num_zslices=32):

        self.files = filenames
        self.labels = labels
        self.transform = transform
        self.num_zslices = num_zslices

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.files[index]
        labels = np.array(self.labels[index])
        images = sitk.ReadImage(filepath)
        images = sitk.GetArrayFromImage(images)
        #imgs = []
        #for n in range(len(images)):
        #    image = self.transform(image = images[n].transpose(1,2,0))['image']
        #    image = image.transpose(2, 0, 1).astype(np.float32)
        #    imgs.append(image)
        #images = np.stack(images,0)
        labels  = np.repeat(labels[np.newaxis], self.num_zslices, axis=0)
        images = torch.from_numpy(images).float()
        labels = torch.from_numpy(labels).float()
            
        return images, labels, os.path.basename(filepath)



class NiftiDataset(Dataset):
    def __init__(self, filenames, labels, transform, num_zslices=32):

        self.files = filenames
        self.labels = labels
        self.transform = transform
        self.num_zslices = num_zslices

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.files[index]
        labels = np.array(self.labels[index])
        image = sitk.ReadImage(filepath)
        image = sitk.GetArrayFromImage(image)[np.newaxis]
        image = torch.from_numpy(image)
        #(image.size(),image.max(),image.min())
        if self.transform is not None:
            image = self.transform(image).squeeze()
        #print(image.size())
        # スライスをサンプリングするインデックスを計算
        # z_indices = torch.linspace(0, image.shape[0]-1, self.num_zslices).long()
    # スライスをサンプリングするインデックスを計算
        total_slices = image.shape[0]
        step = total_slices // self.num_zslices
        if step > 0:
            start_slice = torch.randint(0, step, (1,)).item()
        else:
            # stepが0の場合の処理（例えば、デフォルトの値を設定するなど）
            start_slice = 0
            step=1

        z_indices = torch.arange(start_slice, total_slices, step, dtype=torch.long)[:self.num_zslices]

        # 各サンプリングスライスの中央から5スライスを抽出
        extracted_slices_list = []
        for center_idx in z_indices:
            start_idx = torch.clamp(center_idx - 2, 0, image.shape[0]-5)  # 5枚分のスライスが取得できるように制約
            end_idx = start_idx + 5  # 常に5スライス取得
            slices = image[start_idx:end_idx]

            extracted_slices_list.append(slices)

        # 5スライスをチャンネル方向に結合
        extracted_slices_stack = torch.cat(extracted_slices_list, dim=0).float()
        labels  = np.repeat(labels[np.newaxis], self.num_zslices, axis=0)
        labels = torch.from_numpy(labels).float()
            
        return extracted_slices_stack, labels, os.path.basename(filepath)
