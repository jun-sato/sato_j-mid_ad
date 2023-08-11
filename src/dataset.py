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
