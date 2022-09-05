# Lib Load
import os
from tkinter import Image
import cv2
import glob
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class ImageNetDataset(Dataset):
    def __init__(self, path, transforms, is_train):
        super(ImageNetDataset, self).__init__()
        self.transforms = transforms
        self.is_train = is_train


class ImageNet(pl.LightningDataModule):
    def __init__(self, path, workers, train_transforms, val_transforms, batch_size=None):
        super(ImageNet, self).__init__()
        self.path = path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader

    def val_dataloader(self):
        return DataLoader


if __name__ == '__main__':
    '''
    Dataset Loadaer Test
    run$ python -m dataset.classification/ImageNet_1k
    '''
    import albumentations
    import albumentations.pytorch
    
    train_transforms = albumentations.Compose