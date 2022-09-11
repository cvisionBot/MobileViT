# Lib Load
import os
from tkinter import Image
import cv2
import glob
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from dataset.classification.utils import imagenet_collate_fn

class ImageNetDataset(Dataset):
    def __init__(self, path, transforms, is_train):
        super(ImageNetDataset, self).__init__()
        self.transforms = transforms
        self.is_train = is_train

        if is_train:
            self.data = glob.glob(path + '/train/**/*.JPEG')
            self.train_list = dict()
    
        else:
            self.data = glob.glob(path + '/val/**/*.JPEG')
            self.val_list = dict()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]
        transformed = self.transforms(image=img)['image']
        return transformed, label


class ImageNet(pl.LightningDataModule):
    def __init__(self, path, workers, train_transforms, val_transforms, batch_size=None):
        super(ImageNet, self).__init__()
        self.path = path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(ImageNetDataset(path=self.path, transforms=self.train_transforms, is_train=True),
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=imagenet_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(ImageNetDataset(path=self.path, transofrms=self.val_transforms, is_train=False),
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=imagenet_collate_fn,
        )


if __name__ == '__main__':
    '''
    Dataset Loadaer Test
    run$ python -m dataset.classification/ImageNet_1k
    '''
    import albumentations
    import albumentations.pytorch
    
    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        # albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2()
    ])

    loader = DataLoader(ImageNet(path='/mnt/YIS', transforms=train_transforms, is_train=True))

    for batch, sample in enumerate(loader):
        print('image : ', sample['image'])
        print('label : ', sample['label'])
        print('sample_id : ', sample['sample_id'])
        print('on_gpu : ', sample['on_gpu'])
        break
    
    