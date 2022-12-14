# Lib Load
import os
import cv2
import glob
import torch
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import Dataset, DataLoader


class ImageNetDataset(Dataset):
    def __init__(self, path, transforms, is_train):
        super(ImageNetDataset, self).__init__()
        self.transforms = transforms
        self.is_train = is_train
        with open(path + '/train.txt', 'r') as f:
            self.train_label_list = f.read().splitlines()
        
        with open(path + '/valid.txt', 'r') as f:
            self.valid_label_list = f.read().splitlines()

        if is_train:
            self.data = glob.glob(path + '/train/**/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-2] + (' ')
                self.train_list[data] = self.train_label_list.index(label)

    
        else:
            self.data = glob.glob(path + '/valid/**/*.JPEG')
            self.val_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-2] + (' ')
                self.val_list[data] = self.valid_label_list.index(label)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[img_file]
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
            #collate_fn=imagenet_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(ImageNetDataset(path=self.path, transforms=self.val_transforms, is_train=False),
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            #collate_fn=imagenet_collate_fn,
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
        albumentations.Normalize(0, 1),
        albumentations.Resize(256, 256, always_apply=True),
        albumentations.pytorch.ToTensorV2()
    ])

    loader = DataLoader(ImageNetDataset(path='/mnt', transforms=train_transforms, is_train=True))

    for batch, sample in enumerate(loader):
        # print('image : ', sample['img'])
        # print(sample['img'][0].shape)
        # print('class : ', sample['class'])
        pass
    