import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from utils import rle2mask

LABEL_MAP = {'Fish': 0, 'Flower': 1, 'Gravel': 2, 'Sugar': 3}

# DINOv3 SAT-493M normalization (satellite imagery)
MEAN = [0.430, 0.411, 0.296]
STD = [0.213, 0.156, 0.143]


def get_train_transforms():
    h, w = config.IMG_SIZE  # (512, 768)
    return A.Compose([
        A.Resize(height=h, width=w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    h, w = config.IMG_SIZE
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_test_transforms():
    return get_val_transforms()


class CloudDataset(Dataset):
    def __init__(self, csv_path, fold, split='train', transform=None):
        """
        Args:
            csv_path: path to train_fold.csv
            fold: which fold to use as validation
            split: 'train' or 'val' or 'test'
            transform: albumentations transform
        """
        self.split = split
        self.transform = transform

        if split == 'test':
            sub_csv = os.path.join(config.DATA_DIR, 'sample_submission.csv')
            df = pd.read_csv(sub_csv)
            df['Image'] = df['Image_Label'].map(lambda v: v[:v.find('_')])
            df['Label'] = df['Image_Label'].map(lambda v: v[v.find('_') + 1:])
            df['LabelIndex'] = df['Label'].map(lambda v: LABEL_MAP[v])
            df['EncodedPixels'] = df['EncodedPixels'].fillna('')
            self.images_dir = os.path.join(config.DATA_DIR, 'test_images')
        else:
            df = pd.read_csv(csv_path)
            df['EncodedPixels'] = df['EncodedPixels'].fillna('')
            if split == 'train':
                df = df[df['Fold'] != fold]
            elif split == 'val':
                df = df[df['Fold'] == fold]
            self.images_dir = os.path.join(config.DATA_DIR, 'train_images')

        self.df = df.set_index('Image')
        self.image_ids = list(self.df.index.unique())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.images_dir, image_id)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, _ = image.shape

        # Build 4-channel mask
        rows = self.df.loc[image_id]
        mask = np.zeros((H, W, 4), dtype=np.uint8)
        if isinstance(rows, pd.Series):
            # Single row (shouldn't happen for properly structured data)
            cls_idx = int(rows['LabelIndex'])
            encoded = rows['EncodedPixels']
            if encoded != '':
                mask[:, :, cls_idx] = rle2mask(H, W, encoded)
        else:
            for _, row in rows.iterrows():
                cls_idx = int(row['LabelIndex'])
                encoded = row['EncodedPixels']
                if encoded != '':
                    mask[:, :, cls_idx] = rle2mask(H, W, encoded)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']       # (3, H, W) tensor
            mask = transformed['mask']         # (H, W, 4) tensor
            mask = mask.permute(2, 0, 1).float()  # (4, H, W)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return {'image_id': image_id, 'image': image, 'mask': mask}


def build_dataloaders(fold):
    train_ds = CloudDataset(
        config.TRAIN_CSV, fold=fold, split='train',
        transform=get_train_transforms(),
    )
    val_ds = CloudDataset(
        config.TRAIN_CSV, fold=fold, split='val',
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader


def build_test_dataloader():
    test_ds = CloudDataset(
        csv_path=None, fold=None, split='test',
        transform=get_test_transforms(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return test_loader
