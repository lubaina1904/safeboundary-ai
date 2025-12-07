"""
PyTorch Dataset for Bladder Segmentation
Handles loading and augmentation of training data
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class BladderSegmentationDataset(Dataset):
    """
    Dataset for bladder segmentation training
    """
    def __init__(self, 
                 images_dir,
                 masks_dir=None,
                 transform=None,
                 image_size=512):
        """
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks (if None, assumes same as images_dir)
            transform: Albumentations transform
            image_size: Size to resize images to
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else self.images_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                 list(self.images_dir.glob('*.png')))
        
        # Filter to only images that have corresponding masks
        self.image_files = [f for f in self.image_files 
                          if not f.stem.endswith('_mask') and self._has_mask(f)]
        
        print(f"Found {len(self.image_files)} image-mask pairs")
    
    def _has_mask(self, image_path):
        """Check if mask exists for image"""
        mask_path = self.masks_dir / f"{image_path.stem}_mask.png"
        return mask_path.exists()
    
    def _get_mask_path(self, image_path):
        """Get mask path for image"""
        return self.masks_dir / f"{image_path.stem}_mask.png"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self._get_mask_path(image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Normalize mask to 0-1
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure correct shape
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
            'mask_path': str(mask_path)
        }


def get_training_augmentation(config):
    """
    Get training augmentation pipeline
    """
    return A.Compose([
        # Geometric transformations
        A.Rotate(limit=config.get('rotation_limit', 30), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        
        # Optical transformations (simulate endoscope variations)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.CLAHE(clip_limit=4.0, p=0.5),
        
        # Simulate surgical obstructions
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.4),
        
        # Simulate blood/smoke
        A.CoarseDropout(
            max_holes=8,
            max_height=50,
            max_width=50,
            min_holes=1,
            min_height=20,
            min_width=20,
            fill_value=0,
            p=0.3
        ),
        
        A.RandomFog(
            fog_coef_lower=0.1,
            fog_coef_upper=0.3,
            alpha_coef=0.1,
            p=0.2
        ),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation():
    """
    Get validation augmentation (minimal)
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_dataloaders(config):
    """
    Create train and validation dataloaders
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader
    """
    # Paths
    data_dir = config.get('data_dir', 'data/annotations')
    pseudo_labels_dir = config.get('pseudo_labels_dir', None)
    
    # Get all images
    all_images = list(Path(data_dir).glob('*.jpg')) + list(Path(data_dir).glob('*.png'))
    all_images = [f for f in all_images if not f.stem.endswith('_mask')]
    
    # Add pseudo-labeled images if available
    if pseudo_labels_dir and Path(pseudo_labels_dir).exists():
        pseudo_images = list(Path(pseudo_labels_dir).glob('*.jpg')) + \
                       list(Path(pseudo_labels_dir).glob('*.png'))
        pseudo_images = [f for f in pseudo_images if not f.stem.endswith('_mask')]
        all_images.extend(pseudo_images)
        print(f"Added {len(pseudo_images)} pseudo-labeled images")
    
    # Split into train/val
    val_split = config.get('val_split', 0.15)
    n_val = int(len(all_images) * val_split)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(all_images))
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_images = [all_images[i].parent for i in train_indices]
    val_images = [all_images[i].parent for i in val_indices]
    
    # Create datasets
    train_dataset = BladderSegmentationDataset(
        images_dir=data_dir,
        transform=get_training_augmentation(config),
        image_size=config.get('image_size', 512)
    )
    
    val_dataset = BladderSegmentationDataset(
        images_dir=data_dir,
        transform=get_validation_augmentation(),
        image_size=config.get('image_size', 512)
    )
    
    # Filter datasets by indices
    train_dataset.image_files = [train_dataset.image_files[i] for i in train_indices]
    val_dataset.image_files = [val_dataset.image_files[i] for i in val_indices]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    config = {
        'data_dir': 'data/annotations',
        'batch_size': 4,
        'image_size': 512,
        'val_split': 0.15,
        'num_workers': 2
    }
    
    train_loader, val_loader = create_dataloaders(config)
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch images shape: {batch['image'].shape}")
    print(f"Batch masks shape: {batch['mask'].shape}")
    print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"Mask range: [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")