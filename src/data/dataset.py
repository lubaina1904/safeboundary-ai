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
                 masks_dir,
                 transform=None,
                 image_size=512):
        """
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            transform: Albumentations transform
            image_size: Size to resize images to
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Load images (jpg + png)
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )

        # Remove mask files
        self.image_files = [f for f in self.image_files if not f.stem.endswith("_mask")]

        # Keep only images with masks
        self.image_files = [f for f in self.image_files if self._has_mask(f)]

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
        
        # Normalize mask
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Add channel if needed
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path)
        }


def get_training_augmentation(config):
    return A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        
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
        
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.4),
        
        # Remove deprecated params
        A.CoarseDropout(max_holes=8, p=0.3),
        A.RandomFog(p=0.2),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation():
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
    """
    data_dir = Path(config['data_dir'])
    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Image or mask directory not found: {image_dir}, {mask_dir}")

    # Split into train/val
    val_split = config.get("val_split", 0.15)
    all_images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    all_images = [f for f in all_images if (mask_dir / f"{f.stem}_mask.png").exists()]
    
    n_val = int(len(all_images) * val_split)
    val_images = all_images[:n_val]
    train_images = all_images[n_val:]

    # Create datasets
    train_dataset = BladderSegmentationDataset(
        images_dir=image_dir,
        masks_dir=mask_dir,
        transform=get_training_augmentation(config),
        image_size=config.get('image_size', 512)
    )
    
    val_dataset = BladderSegmentationDataset(
        images_dir=image_dir,
        masks_dir=mask_dir,
        transform=get_validation_augmentation(),
        image_size=config.get('image_size', 512)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader

