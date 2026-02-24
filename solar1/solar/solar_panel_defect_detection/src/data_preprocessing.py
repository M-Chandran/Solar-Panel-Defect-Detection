import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Try to import albumentations, but make it optional
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("Albumentations not available. Using basic augmentations.")
    A = None
    ToTensorV2 = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolarPanelDataset(Dataset):
    """
    Dataset class for solar panel images.
    Handles loading, preprocessing, and augmentation.
    """
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        """Get all image paths from data directory."""
        image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
        logger.info(f"Found {len(image_paths)} images in {self.data_dir}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_train:
            # For self-supervised learning, return two augmented views
            view1 = self._augment_image(image)
            view2 = self._augment_image(image)
            return view1, view2
        else:
            return image

    def _load_image(self, img_path):
        """Load and preprocess image."""
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize resolution (resize to 224x224 for ResNet)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Noise reduction using Gaussian blur
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Contrast enhancement using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return image

    def _augment_image(self, image):
        """Apply augmentations for self-supervised learning using ten-crop strategy."""
        # Convert to numpy array if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        
        # Apply ten-crop augmentation as per paper
        # First, apply ten-crop (5 crops + 5 mirrored crops)
        crops = self._ten_crop_augmentation(image)
        
        # Randomly select one crop
        selected_crop = crops[np.random.randint(len(crops))]
        
        # Apply additional augmentations: random horizontal and vertical flips
        if ALBUMENTATIONS_AVAILABLE:
            augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),  # Random horizontal flip
                A.VerticalFlip(p=0.5),    # Random vertical flip
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.GaussNoise(var_limit=(10, 50), p=0.1),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            augmented = augmentation(image=selected_crop)
            return augmented['image']
        else:
            # Fallback to torchvision transforms
            transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(selected_crop)

    def _ten_crop_augmentation(self, image, crop_size=224, scale_factor=1.2):
        """
        Ten-crop data augmentation as described in the paper.
        Creates 5 crops (4 corners + center) and their mirrored versions.
        
        Args:
            image: Input image (H x W x C)
            crop_size: Size of the crop (C x C)
            scale_factor: Scale factor for enlargement (default 1.2)
        
        Returns:
            list of 10 cropped images
        """
        h, w = image.shape[:2]
        
        # Calculate scaled size S = max(C, W, H) * scale
        S = int(max(crop_size, w, h) * scale_factor)
        
        # Resize image to S x S
        resized = cv2.resize(image, (S, S), interpolation=cv2.INTER_LINEAR)
        
        crops = []
        
        # Define 5 crop positions: 4 corners + center
        positions = [
            (0, 0),                    # Top-left
            (0, S - crop_size),        # Top-right
            (S - crop_size, 0),        # Bottom-left
            (S - crop_size, S - crop_size),  # Bottom-right
            ((S - crop_size) // 2, (S - crop_size) // 2)  # Center
        ]
        
        # Generate 5 crops
        for y, x in positions:
            crop = resized[y:y+crop_size, x:x+crop_size]
            crops.append(crop)
            # Add mirrored version (horizontal flip)
            mirrored = cv2.flip(crop, 1)
            crops.append(mirrored)
        
        return crops


def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size (default 64 as per paper)
        num_workers: Number of worker processes for data loading
    """
    # Define transforms
    train_transform = None  # Augmentations handled in dataset
    
    if ALBUMENTATIONS_AVAILABLE:
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        val_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Create datasets
    train_dataset = SolarPanelDataset(data_dir, transform=train_transform, is_train=True)
    val_dataset = SolarPanelDataset(data_dir, transform=val_transform, is_train=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for MoCo queue consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def preprocess_image(image_array):
    """
    Preprocess a single image array for inference.
    Applies resizing, noise reduction, contrast enhancement, and normalization.
    Returns a preprocessed tensor.
    """
    # Ensure image is numpy array
    if isinstance(image_array, torch.Tensor):
        image_array = image_array.permute(1, 2, 0).numpy()

    # Resize to 224x224
    image = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Noise reduction using Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Contrast enhancement using CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize and convert to tensor
    if ALBUMENTATIONS_AVAILABLE:
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        processed = transform(image=image)['image']
    else:
        # Basic normalization without albumentations
        image = image / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        processed = torch.from_numpy(image.transpose(2, 0, 1)).float()

    return processed

def generate_multi_view_images(image, num_views=4):
    """
    Generate multiple augmented views of an image for inference.
    """
    views = []
    for _ in range(num_views):
        augmented = A.Compose([
            A.RandomCrop(width=224, height=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])(image=image)['image']
        views.append(augmented)
    return torch.stack(views)

if __name__ == "__main__":
    # Example usage
    data_dir = "../data"
    if os.path.exists(data_dir):
        train_loader, val_loader = get_data_loaders(data_dir)
        logger.info(f"Train loader: {len(train_loader)} batches")
        logger.info(f"Val loader: {len(val_loader)} batches")
    else:
        logger.warning(f"Data directory {data_dir} not found. Please add solar panel images.")
