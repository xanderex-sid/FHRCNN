import os
import random
from PIL import Image
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


import numpy as np

class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, hr_image_folder, set_type, hr_img_size, lr_img_size, color_channels, downsample_mode):
        self.hr_image_folder = hr_image_folder
        self.set_type = set_type
        self.hr_img_size = hr_img_size
        self.lr_img_size = lr_img_size
        self.color_channels = color_channels
        self.downsample_mode = downsample_mode
        self.image_fns = [f for f in os.listdir(hr_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split the dataset based on set_type (train, val)
        if set_type == "train":
            self.image_fns = self.image_fns[:-200]
        elif set_type == "val":
            self.image_fns = self.image_fns[-200:-100]
        else:
            self.image_fns = self.image_fns[-100:]

        # Transformations (could include cropping, flipping, etc.)
        self.transform = A.Compose(
            [
                A.RandomCrop(width=hr_img_size[0], height=hr_img_size[1], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.8),
                A.ToFloat(max_value=255)
            ]
        )

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        image_fn = self.image_fns[idx]
        hr_image_pil = Image.open(os.path.join(self.hr_image_folder, image_fn))
        hr_image = np.array(hr_image_pil)

        # Apply transformations to the HR image
        hr_image = self.transform(image=hr_image)["image"]
        resize_transform = transforms.Resize((20, 20))

        # Generate LR image by resizing HR image
        lr_image_pil = hr_image_pil.resize(self.lr_img_size, resample=self.downsample_mode)
        lr_image = np.array(lr_image_pil)

        # Normalize to float32
        hr_image = hr_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        lr_image = lr_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Ensure the images have the correct channel order
        hr_image = np.transpose(hr_image, (2, 0, 1))  # Convert to (C, H, W)
        lr_image = np.transpose(lr_image, (2, 0, 1))  # Convert to (C, H, W)

        # Return LR and HR images as tensors
        return torch.tensor(lr_image), resize_transform(torch.tensor(hr_image))
