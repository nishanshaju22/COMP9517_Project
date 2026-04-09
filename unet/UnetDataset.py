import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EWSDataset(Dataset):
    """EWS (Eschikon Wheat Segmentation) dataset loader."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith("6.png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".png", "_mask.png")

        image = np.array(Image.open(os.path.join(self.image_dir, img_name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.image_dir, mask_name)).convert("L"))

        # Binarise mask: plant=1, soil=0
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask


def get_transforms(split="train", img_size=320):
    """Return augmentation pipeline for each split."""
    if split == "train":
        return A.Compose([
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.PadIfNeeded(img_size, img_size, border_mode=0),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])