import sys

import cv2
sys.path.append('..')

import os
import numpy as np
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 350
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_image_mask_pairs(folder):

    image_files = []
    mask_files = []

    for file in os.listdir(folder):

        if file.endswith(".png") and "_mask" not in file:

            image_path = os.path.join(folder, file)

            mask_name = file.replace(".png", "_mask.png")
            mask_path = os.path.join(folder, mask_name)

            if os.path.exists(mask_path):

                image_files.append(image_path)
                mask_files.append(mask_path)

    return image_files, mask_files

def add_gaussian_noise(image):

    image = image.numpy()

    noise = np.random.normal(
        0,
        0.05,
        image.shape
    )

    noisy = image + noise

    noisy = np.clip(noisy, 0, 1)

    return torch.tensor(noisy, dtype=torch.float32)

def add_motion_blur(image):

    img = image.numpy()

    kernel_size = 7

    kernel = np.zeros(
        (kernel_size, kernel_size)
    )

    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)

    kernel = kernel / kernel_size

    blurred = cv2.filter2D(
        img.transpose(1,2,0),
        -1,
        kernel
    )

    blurred = blurred.transpose(2,0,1)

    return torch.tensor(
        blurred,
        dtype=torch.float32
    )
    
def reduce_brightness(image):

    image = image * 0.4

    return image

def make_transforms(augment=True, img_size=350):

    val_tf = A.Compose([
        A.Resize(img_size, img_size),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2(),
    ])

    if not augment:
        return val_tf, val_tf

    train_tf = A.Compose([

        A.RandomCrop(img_size, img_size),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.3),

        A.Rotate(limit=15, p=0.5),

        A.ColorJitter(
            brightness=0.3,
            contrast=0.2,
            saturation=0.25,
            hue=0.0,
            p=0.5
        ),

        A.GaussNoise(
            var_limit=(10.0, 40.0),
            p=0.2
        ),

        A.GaussianBlur(
            blur_limit=3,
            p=0.2
        ),

        # Upscale trick
        A.Resize(img_size * 2, img_size * 2),

        A.RandomCrop(img_size, img_size),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2(),
    ])

    return train_tf, val_tf
