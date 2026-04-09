import sys

import cv2
sys.path.append('..')

import os
import numpy as np
import torch
from torchvision import transforms

IMAGE_SIZE = 350

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

def mask_transform(mask):

    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE))

    mask = np.array(mask)

    mask = mask / 255

    mask = torch.tensor(mask, dtype=torch.float32)

    return mask.unsqueeze(0)

base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

aug_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),

    transforms.ToTensor(),
])

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