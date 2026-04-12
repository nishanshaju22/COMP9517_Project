import numpy as np
import glob
import os
import cv2
import albumentations as A

def get_stats(mask, label):
    tp = np.sum((mask == 0) & (label == 0)).astype(float)
    fp = np.sum((mask == 0) & (label == 255)).astype(float)
    fn = np.sum((mask == 255) & (label == 0)).astype(float)
    tn = np.sum((mask == 255) & (label == 255)).astype(float)
    acc = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    IoU = tp / (tp + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return acc, recall, precision, IoU, f1


# Function that outputs images and labels given an array of paths
def get_imgs(path):
    data = glob.glob(os.path.join(path, "*.png"))
    imgs = [x for x in data if "mask" not in x]
    labels = [x for x in data if "mask" in x]
    return imgs, labels

def distortions():
    # Distortions from 
    # GaussNoise = Gaussian Noise
    # GaussianBlur = Blur
    # RandomBrightnessContrast = Brightness and Contrast
    # CoarseDropout = Partial Occlusion
    transform = A.Compose([A.GaussNoise(std_range=(0.1, 0.5), p=0.5), A.GaussianBlur(blur_limit=(3, 7), p=0.3), 
                           A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),
                           A.CoarseDropout(hole_height_range=(8,16),hole_width_range=(8,16),p=0.5)])
    return transform