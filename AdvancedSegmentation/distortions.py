import numpy as np
import cv2
import albumentations as A

# Apply distortions of GaussianNoise, Blur, Brightness & Contrast and Partial Occlusion with a certain porbability to an image
# Returns a transform function that takes in images and labels
def distortions():
    # GaussNoise = Gaussian Noise
    # GaussianBlur = Blur
    # RandomBrightnessContrast = Brightness and Contrast
    # CoarseDropout = Partial Occlusion
    transform = A.Compose([A.GaussNoise(std_range=(0.1, 0.5), p=0.5), A.GaussianBlur(blur_limit=(3, 7), p=0.3), 
                           A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),
                           A.CoarseDropout(hole_height_range=(8,16),hole_width_range=(8,16),p=0.5)])
    return transform


# Post-processing to be used after distortions are applied
# Can also be used if distortions are not applied
# Takes in mask and returns mask
def post_process(mask):
    mask = cv2.bitwise_not(mask)
    mask = mask.astype(np.uint8)

    # Apply opening to remove salt and pepper noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Remove objects with small areas < 50 pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            mask[labels == i] = 0

    # medianBlur emoves salt and pepper noise but keeps edges
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.bitwise_not(mask)
    return mask