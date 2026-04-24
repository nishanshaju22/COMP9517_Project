"""
distortions.py

Image distortion utilities for robustness stress-testing.
Masks are NEVER distorted — ground truth stays clean.
Only the input RGB image is modified before feature extraction.
"""

import cv2
import numpy as np

DISTORTION_TYPES = ["blur", "noise", "brightness", "occlusion"]
SEVERITY_LEVELS  = [1, 2, 3]

def apply_distortions(
    img_rgb: np.ndarray,
    distortion: str = "blur",
    severity: int = 2,
) -> np.ndarray:
   
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")

    img = img_rgb.copy()

    if distortion == "blur":
        ksize = {1: 7, 2: 15, 3: 25}[severity]
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    elif distortion == "noise":
        std = {1: 15, 2: 30, 3: 50}[severity]
        noise = np.random.normal(0, std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif distortion == "brightness":
        scale = {1: 0.7, 2: 0.5, 3: 0.3}[severity]
        img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    elif distortion == "occlusion":
        patch = {1: 60, 2: 100, 3: 150}[severity]
        H, W = img.shape[:2]
        x = np.random.randint(0, W - patch)
        y = np.random.randint(0, H - patch)
        img[y:y + patch, x:x + patch] = 0

    else:
        raise ValueError(
            f"Unknown distortion '{distortion}'. "
        )

    return img
