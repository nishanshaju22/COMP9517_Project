"""
distortions.py

Image distortion utilities for robustness stress-testing.
Imported by model_rf.ipynb, model_sgd.ipynb, and model_xgb.ipynb.

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
    """
    Apply a realistic image distortion to an RGB image for stress-testing.

    Used to evaluate how robust a trained model is to real-world degradation
    (e.g. camera blur, low light, sensor noise, partial occlusion).
    Masks are NOT distorted — the ground truth stays the same, only the input
    image changes.

    Args:
        img_rgb:    (H, W, 3) uint8 RGB image.
        distortion: One of 'blur', 'noise', 'brightness', 'occlusion'.
        severity:   Integer 1-3 controlling how strong the distortion is.
                    1 = mild, 2 = moderate, 3 = severe.

    Returns:
        (H, W, 3) uint8 distorted RGB image.

    Distortion details:
        blur       — Gaussian blur. Simulates out-of-focus or motion blur.
                     Severity maps to kernel sizes 7, 15, 25.
        noise      — Gaussian noise added to all channels.
                     Severity maps to std devs 15, 30, 50.
        brightness — Reduces overall brightness by scaling pixel values down.
                     Severity maps to scale factors 0.7, 0.5, 0.3.
        occlusion  — Blacks out a random rectangular patch.
                     Severity maps to patch sizes 60x60, 100x100, 150x150.
    """
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
            "Choose from: 'blur', 'noise', 'brightness', 'occlusion'."
        )

    return img
