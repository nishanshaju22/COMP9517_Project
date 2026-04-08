"""
features.py


Imported by model_rf.py, model_sgd.py, and model_xgb.py.
 
Features extracted per pixel:
    RGB (3) | HSV (3) | Lab (3) | ExG (1) | NDI (1) | LBP (1) | Sobel (1)
    = 13 features total
"""
 
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from pathlib import Path
from typing import Optional , Union
 

# Vegetation indices

 
def compute_ExG(img_rgb: np.ndarray) -> np.ndarray:
    """
    Excess Green Index: ExG = 2G - R - B.
    Normalises channels to [0, 1] before computing so the result is
    scale-invariant regardless of uint8 vs float input.
 
    Returns a (H, W) float32 array.
    """
    img = img_rgb.astype(np.float32) / 255.0
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    return (2.0 * G - R - B).astype(np.float32)
 
 
def compute_NDI(img_rgb: np.ndarray) -> np.ndarray:

    """
    Normalised Difference Index: NDI = (G - R) / (G + R + 1e-6).
    Useful companion to ExG — captures the green/red ratio directly.
 
    Returns a (H, W) float32 array in [-1, 1].
    """
    img = img_rgb.astype(np.float32) / 255.0
    R, G = img[..., 0], img[..., 1]
    return ((G - R) / (G + R + 1e-6)).astype(np.float32)
 
 

# Texture

 
def compute_LBP(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:

    """
    Local Binary Pattern on a single-channel (grayscale) image.
    Uses 'uniform' method which produces sparse, rotation-invariant codes.
 
    Args:
        gray:     (H, W) uint8 grayscale image.
        radius:   LBP neighbourhood radius (default 1 — compact & fast).
        n_points: Number of points in the LBP ring (default 8).
 
    Returns a (H, W) float32 array normalised to [0, 1].
    """

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_max = n_points + 2  # number of uniform patterns
    return (lbp / lbp_max).astype(np.float32)
 
 
def compute_Sobel(gray: np.ndarray) -> np.ndarray:

    """
    Sobel edge magnitude — useful as a secondary texture cue alongside LBP.
    Soil tends to have strong random edges; leaves have structured venation.
 
    Returns a (H, W) float32 array normalised to [0, 1].
    """

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag_max = mag.max()
    if mag_max > 0:
        mag /= mag_max
    return mag.astype(np.float32)
 
 
 
# Core feature extraction

 
def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a 13-dimensional feature vector for every pixel in an RGB image.
 
    Feature layout (column order):
        0  R       — red channel [0, 255]
        1  G       — green channel [0, 255]
        2  B       — blue channel [0, 255]
        3  H       — hue [0, 179] (OpenCV convention)
        4  S       — saturation [0, 255]
        5  V       — value [0, 255]
        6  L       — CIE L* (lightness) [0, 100]
        7  a       — CIE a* (greenred) [-128, 127]
        8  b       — CIE b* (blue–yellow) [-128, 127]
        9  ExG     — Excess Green Index (float32)
        10 NDI     — Normalised Difference Index (float32, [-1, 1])
        11 LBP     — Local Binary Pattern (float32, [0, 1])
        12 Sobel   — Sobel edge magnitude (float32, [0, 1])
 
    Args:
        img_rgb: (H, W, 3) uint8 RGB image.
 
    Returns:
        (H*W, 13) float32 feature matrix — one row per pixel.
    """
    H, W = img_rgb.shape[:2]
    N = H * W
 
    # Colour spaces 
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
 
    # Vegetation indices 
    exg = compute_ExG(img_rgb)   # (H, W)
    ndi = compute_NDI(img_rgb)   # (H, W)
 
    #  Texture 
    lbp = compute_LBP(gray)      # (H, W)
    sobel = compute_Sobel(gray)  # (H, W)
 
    #  Stack into (H*W, 13) 
    features = np.column_stack([
        img_rgb[:, :, 0].reshape(N),    # R
        img_rgb[:, :, 1].reshape(N),    # G
        img_rgb[:, :, 2].reshape(N),    # B
        img_hsv[:, :, 0].reshape(N),    # H
        img_hsv[:, :, 1].reshape(N),    # S
        img_hsv[:, :, 2].reshape(N),    # V
        img_lab[:, :, 0].reshape(N),    # L
        img_lab[:, :, 1].reshape(N),    # a
        img_lab[:, :, 2].reshape(N),    # b
        exg.reshape(N),                 # ExG
        ndi.reshape(N),                 # NDI
        lbp.reshape(N),                 # LBP
        sobel.reshape(N),               # Sobel
    ]).astype(np.float32)
 
    return features
 
 

# Stratified pixel sampler

 
def sample_pixels_stratified(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    n_per_class: int = 5000,
    rng: Optional[np.random.Generator] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample an equal number of wheat (1) and soil (0) pixels from one image.
    Stratification prevents class imbalance from dominating the training table.
 
    Args:
        img_rgb:      (H, W, 3) uint8 RGB image.
        mask:         (H, W) binary uint8 mask — 255 or 1 = wheat, 0 = soil.
        n_per_class:  How many pixels to sample from each class.
        rng:          Optional numpy Generator for reproducibility.
 
    Returns:
        X: (2*n_per_class, 13) float32 feature matrix.
        y: (2*n_per_class,)   int8 label vector  (0 = soil, 1 = wheat).
    """
    if rng is None:
        rng = np.random.default_rng()
 
    # Normalise mask to binary {0, 1}
    binary_mask = (mask > 0).astype(np.uint8)
 
    flat_features = extract_features(img_rgb)       # (H*W, 13)
    flat_labels = binary_mask.reshape(-1)            # (H*W,)
 
    wheat_idx = np.where(flat_labels == 1)[0]
    soil_idx  = np.where(flat_labels == 0)[0]
 
    # Sample with replacement if a class has fewer pixels than requested
    n_wheat = min(n_per_class, len(wheat_idx))
    n_soil  = min(n_per_class, len(soil_idx))
 
    sampled_wheat = rng.choice(wheat_idx, n_wheat, replace=False)
    sampled_soil  = rng.choice(soil_idx,  n_soil,  replace=False)
 
    idx = np.concatenate([sampled_wheat, sampled_soil])
    X = flat_features[idx]
    y = flat_labels[idx].astype(np.int8)
 
    return X, y
 
 

# Dataset loader

 
def load_image_mask_pair(
    img_path: Union[str, Path, None],
    mask_path: Union[str, Path,None],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an RGB image and its binary mask from disk.
 
    The EWS dataset stores masks as single-channel PNGs where plant pixels = 255.
    This function normalises the mask to {0, 1}.
 
    Returns:
        img_rgb:  (350, 350, 3) uint8 RGB array.
        mask:     (350, 350)    uint8 binary array — 1 = wheat, 0 = soil.
    """
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
 
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = (mask_raw > 0).astype(np.uint8)
 
    return img_rgb, mask
 
 
def build_training_table(
    img_paths: list[Path],
    mask_paths: list[Path],
    n_per_class_per_image: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Build a combined feature matrix and label vector from a list of
    image/mask pairs using stratified sampling.
 
    Args:
        img_paths:              Ordered list of image file paths.
        mask_paths:             Ordered list of corresponding mask file paths.
        n_per_class_per_image:  Pixels sampled per class per image.
        seed:                   Random seed for reproducibility.
 
    Returns:
        X: (N, 13) float32 — stacked features from all images.
        y: (N,)    int8    — stacked labels.
    """

    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
 
    for img_path, mask_path in zip(img_paths, mask_paths):
        img_rgb, mask = load_image_mask_pair(img_path, mask_path)
        X, y = sample_pixels_stratified(img_rgb, mask, n_per_class_per_image, rng)
        X_parts.append(X)
        y_parts.append(y)
 
    return np.vstack(X_parts), np.concatenate(y_parts)
 
 

# Feature names (for model inspection / SHAP)

 
FEATURE_NAMES = [
    "R", "G", "B",
    "H", "S", "V",
    "L", "a", "b",
    "ExG", "NDI", "LBP", "Sobel",
]
 