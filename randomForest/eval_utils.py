"""
eval_utils.py
evaluation and post-processing utilities.
Imported by model_rf.ipynb, model_sgd.ipynb, and model_xgb.ipynb.
"""
 
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Optional
from features import extract_features

# Post-processing

 
def morphological_cleanup(
    pred_mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 7,
) -> np.ndarray:
    """
    Remove salt-and-pepper noise from a raw pixel-wise prediction mask.
 
    Steps:
        1. Opening  (erosion → dilation) — removes small isolated false-positive
           blobs (misclassified soil pixels predicted as wheat).
        2. Closing  (dilation → erosion) — fills small holes inside wheat regions
           (misclassified pixels inside a leaf predicted as soil).
 
    Args:
        pred_mask:    (H, W) binary uint8 mask — 1 = wheat, 0 = soil.
        open_kernel:  Side length of the opening structuring element (px).
        close_kernel: Side length of the closing structuring element (px).
 
    Returns:
        Cleaned (H, W) binary uint8 mask.
    """
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel,  open_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
 
    opened = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_OPEN,  k_open)
    closed = cv2.morphologyEx(opened,                     cv2.MORPH_CLOSE, k_close)
    return closed
 
 

# Metrics

 
def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Intersection over Union for binary segmentation.
 
    IoU = TP / (TP + FP + FN)
 
    Args:
        y_true: Flat or 2-D ground-truth binary array.
        y_pred: Flat or 2-D predicted binary array.
 
    Returns:
        IoU score as a float in [0, 1].
    """
    y_true = y_true.reshape(-1).astype(bool)
    y_pred = y_pred.reshape(-1).astype(bool)
 
    intersection = np.logical_and(y_true, y_pred).sum()
    union        = np.logical_or(y_true,  y_pred).sum()
 
    if union == 0:
        return 1.0  
    return float(intersection / union)
 
 
def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    apply_cleanup: bool = False,
    mask_shape: Optional[tuple[int, int]] = None
) -> dict:
    """
    Compute all required project metrics for a single predicted mask.
 
    Metrics returned:
        precision, recall, f1, iou
 
    Args:
        y_true:        Ground-truth binary array (flat or 2-D).
        y_pred:        Predicted binary array (flat or 2-D).
        apply_cleanup: If True, run morphological_cleanup before scoring.
                       Requires mask_shape.
        mask_shape:    (H, W) needed to reshape for cleanup.
 
    Returns:
        dict with keys: precision, recall, f1, iou
    """
    y_true_flat = y_true.reshape(-1).astype(np.uint8)
    y_pred_flat = y_pred.reshape(-1).astype(np.uint8)
 
    if apply_cleanup:
        if mask_shape is None:
            raise ValueError("mask_shape must be provided when apply_cleanup=True")
        y_pred_2d   = y_pred_flat.reshape(mask_shape)
        y_pred_2d   = morphological_cleanup(y_pred_2d)
        y_pred_flat = y_pred_2d.reshape(-1)
 
    return {
        "precision": precision_score(y_true_flat, y_pred_flat, zero_division=0),
        "recall":    recall_score(y_true_flat,    y_pred_flat, zero_division=0),
        "f1":        f1_score(y_true_flat,         y_pred_flat, zero_division=0),
        "iou":       compute_iou(y_true_flat,      y_pred_flat),
    }
 
 
def evaluate_dataset(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    apply_cleanup: bool = False,
) -> dict:
    """
    Compute mean metrics across a list of image predictions.
 
    Args:
        gt_masks:      List of (H, W) ground-truth binary masks.
        pred_masks:    List of (H, W) predicted binary masks.
        apply_cleanup: Whether to apply morphological cleanup before scoring.
 
    Returns:
        dict with keys: precision, recall, f1, iou (all mean floats),
        plus 'per_image' — a list of per-image metric dicts.
    """
    per_image = []
    for gt, pred in zip(gt_masks, pred_masks):
        metrics = evaluate(
            gt, pred,
            apply_cleanup=apply_cleanup,
            mask_shape=gt.shape[:2],
        )
        per_image.append(metrics)
 
    keys = ["precision", "recall", "f1", "iou"]
    means = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
    means["per_image"] = per_image
    return means
 
 

# Mask reconstruction

 
def predictions_to_mask(
    y_pred: np.ndarray,
    H: int = 350,
    W: int = 350,
) -> np.ndarray:
    """
    Reshape a flat prediction array back into a (H, W) binary image.
 
    Args:
        y_pred: (H*W,) predicted label array from model.predict().
        H, W:   Image dimensions (default 350×350 for EWS).
 
    Returns:
        (H, W) uint8 binary mask — 1 = wheat, 0 = soil.
    """
    return y_pred.reshape(H, W).astype(np.uint8)
 
def predict_on_image(model, img_rgb, apply_cleanup = False):
    X = extract_features(img_rgb)
    y_pred = model.predict(X)
    mask = predictions_to_mask(y_pred, 350, 350)
    if apply_cleanup:
        mask = morphological_cleanup(mask)
    return mask



 
def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Print a formatted metrics summary to stdout.
 
    Args:
        metrics:    Dict returned by evaluate() or evaluate_dataset().
        model_name: Label shown in the header.
    """
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    for key in ["precision", "recall", "f1", "iou"]:
        if key in metrics:
            print(f"  {key.capitalize():<12} {metrics[key]:.4f}")
    print(f"{'='*40}\n")
 