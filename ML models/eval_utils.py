"""
eval_utils.py
evaluation and post-processing utilities.
"""
 
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Optional
from features import extract_features

# Remove salt-and-pepper noise from a raw pixel-wise prediction mask.
def morphological_cleanup(
    pred_mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 7,
) -> np.ndarray:
   
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel,  open_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
 
    opened = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_OPEN,  k_open)
    closed = cv2.morphologyEx(opened,                     cv2.MORPH_CLOSE, k_close)
    return closed
 
# Metrics
def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  
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
    
    y_true_flat = y_true.reshape(-1).astype(np.uint8)
    y_pred_flat = y_pred.reshape(-1).astype(np.uint8)

    if apply_cleanup:
        y_pred_2d = y_pred_flat.reshape(mask_shape)
        y_pred_2d = morphological_cleanup(y_pred_2d)
        y_pred_flat = y_pred_2d.reshape(-1)

    return {
        "precision": precision_score(y_true_flat, y_pred_flat, zero_division=0),
        "recall":    recall_score(y_true_flat,    y_pred_flat, zero_division=0),
        "f1":        f1_score(y_true_flat,         y_pred_flat, zero_division=0),
        "iou":       compute_iou(y_true_flat,      y_pred_flat),
    }
 
 #Compute mean metrics across a list of image predictions.
def evaluate_dataset(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    apply_cleanup: bool = False,
) -> dict:
   
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
 
 

#   Reshape a flat prediction array back into a (H, W) binary image.
def  reshape_mask(
    y_pred: np.ndarray,
    H: int = 350,
    W: int = 350,
) -> np.ndarray:
    return y_pred.reshape(H, W).astype(np.uint8)
 
def predict_on_image(model, img_rgb, apply_cleanup = False):
    X = extract_features(img_rgb)
    y_pred = model.predict(X)
    mask = reshape_mask(y_pred, 350, 350)
    if apply_cleanup:
        mask = morphological_cleanup(mask)
    return mask

#  Extract features from a full image and return a (H, W) predicted mask.
def predict_mask(model, img_rgb: np.ndarray) -> np.ndarray:

    H, W = img_rgb.shape[:2]
    X = extract_features(img_rgb)          
    y_pred = model.predict(X)              
    return reshape_mask(y_pred, H, W)


def print_metrics(metrics: dict, model_name: str = "Model") -> None:

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    for key in ["precision", "recall", "f1", "iou"]:
        if key in metrics:
            print(f"  {key.capitalize():<12} {metrics[key]:.4f}")
    print(f"{'='*40}\n")
 