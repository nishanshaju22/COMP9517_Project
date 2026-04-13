"""
train_utils.py

Shared training utilities for model_rf.ipynb and hyperparam_rf.ipynb.
Handles file discovery, trial running, and result saving.
"""

import json
import time
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple

from features import extract_features, load_image_mask_pair, build_training_table
from eval_utils import evaluate, reshape_mask, morphological_cleanup





def get_image_mask_pairs(directory: Path) -> Tuple[List[Path], List[Path]]:
    """
    Discover all image/mask pairs in a directory.
    Masks are identified by the '_mask' suffix before .png.

    Args:
        directory: Path to train/, validation/, or test/ folder.

    Returns:
        img_paths, mask_paths — matched and sorted lists.
    """
    all_pngs = sorted(directory.glob("*.png"))
    mask_paths = [p for p in all_pngs if p.stem.endswith("_mask")]
    img_paths = []
    for mask_path in mask_paths:
        img_stem = mask_path.stem[: -len("_mask")]
        img_path = directory / f"{img_stem}.png"
        img_paths.append(img_path)
    return img_paths, mask_paths



# Dataset builder

def build_train_val_tables(
    train_dir: Path,
    val_dir: Path,
    n_per_class: int = 5000,
    seed: int = 42,
):
    train_img_paths, train_mask_paths = get_image_mask_pairs(train_dir)
    val_img_paths,   val_mask_paths   = get_image_mask_pairs(val_dir)

    print(f"Train: {len(train_img_paths)} images")
    print(f"Val:   {len(val_img_paths)} images")
    print("Building training table...")

    X_train, y_train = build_training_table(
        train_img_paths, train_mask_paths,
        n_per_class_per_image=n_per_class,
        seed=seed,
    )
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    return X_train, y_train, val_img_paths, val_mask_paths


# Trial runner


def run_trial(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_img_paths: List[Path],
    val_mask_paths: List[Path],
    apply_cleanup: bool = True,
) -> dict:
    """
    Train a RandomForestClassifier with given params and evaluate on val set.

    Args:
        params:          RF hyperparameters (passed directly to sklearn).
        X_train:         Training feature matrix.
        y_train:         Training labels.
        val_img_paths:   Validation image paths.
        val_mask_paths:  Validation mask paths.
        apply_cleanup:   Whether to apply morphological cleanup before scoring.

    Returns:
        dict with mean_iou, std_iou, train_time_s, inference_time_s.
    """
    model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    ious = []
    t1 = time.time()
    for img_path, mask_path in zip(val_img_paths, val_mask_paths):
        img_rgb, gt_mask = load_image_mask_pair(img_path, mask_path)
        X = extract_features(img_rgb)
        y_pred = model.predict(X)
        pred_mask = reshape_mask(y_pred, 350, 350)
        if apply_cleanup:
            pred_mask = morphological_cleanup(pred_mask)
        m = evaluate(gt_mask, pred_mask, apply_cleanup=False)
        ious.append(m["iou"])
    inference_time = time.time() - t1

    return {
        "mean_iou":         round(float(np.mean(ious)), 6),
        "std_iou":          round(float(np.std(ious)), 6),
        "train_time_s":     round(train_time, 3),
        "inference_time_s": round(inference_time, 3),
    }



# Results saver


def save_hyperparam_results(all_results: list, output_path: str = "rf_hyperparam_results.json") -> dict:
    """
    Find the best trial, save all results to JSON, and return the best entry.

    Args:
        all_results:  List of trial result dicts from run_trial.
        output_path:  Where to save the JSON.

    Returns:
        best trial dict.
    """
    best = max(all_results, key=lambda x: x["mean_iou"])

    output = {
        "best_hypothesis": best["hypothesis"],
        "best_params":     best["params"],
        "best_iou":        best["mean_iou"],
        "rationale":       best["rationale"],
        "n_trials":        len(all_results),
        "all_trials":      all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest: {best['hypothesis']}  IoU={best['mean_iou']:.4f}")
    print(f"Saved: {output_path}")
    return best



# Params loader (used by model_rf.ipynb)


def load_best_params(json_path: str = "rf_hyperparam_results.json") -> dict:
    """
    Load the best hyperparameters from a saved JSON file.

    Args:
        json_path: Path to the hyperparam results JSON.

    Returns:
        dict of RF params ready to pass to RandomForestClassifier.
    """
    with open(json_path) as f:
        data = json.load(f)
    print(f"Loaded params from: {data['best_hypothesis']}  (IoU={data['best_iou']:.4f})")
    return data["best_params"]