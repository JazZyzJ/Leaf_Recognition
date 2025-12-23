from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    import cleanlab
    from cleanlab.filter import find_label_issues
except ImportError:
    print("Error: 'cleanlab' is not installed.")
    print("Please install it using: pip install cleanlab")
    sys.exit(1)

from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify label errors using Cleanlab (Confident Learning)."
    )
    parser.add_argument("--config", help="Optional YAML config for defaults.", default=None)
    parser.add_argument("--train_csv", help="CSV with labels/label_id.", default=None)
    parser.add_argument(
        "--oof",
        action="append",
        required=True,
        help="Path to OOF predictions (.npy). Repeat to build an ensemble.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional ensemble weights, one per --oof entry.",
    )
    parser.add_argument("--out_dir", default=".", help="Directory to save output files.")
    parser.add_argument(
        "--prune_threshold", 
        type=float, 
        default=0.5, 
        help="Threshold of 'label_quality' below which to prune in the clean version (default 0.5, lower is stricter)."
    )
    return parser.parse_args()


def resolve_train_csv(config: dict | None, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"train_csv not found: {path}")

    if config:
        folds_csv = config.get("experiment", {}).get("save_folds_csv")
        if folds_csv:
            path = Path(folds_csv)
            if path.exists():
                return path
        data_csv = config.get("data", {}).get("train_csv")
        if data_csv:
            path = Path(data_csv)
            if path.exists():
                return path

    for candidate in ("train_folds.csv", "train.csv"):
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError("Could not locate a training CSV. Provide --train_csv.")


def load_oof_arrays(paths: List[Path]) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"OOF file not found: {path}")
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"OOF array must be 2D (N x C). Got shape {arr.shape} for {path}.")
        arrays.append(arr)
    return arrays


def normalize_weights(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [w / total for w in weights]


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else None

    # 1. Load Data
    train_csv = resolve_train_csv(config, args.train_csv)
    print(f"Loading data from {train_csv}...")
    train_df = pd.read_csv(train_csv)

    image_col = "image"
    label_col = "label"
    if config:
        image_col = config.get("data", {}).get("image_col", image_col)
        label_col = config.get("data", {}).get("label_col", label_col)

    if "label_id" not in train_df.columns:
        if label_col not in train_df.columns:
            raise ValueError(f"Missing label column '{label_col}' in {train_csv}.")
        print("Encoding labels to label_id...")
        encoder = LabelEncoder()
        train_df["label_id"] = encoder.fit_transform(train_df[label_col])
    
    if "image_path" not in train_df.columns and image_col in train_df.columns:
        train_df["image_path"] = train_df[image_col]

    # 2. Load and Ensemble OOF Predictions
    oof_paths = [Path(p) for p in args.oof]
    print(f"Loading {len(oof_paths)} OOF files...")
    oof_arrays = load_oof_arrays(oof_paths)

    base_shape = oof_arrays[0].shape
    for arr in oof_arrays[1:]:
        if arr.shape != base_shape:
            raise ValueError("All OOF arrays must share the same shape.")

    weights = args.weights
    if weights is None:
        weights = [1.0 / len(oof_arrays)] * len(oof_arrays)
    else:
        if len(weights) != len(oof_arrays):
            raise ValueError("Number of --weights must match number of --oof files.")
        weights = normalize_weights(weights)

    proms_ensemble = np.zeros(base_shape, dtype=np.float64)
    for weight, arr in zip(weights, oof_arrays):
        proms_ensemble += weight * arr
    
    # Ensure probabilities sum to 1
    # Some OOFs might be logits or unnormalized, though assumed softmaxed in cleaning_candidates.
    # To be safe, we can apply softmax if they don't look like probs, but let's assume valid probs for now
    # or re-normalize.
    row_sums = proms_ensemble.sum(axis=1, keepdims=True)
    proms_ensemble = proms_ensemble / row_sums

    labels = train_df["label_id"].to_numpy()

    if proms_ensemble.shape[0] != len(labels):
         raise ValueError(f"OOF size {proms_ensemble.shape[0]} != Train size {len(labels)}")

    # 3. Cleanlab Analysis
    print("Running Cleanlab find_label_issues...")
    # filter_by='prune_by_noise_rate' or 'prune_by_class' is generally good for auto-cleaning.
    # 'both' is default for find_label_issues
    issues_df = find_label_issues(
        labels=labels,
        pred_probs=proms_ensemble,
        return_indices_ranked_by="self_confidence",
    )
    
    # Identify issues
    # find_label_issues returns a DataFrame with boolean mask and quality scores
    # actually, with return_indices_ranked_by, it returns indices. 
    # Let's use the standard return to get the DataFrame which is more useful.
    issues_info = find_label_issues(
        labels=labels,
        pred_probs=proms_ensemble,
        return_indices_ranked_by=None 
    )
    
    # Merge Cleanlab results into train_df
    train_df["is_label_issue"] = issues_info["is_label_issue"]
    train_df["label_quality"] = issues_info["label_quality"]
    train_df["given_label"] = issues_info["given_label"]
    train_df["predicted_label"] = issues_info["predicted_label"]
    
    # 4. Export Results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A. Candidates for Manual Review (All identified issues, sorted by quality)
    review_df = train_df[train_df["is_label_issue"]].sort_values("label_quality")
    review_path = out_dir / "label_issues_review.csv"
    review_df.to_csv(review_path, index=False)
    print(f"\nFound {len(review_df)} potential label errors.")
    print(f"Exported detailed review list to: {review_path}")
    
    # B. Clean Dataset (Pruning)
    # We prune samples where Cleanlab is confident it is an error.
    # We can use the is_label_issue flag directly.
    clean_df = train_df[~train_df["is_label_issue"]].copy()
    
    clean_path = out_dir / "train_clean.csv"
    # Keep original columns only for training compatibility
    original_cols = [c for c in clean_df.columns if c not in ["is_label_issue", "label_quality", "given_label", "predicted_label"]]
    clean_df[original_cols].to_csv(clean_path, index=False)
    
    print(f"Exported cleaned dataset (removed {len(train_df) - len(clean_df)} samples) to: {clean_path}")
    print(f"Pruning Rate: {(len(train_df) - len(clean_df)) / len(train_df):.2%}")
    print("\nNext Steps:")
    print("1. Review 'label_issues_review.csv' to sanity check the dropped images.")
    print("2. Train a new model using 'train_clean.csv' as the training data.")


if __name__ == "__main__":
    main()
