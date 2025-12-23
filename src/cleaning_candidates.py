from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ranked data-cleaning candidates from OOF predictions."
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
    parser.add_argument("--top_k", type=int, default=200, help="Number of samples to export.")
    parser.add_argument(
        "--keep_all",
        action="store_true",
        help="Export all samples instead of only top_k.",
    )
    parser.add_argument("--output_csv", default="cleaning_candidates.csv", help="Output CSV path.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Clamp for log loss.")
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

    train_csv = resolve_train_csv(config, args.train_csv)
    train_df = pd.read_csv(train_csv)

    image_col = "image"
    label_col = "label"
    if config:
        image_col = config.get("data", {}).get("image_col", image_col)
        label_col = config.get("data", {}).get("label_col", label_col)

    if "label_id" not in train_df.columns:
        if label_col not in train_df.columns:
            raise ValueError(f"Missing label column '{label_col}' in {train_csv}.")
        encoder = LabelEncoder()
        train_df["label_id"] = encoder.fit_transform(train_df[label_col])

    if "image_path" not in train_df.columns and image_col in train_df.columns:
        train_df["image_path"] = train_df[image_col]

    oof_paths = [Path(p) for p in args.oof]
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

    probs = np.zeros(base_shape, dtype=np.float64)
    for weight, arr in zip(weights, oof_arrays):
        probs += weight * arr

    label_ids = train_df["label_id"].to_numpy()
    if probs.shape[0] != label_ids.shape[0]:
        raise ValueError(
            f"OOF rows ({probs.shape[0]}) do not match train rows ({label_ids.shape[0]})."
        )

    p_true = probs[np.arange(len(label_ids)), label_ids]
    p_true = np.clip(p_true, args.epsilon, 1.0)
    losses = -np.log(p_true)

    order = np.argsort(-losses)
    if not args.keep_all:
        order = order[: args.top_k]

    output = train_df.iloc[order].copy()
    output["p_true"] = p_true[order]
    output["loss"] = losses[order]
    output["rank"] = np.arange(1, len(order) + 1)
    output["oof_sources"] = " | ".join(str(p) for p in oof_paths)
    output["oof_weights"] = " | ".join(f"{w:.6f}" for w in weights)
    output["decision"] = ""
    output["notes"] = ""

    output_path = Path(args.output_csv)
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output)} candidates to {output_path}")


if __name__ == "__main__":
    main()
