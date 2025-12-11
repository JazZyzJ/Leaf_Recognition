from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import LeafDataset
from models import create_model
from train import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from utils import load_config, prepare_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Leaf Classification")
    parser.add_argument("--config", required=True, help="Config used during training")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument(
        "--weights_pattern",
        default=None,
        help="Optional glob or comma-separated list of weight files. Default uses training convention.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Optional custom suffix for submission file name",
    )
    return parser.parse_args()


def load_label_names(config: Dict, logs_dir: Path, experiment_name: str) -> List[str]:
    mapping_path = logs_dir / f"{experiment_name}_label_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["classes"]
    # fallback to fitting LabelEncoder again if mapping missing
    train_csv = Path(config["data"]["train_csv"])
    train_df = pd.read_csv(train_csv)
    labels = sorted(train_df[config["data"].get("label_col", "label")].unique())
    return labels


def build_test_loader(test_df: pd.DataFrame, config: Dict) -> DataLoader:
    img_size = config["training"]["img_size"]
    _, valid_transform = get_transforms(img_size)
    dataset = LeafDataset(
        test_df,
        image_root=config["data"].get("image_root", "."),
        transform=valid_transform,
        image_col="image_path",
        label_col=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def collect_weight_paths(args: argparse.Namespace, config: Dict, dirs: Dict[str, Path]) -> List[Path]:
    if args.weights_pattern:
        if "," in args.weights_pattern:
            return [Path(p.strip()) for p in args.weights_pattern.split(",")]
        return [Path(p) for p in Path().glob(args.weights_pattern)]
    folds = config["experiment"].get("num_folds", 5)
    experiment_name = config["experiment"]["name"]
    paths = []
    for fold in range(folds):
        paths.append(Path(dirs["weights_dir"]) / f"{experiment_name}_fold{fold}.pth")
    return paths


def infer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dirs = prepare_dirs(config)
    logs_dir = dirs.get("logs_dir", Path("logs"))
    experiment_name = config["experiment"]["name"]
    label_col = config["data"].get("label_col", "label")
    image_col = config["data"].get("image_col", "image")

    test_df = pd.read_csv(config["data"]["test_csv"])
    test_df["image_path"] = test_df[image_col]

    loader = build_test_loader(test_df, config)
    device_str = args.device or config["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    weight_paths = collect_weight_paths(args, config, dirs)
    if not weight_paths:
        raise FileNotFoundError("No weight files found for inference")

    label_names = load_label_names(config, Path(logs_dir), experiment_name)
    num_classes = len(label_names)

    all_probs = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for weight_path in weight_paths:
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing weights: {weight_path}")
        model = create_model(
            config["model"]["model_name"],
            num_classes=num_classes,
            pretrained=False,
        )
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        fold_probs: List[np.ndarray] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                fold_probs.append(outputs.softmax(dim=1).cpu().numpy())
        fold_probs_arr = np.concatenate(fold_probs, axis=0)
        all_probs += fold_probs_arr
        print(f"Loaded predictions from {weight_path}")

    all_probs /= len(weight_paths)
    pred_indices = all_probs.argmax(axis=1)
    predictions = [label_names[idx] for idx in pred_indices]

    submission = pd.DataFrame({image_col: test_df[image_col], label_col: predictions})
    output_name = args.output_name or f"{experiment_name}_foldavg.csv"
    output_path = Path(dirs["submissions_dir"]) / output_name
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    infer(parse_args())
