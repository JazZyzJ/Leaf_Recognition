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
from train import IMAGENET_MEAN, IMAGENET_STD
from utils import load_config, prepare_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference / ensemble for Leaf Classification")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config file(s) to use. Repeat flag to ensemble multiple experiments.",
    )
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument(
        "--weights-pattern",
        action="append",
        help="Optional glob or comma-separated paths per config (same order as --config).",
    )
    parser.add_argument("--output_name", default=None, help="Custom submission filename")
    parser.add_argument(
        "--tta",
        choices=["none", "hflip"],
        default="none",
        help="Apply test-time augmentation by adding a horizontal flip pass.",
    )
    return parser.parse_args()


def load_label_names(config: Dict, logs_dir: Path, experiment_name: str) -> List[str]:
    mapping_path = logs_dir / f"{experiment_name}_label_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["classes"]
    train_df = pd.read_csv(config["data"]["train_csv"])
    label_col = config["data"].get("label_col", "label")
    return sorted(train_df[label_col].unique())


def build_transform(img_size: int, hflip: bool = False) -> A.Compose:
    transforms = []
    if hflip:
        transforms.append(A.HorizontalFlip(p=1.0))
    transforms.extend(
        [
            A.Resize(img_size + 32, img_size + 32),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_loader(test_df: pd.DataFrame, config: Dict, transform: A.Compose) -> DataLoader:
    dataset = LeafDataset(
        test_df,
        image_root=config["data"].get("image_root", "."),
        transform=transform,
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


def collect_weight_paths(config: Dict, dirs: Dict[str, Path], pattern: str | None = None) -> List[Path]:
    if pattern:
        if "," in pattern:
            return [Path(p.strip()) for p in pattern.split(",")]
        return sorted(Path().glob(pattern))
    folds = config["experiment"].get("num_folds", 5)
    experiment_name = config["experiment"]["name"]
    return [Path(dirs["weights_dir"]) / f"{experiment_name}_fold{fold}.pth" for fold in range(folds)]


def run_model(
    config: Dict,
    weight_paths: List[Path],
    device: torch.device,
    loaders: List[DataLoader],
    num_samples: int,
    num_classes: int,
) -> np.ndarray:
    total_probs = np.zeros((num_samples, num_classes), dtype=np.float32)

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

        fold_probs = np.zeros_like(total_probs)
        for loader in loaders:
            offset = 0
            preds_chunks: List[np.ndarray] = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device, non_blocking=True)
                    outputs = model(images)
                    preds_chunks.append(outputs.softmax(dim=1).cpu().numpy())
            fold_probs += np.concatenate(preds_chunks, axis=0)

        fold_probs /= len(loaders)
        total_probs += fold_probs
        print(f"Loaded predictions from {weight_path}")

    total_probs /= len(weight_paths)
    return total_probs


def infer(args: argparse.Namespace) -> None:
    config_paths = args.config
    weight_patterns = args.weights_pattern or []

    base_config = load_config(config_paths[0])
    label_col = base_config["data"].get("label_col", "label")
    image_col = base_config["data"].get("image_col", "image")

    test_df = pd.read_csv(base_config["data"]["test_csv"])
    test_df["image_path"] = test_df[image_col]
    num_samples = len(test_df)

    device_str = args.device or base_config["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    base_dirs = prepare_dirs(base_config)
    label_names = load_label_names(
        base_config, Path(base_dirs.get("logs_dir", Path("logs"))), base_config["experiment"]["name"]
    )
    agg_probs = None
    for idx, config_path in enumerate(config_paths):
        config = load_config(config_path)
        cfg_dirs = prepare_dirs(config)
        pattern = None
        if weight_patterns:
            if len(weight_patterns) == 1:
                pattern = weight_patterns[0]
            elif idx < len(weight_patterns):
                pattern = weight_patterns[idx]
        weights = collect_weight_paths(config, cfg_dirs, pattern)

        transforms = [build_transform(config["training"]["img_size"], hflip=False)]
        if args.tta == "hflip":
            transforms.append(build_transform(config["training"]["img_size"], hflip=True))
        loaders = [build_loader(test_df, config, transform) for transform in transforms]

        probs = run_model(
            config,
            weights,
            device,
            loaders,
            num_samples,
            num_classes=len(label_names),
        )
        if agg_probs is None:
            agg_probs = probs
        else:
            agg_probs += probs

    agg_probs /= len(config_paths)
    experiment_tags = "+".join(Path(c).stem for c in config_paths)

    predictions = [label_names[idx] for idx in agg_probs.argmax(axis=1)]

    submission = pd.DataFrame({image_col: test_df[image_col], label_col: predictions})
    output_name = args.output_name or f"{experiment_tags}_ensemble.csv"
    submission_dir = Path(base_config.get("paths", {}).get("submissions_dir", "submissions"))
    submission_dir.mkdir(parents=True, exist_ok=True)
    output_path = submission_dir / output_name
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    infer(parse_args())
