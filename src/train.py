from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import amp
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV2
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from dataset import LeafDataset
from models import create_model
from utils import AverageMeter, accuracy, load_config, prepare_dirs, save_json, set_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet baseline for Leaf Classification")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Override device")
    return parser.parse_args()


def get_transforms(img_size: int) -> Tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            A.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            A.Resize(img_size + 32, img_size + 32),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return train_transform, valid_transform


def build_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Dict,
    transforms: Tuple[A.Compose, A.Compose],
) -> Tuple[DataLoader, DataLoader]:
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    image_root = config["data"].get("image_root", ".")
    pin_memory = torch.cuda.is_available()

    train_dataset = LeafDataset(
        train_df,
        image_root=image_root,
        transform=transforms[0],
        image_col="image_path",
        label_col="label_id",
    )
    valid_dataset = LeafDataset(
        valid_df,
        image_root=image_root,
        transform=transforms[1],
        image_col="image_path",
        label_col="label_id",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    params = model.parameters()
    training_cfg = config["training"]
    lr = training_cfg["lr"]
    weight_decay = training_cfg.get("weight_decay", 0.0)
    optimizer_name = training_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = training_cfg.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    steps_per_epoch: int,
    ) -> Tuple[torch.optim.lr_scheduler._LRScheduler | None, bool]:
    training_cfg = config["training"]
    scheduler_name = training_cfg.get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        t_max = training_cfg.get("t_max", training_cfg["num_epochs"])
        min_lr = training_cfg.get("min_lr", 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
        return scheduler, False
    if scheduler_name == "onecycle":
        epochs = training_cfg["num_epochs"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_cfg["lr"],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return scheduler, True
    return None, False


def build_mixup_fn(config: Dict, num_classes: int) -> Mixup | None:
    mix_cfg = config["training"].get("mixup", {})
    if not mix_cfg.get("enabled", False):
        return None

    mixup_alpha = mix_cfg.get("mixup_alpha", 0.0)
    cutmix_alpha = mix_cfg.get("cutmix_alpha", 0.0)
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return None

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mix_cfg.get("prob", 1.0),
        switch_prob=mix_cfg.get("switch_prob", 0.5),
        mode=mix_cfg.get("mode", "batch"),
        label_smoothing=mix_cfg.get("label_smoothing", 0.0),
        num_classes=num_classes,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scheduler_step_per_batch: bool = False,
    grad_clip: float | None = None,
    use_amp: bool = True,
    ema: ModelEmaV2 | None = None,
    mixup_fn: Mixup | None = None,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    optimizer.zero_grad()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_for_metrics = targets.clone()

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)

        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        acc_value = accuracy(outputs, targets_for_metrics)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc_value, images.size(0))

    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_probs = outputs.softmax(dim=1).cpu().numpy()
            probs.append(batch_probs)

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy(outputs, targets), images.size(0))

    fold_probs = np.concatenate(probs, axis=0)
    return loss_meter.avg, acc_meter.avg, fold_probs


def run_training(config_path: str, device_override: str | None = None) -> None:
    config = load_config(config_path)
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    dirs = prepare_dirs(config)
    experiment_name = config["experiment"]["name"]

    train_csv = Path(config["data"]["train_csv"])
    train_df = pd.read_csv(train_csv)
    train_df["image_path"] = train_df[config["data"].get("image_col", "image")]

    label_col = config["data"].get("label_col", "label")
    label_encoder = LabelEncoder()
    train_df["label_id"] = label_encoder.fit_transform(train_df[label_col])

    folds = config["experiment"].get("num_folds", 5)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    train_df["fold"] = -1
    for fold_id, (_, valid_idx) in enumerate(skf.split(train_df, train_df["label_id"])):
        train_df.loc[valid_idx, "fold"] = fold_id

    folds_csv = config["experiment"].get("save_folds_csv")
    if folds_csv:
        train_df.to_csv(folds_csv, index=False)

    num_classes = train_df["label_id"].nunique()
    mixup_fn = build_mixup_fn(config, num_classes)
    logs_dir = dirs.get("logs_dir", Path("logs"))
    save_json(
        Path(logs_dir) / f"{experiment_name}_label_mapping.json",
        {"classes": label_encoder.classes_.tolist()},
    )

    img_size = config["training"]["img_size"]
    train_transform, valid_transform = get_transforms(img_size)

    device_str = device_override or config["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float32)
    history: List[Dict] = []

    for fold in range(folds):
        print(f"Starting fold {fold}")
        train_subset = train_df[train_df["fold"] != fold].reset_index(drop=True)
        valid_mask = train_df["fold"] == fold
        valid_indices = train_df[valid_mask].index.to_numpy()
        valid_subset = train_df[valid_mask].reset_index(drop=True)

        train_loader, valid_loader = build_dataloaders(
            train_subset, valid_subset, config, (train_transform, valid_transform)
        )

        model = create_model(
            config["model"]["model_name"],
            num_classes=num_classes,
            pretrained=config["model"].get("pretrained", True),
        )
        model.to(device)

        optimizer = build_optimizer(model, config)
        scheduler, step_per_batch = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

        if mixup_fn is not None:
            train_criterion = SoftTargetCrossEntropy()
        else:
            train_criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.0))
        val_criterion = nn.CrossEntropyLoss()
        amp_enabled = config["training"].get("amp", True) and device.type == "cuda"
        scaler = amp.GradScaler(device_type=device.type, enabled=amp_enabled)
        grad_clip = config["training"].get("grad_clip")
        ema_cfg = config["training"].get("ema", {})
        ema: ModelEmaV2 | None = None
        if ema_cfg.get("enabled", False):
            ema_decay = ema_cfg.get("decay", 0.9998)
            ema_device = ema_cfg.get("device", "")
            ema = ModelEmaV2(model, decay=ema_decay, device=ema_device)

        best_acc = 0.0
        best_probs = None
        epochs = config["training"]["num_epochs"]
        weights_path = Path(dirs["weights_dir"]) / f"{experiment_name}_fold{fold}.pth"

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                train_criterion,
                optimizer,
                device,
                scaler,
                scheduler=scheduler,
                scheduler_step_per_batch=step_per_batch,
                grad_clip=grad_clip,
                use_amp=scaler.is_enabled(),
                ema=ema,
                mixup_fn=mixup_fn,
            )

            eval_model = ema.module if ema is not None else model
            val_loss, val_acc, val_probs = validate(eval_model, valid_loader, val_criterion, device)
            if scheduler is not None and not step_per_batch:
                scheduler.step()

            epoch_log = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_log)
            print(json.dumps(epoch_log))

            if val_acc > best_acc:
                best_acc = val_acc
                best_probs = val_probs.copy()
                torch.save(eval_model.state_dict(), weights_path)
                print(f"Fold {fold} epoch {epoch}: improved val acc to {val_acc:.4f}")

        if best_probs is None:
            raise RuntimeError("Validation never executed for fold {fold}")
        oof_preds[valid_indices, :] = best_probs
        print(f"Saved best model for fold {fold} to {weights_path}")

    oof_path = Path(dirs["oof_dir"]) / f"{experiment_name}_oof.npy"
    np.save(oof_path, oof_preds)
    print(f"Saved OOF predictions to {oof_path}")

    cv_acc = accuracy_score(train_df["label_id"], oof_preds.argmax(axis=1))
    summary = {"experiment": experiment_name, "cv_accuracy": cv_acc, "folds": folds}
    save_json(Path(logs_dir) / f"{experiment_name}_summary.json", summary)
    save_json(Path(logs_dir) / f"{experiment_name}_history.json", {"history": history})

    print(f"CV Accuracy: {cv_acc:.4f}")


def main() -> None:
    args = parse_args()
    run_training(args.config, device_override=args.device)


if __name__ == "__main__":
    main()
