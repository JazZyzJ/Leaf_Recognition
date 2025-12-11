from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dirs(config: Dict) -> Dict[str, Path]:
    base_dirs = config.get("paths", {})
    resolved = {}
    for key, value in base_dirs.items():
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        resolved[key] = path
    return resolved


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


def save_json(path: str | os.PathLike, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
