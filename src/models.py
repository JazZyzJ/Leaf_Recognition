from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    checkpoint_path: Optional[str | Path] = None,
) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            missing_msg = ", ".join(missing)
            unexpected_msg = ", ".join(unexpected)
            print(f"Loaded checkpoint with missing keys: {missing_msg}; unexpected: {unexpected_msg}")
    return model
