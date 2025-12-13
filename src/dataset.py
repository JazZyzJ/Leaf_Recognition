from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    """PyTorch dataset that reads images from disk and applies transforms."""

    def __init__(
        self,
        df,
        image_root: str | Path,
        transform: Optional[Any] = None,
        image_col: str = "image_path",
        label_col: Optional[str] = "label_id",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.is_test = label_col is None or label_col not in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        row = self.df.iloc[idx]
        image_rel = row[self.image_col]
        image_path = self.image_root / image_rel
        image = self._read_image(image_path)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.is_test:
            # Return dummy label -1 for test mode to avoid None in collate
            label = -1
        else:
            label = int(row[self.label_col])
        return image, label

    def __repr__(self) -> str:  # pragma: no cover - simple helper
        name = self.__class__.__name__
        return f"{name}(size={len(self)}, image_root='{self.image_root}')"
