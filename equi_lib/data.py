"""Cityscapes val (680x340) loading and fixed evaluation splits.

The val list is sorted by city (267 Frankfurt, 59 Lindau, 174 Munster), so splits are
interleaved to cover every city: test100 = i % 5 == 0, tune100 = i % 5 == 2.
Tune hyperparameters only on tune100; report on test100 / full500.
"""
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from . import DATA_DIR


def _read_list(name: str) -> List[str]:
    with open(os.path.join(DATA_DIR, name)) as f:
        return [line.strip() for line in f if line.strip()]


IMAGE_PATHS = _read_list("val_images.txt")
LABEL_PATHS = _read_list("val_labels.txt")
assert len(IMAGE_PATHS) == len(LABEL_PATHS) == 500


def split_indices(name: str) -> List[int]:
    n = len(IMAGE_PATHS)
    if name == "full500":
        return list(range(n))
    if name == "test100":
        return [i for i in range(n) if i % 5 == 0]
    if name == "tune100":
        return [i for i in range(n) if i % 5 == 2]
    raise ValueError(f"Unknown split {name!r} (use full500, test100, tune100)")


def load_batch(indices: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns images [B,3,340,680] in [0,1] and labels [B,340,680] int64 (255 = ignore)."""
    imgs, lbls = [], []
    for i in indices:
        im = np.array(Image.open(os.path.join(DATA_DIR, IMAGE_PATHS[i])).convert("RGB"), dtype=np.float32) / 255.0
        lb = np.array(Image.open(os.path.join(DATA_DIR, LABEL_PATHS[i])), dtype=np.int64)
        imgs.append(torch.from_numpy(im).permute(2, 0, 1))
        lbls.append(torch.from_numpy(lb))
    return torch.stack(imgs).to(device), torch.stack(lbls).to(device)
