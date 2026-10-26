import json
import os

import torch

from . import DATA_DIR
from models.DRNSeg import DRNSeg

PRESETS = {
    # Settings validated in REPRODUCTION_REFERENCE.md section 1.
    "clean": {"ckpt": "checkpoints/clean_drn_d_22_cityscapes.pth", "eps_v": 6.0 / 255.0, "num_views": 7},
    "adv": {"ckpt": "checkpoints/advtrain_drn_d_22_cityscapes.pth.tar", "eps_v": 10.0 / 255.0, "num_views": 9},
}

with open(os.path.join(DATA_DIR, "info.json")) as _f:
    _INFO = json.load(_f)


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = DRNSeg("drn_d_22", classes=19, pretrained=False)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


class Normalizer:
    def __init__(self, device: torch.device):
        self.mean = torch.tensor(_INFO["mean"], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(_INFO["std"], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
