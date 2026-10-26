import numpy as np
import torch
import torch.nn.functional as F

from learning.utils_learn import fast_hist, per_class_iu

NUM_CLASSES = 19
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle"]


def align_output(out: torch.Tensor, target_hw=(340, 680)) -> torch.Tensor:
    """DRN-D-22 outputs 344 rows for a 340-row input; resize logits to label size."""
    if tuple(out.shape[-2:]) == tuple(target_hw):
        return out
    return F.interpolate(out, size=tuple(target_hw), mode="bilinear", align_corners=False)


class ConfusionMeter:
    def __init__(self):
        self.hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        pred = align_output(logits, labels.shape[-2:]).argmax(1).cpu().numpy()
        gt = labels.cpu().numpy()
        for p, g in zip(pred, gt):
            self.hist += fast_hist(p.flatten(), g.flatten(), NUM_CLASSES)

    def per_class(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return per_class_iu(self.hist) * 100

    def miou(self) -> float:
        return float(np.nanmean(self.per_class()))
