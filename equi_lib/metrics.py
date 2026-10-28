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
    def __init__(self, keep_per_image=False):
        self.hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self.per_image = [] if keep_per_image else None   # E1: one [19, 19] confusion matrix per image

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        pred = align_output(logits, labels.shape[-2:]).argmax(1).cpu().numpy()
        gt = labels.cpu().numpy()
        for p, g in zip(pred, gt):
            h = fast_hist(p.flatten(), g.flatten(), NUM_CLASSES)
            self.hist += h
            if self.per_image is not None:
                self.per_image.append(h.astype(np.int32))

    def per_class(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return per_class_iu(self.hist) * 100

    def miou(self) -> float:
        return float(np.nanmean(self.per_class()))
