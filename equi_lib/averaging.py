"""E2 / P5: prediction averaging over views (IMPLEMENTATION_PLAN.md, P5).

After the defense (or on an undefended input), predict on the image and on transformed copies, map each
prediction back to the label grid (un-flip; resize back), and average the log-softmax outputs (a geometric-mean
ensemble). Averaging log-probabilities keeps 'single' bit-identical to the plain prediction path
(align_output(model(...)[0])); averaging probabilities would move boundary pixels through the resizing alone.

Variants:
  single   : the plain prediction (what every earlier result used)
  avg_flip : identity + horizontal flip                       (2 predictions)
  avg_rs   : identity + flip + resize 0.75x + resize 1.25x    (4 predictions)
Rotations are left out: P5 allows them only if they help on tune100, and they need masked averaging.
"""
import torch
import torch.nn.functional as F

from .metrics import align_output

VARIANTS = {"single": (), "avg_flip": ("flip",), "avg_rs": ("flip", 0.75, 1.25)}


def _logp(model, normalize, x, out_hw):
    # DRNSeg returns (log-softmax, features, softmax); [0] is what every earlier result used.
    return align_output(model(normalize(x))[0], out_hw)


@torch.no_grad()
def predict(model, normalize, x, variant="single", out_hw=None):
    """Returns the mean log-softmax [B, 19, H, W] at out_hw (default: x's size); ConfusionMeter takes its argmax."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {tuple(VARIANTS)}")
    out_hw = tuple(out_hw or x.shape[-2:])
    p = _logp(model, normalize, x, out_hw)
    for v in VARIANTS[variant]:
        if v == "flip":
            p = p + torch.flip(_logp(model, normalize, torch.flip(x, dims=[3]), out_hw), dims=[3])
        else:
            xs = F.interpolate(x, size=(int(x.size(2) * v), int(x.size(3) * v)), mode="bilinear", align_corners=False)
            p = p + _logp(model, normalize, xs, out_hw)
    return p / (1 + len(VARIANTS[variant]))
