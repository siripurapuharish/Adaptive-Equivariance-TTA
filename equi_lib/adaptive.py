"""P4: equivariance-guided adaptive defense (IMPLEMENTATION_PLAN.md, P4).

For each input x (signal computed once from the input, detached):
  e      = signal map at feature resolution (43x85):
             "equi"    : EquiDefense.consistency_map (down|l2_softmax, the P2 winner)
             "entropy" : softmax entropy of the base prediction (signal-ablation baseline)
  s      = mean(e) per image                              (gate score)
  e_s    = avg_pool(e, smooth x smooth, stride 1)          (avoids speckle)
  r      = clip((e_s - a) / (b - a), beta, 1)              (a, b calibrated on tune100, p4_calibrate.py)
  eps(p) = eps_v * bilinear_up(r)                          (image resolution; stays in [beta, 1] * eps_v)
then one_step inside the per-pixel box [-eps(p), eps(p)].

Modes (everything else fixed, so each is a controlled comparison):
  global        : eps(p) = eps_v everywhere (same code path; equals one_step)
  perpixel      : eps(p) from the map                                           <- ours
  perimage      : each image gets the scalar mean of its own eps(p)            (matched-budget control:
                  same total budget per image, no spatial placement)
  shuffled      : eps(p) rolled by half the image height and width             (control: same budget
                  histogram, wrong places)
  gate          : skip the defense if s < tau, else global eps_v
  gate+perpixel : skip if s < tau, else perpixel                               <- ours, clean model
"""
import json
import os

import torch
import torch.nn.functional as F

from . import RESULTS_DIR
from .defense import EquiDefense

MODES = ("global", "perpixel", "perimage", "shuffled", "gate", "gate+perpixel")
SIGNALS = ("equi", "entropy")
CALIB_DIR = os.path.join(RESULTS_DIR, "p4")


def calib_path(model_tag):
    return os.path.join(CALIB_DIR, f"calib_{model_tag}.json")


def signal_map(defense: EquiDefense, x, signal):
    """[B, h, w] signal at feature resolution, no grad."""
    if signal == "equi":
        return defense.consistency_map(x)
    if signal == "entropy":
        with torch.no_grad():
            logp = F.log_softmax(defense._features(x), 1)
            return -(logp.exp() * logp).sum(1)
    raise ValueError(signal)


def smooth_map(e, k):
    if k <= 1:
        return e
    return F.avg_pool2d(e.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False).squeeze(1)


class AdaptiveDefense:
    def __init__(self, model, normalize, eps_v, num_views, device="cuda", model_tag=None,
                 mode="perpixel", signal="equi", beta=0.2, smooth=9, calib=None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if signal not in SIGNALS:
            raise ValueError(f"signal {signal!r} not in {SIGNALS}")
        self.inner = EquiDefense(model, normalize, eps_v, num_views, steps=1, optimizer="one_step", device=device)
        self.eps_v, self.mode, self.signal = eps_v, mode, signal
        self.beta, self.smooth = float(beta), int(smooth)
        if calib is None:
            if model_tag is None:
                raise ValueError("need model_tag or calib")
            with open(calib_path(model_tag)) as f:
                calib = json.load(f)
        cut = calib["signals"][signal][f"smooth{self.smooth}"]
        self.a, self.b = float(cut["a"]), float(cut["b"])
        if not self.b > self.a:
            raise ValueError(f"calibration has b <= a ({self.b} <= {self.a})")
        self.tau = float(calib["signals"][signal]["tau"])
        self.calib_info = {"a": self.a, "b": self.b, "tau": self.tau, "calib_git": calib.get("git")}
        self._stats = []

    def budget(self, x):
        """Returns (eps_map [B,1,H,W] for the non-gate modes, gate score s [B])."""
        e = signal_map(self.inner, x, self.signal)
        s = e.flatten(1).mean(1)
        r = ((smooth_map(e, self.smooth) - self.a) / (self.b - self.a)).clamp(self.beta, 1.0)
        r = F.interpolate(r.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)
        r = r.clamp(self.beta, 1.0)   # bilinear interpolation stays in range; clamp guards rounding only
        return self.eps_v * r, s

    def __call__(self, x, y=None):
        B = x.size(0)
        eps_map, s = self.budget(x)
        if self.mode in ("global", "gate"):
            eps_map = torch.full_like(eps_map, self.eps_v)
        elif self.mode == "perimage":
            eps_map = eps_map.mean(dim=(1, 2, 3), keepdim=True).expand_as(eps_map).contiguous()
        elif self.mode == "shuffled":
            eps_map = torch.roll(eps_map, shifts=(x.size(2) // 2, x.size(3) // 2), dims=(2, 3))
        run = torch.ones(B, dtype=torch.bool, device=x.device)
        if self.mode.startswith("gate"):
            run = s >= self.tau
        out = x.clone()
        if run.any():
            xi = x[run]
            out[run] = torch.clamp(xi + self.inner.purify_delta(xi, eps_map[run]), 0.0, 1.0)
        for i in range(B):
            self._stats.append({"score": float(s[i]), "defended": bool(run[i]),
                                "mean_eps_frac": float(eps_map[i].mean() / self.eps_v) if run[i] else 0.0})
        return out

    def pop_stats(self):
        """Per-image stats since the last call (the runner aggregates them per mode)."""
        st, self._stats = self._stats, []
        return st


def make_adaptive(model, normalize, eps_v, num_views, device="cuda", model_tag=None, **opts):
    return AdaptiveDefense(model, normalize, eps_v, num_views, device=device, model_tag=model_tag, **opts)
