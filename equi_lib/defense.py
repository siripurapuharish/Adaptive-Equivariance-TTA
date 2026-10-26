"""EquiDefense: a re-implementation of the upstream equivariance defense
(Equi4Rob/learning/attack_new_single.py: Equi_Set_Defense_good_inloop, l_inf, use_CE=False)
that also exposes the per-pixel equivariance error map (P1 of IMPLEMENTATION_PLAN.md).

optimizer="original" reproduces the upstream loop exactly, including the order of every
random draw (Python `random` for view parameters; the torch CPU generator for the unused
global mask and ColorJitter; the torch CUDA generator for the random start and gradient noise),
so the same seed gives the same result up to CUDA non-determinism.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from learning.transform_geo import rot_img

OPTIMIZERS = ("original", "one_step", "sign_decay", "momentum")
LAYER_IND = 1          # DRNSeg output [1]: 19-channel logits before the 8x upsampling
ANGLE_R = 15           # rotation views: uniform in [-15, 15] degrees
UPSTREAM_PI = 3.1415926  # upstream converts degrees with this constant, not math.pi


def _feature_post_norm(f):
    """Upstream feature_post_norm: flatten (C,H,W) per sample, then L2-normalise."""
    v = torch.flatten(f, start_dim=1)
    return v / (torch.sum(v ** 2, dim=1, keepdim=True) ** 0.5 + 1e-10)


def _resize(x, size):
    # upstream uses F.upsample(mode='bilinear') with align_corners=None, which means False
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class EquiDefense:
    def __init__(self, model, normalize, eps_v, num_views, steps=20, optimizer="original",
                 step_size=1.0, device="cuda"):
        if not 1 <= num_views <= 9:
            raise ValueError("num_views must be in 1..9 (identity + up to 8 transforms)")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer {optimizer!r} not in {OPTIMIZERS}")
        if optimizer == "one_step" and steps != 1:
            raise ValueError("one_step uses steps=1")
        self.model, self.normalize = model, normalize
        self.eps_v, self.num_views, self.steps = eps_v, num_views, steps
        self.optimizer, self.step_size = optimizer, step_size
        s = 1  # upstream colour-jitter strength
        jitter = nn.Sequential(transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
                               transforms.RandomGrayscale(p=0.2))
        self.scripted = torch.jit.script(jitter).to(device)

    def _features(self, x):
        return self.model(self.normalize(x))[LAYER_IND]

    def _view_features(self, x, f0):
        """For each transform t (cnt = 0..num_views-2, upstream order and random draws):
        returns f_t mapped back onto f0's grid and a float validity mask (same shape as f0)."""
        H, W = x.size(2), x.size(3)
        out = []
        for cnt in range(self.num_views - 1):
            theta = None
            if cnt == 0:
                ratio_s = random.uniform(0.8, 1.2)  # upstream draws this and then overwrites it
                w_s = 2
            if cnt == 1:
                w_s = 0.3
                ratio_s = random.uniform(0.8, 1.2)
            if cnt == 2:
                w_s = 2
                ratio_s = random.uniform(0.8, 1.2)
            if cnt == 3:
                w_s = 0.5
            if cnt < 4:
                ratio_s = random.uniform(0.8, 1.2)
                xv = _resize(x, (int(H * w_s), int(W * w_s * ratio_s)))
            elif cnt == 6:
                xv = self.scripted(x)                      # ColorJitter + RandomGrayscale
                ratio_s = random.uniform(0.8, 1.2)
                xv = _resize(xv, (int(H), int(W * ratio_s)))
            elif cnt == 7:
                ratio_s = random.uniform(0.8, 1.2)
                xv = _resize(x, (int(H), int(W * ratio_s)))
                xv = torch.flip(xv, dims=[3])
            else:  # cnt 4, 5: rotation
                theta = random.uniform(-ANGLE_R, ANGLE_R) / 180 * UPSTREAM_PI
                xv = rot_img(x, theta, x.dtype)

            f1 = self._features(xv)
            if theta is not None:
                f1 = rot_img(f1, -theta, f1.dtype)
                mask = rot_img(rot_img(torch.ones_like(f0), theta, f0.dtype), -theta, f0.dtype)
            else:
                f1 = _resize(f1, (f0.size(2), f0.size(3)))
                mask = torch.ones_like(f0)
                if cnt == 7:
                    f1 = torch.flip(f1, dims=[3])
            out.append((f1, mask))
        return out

    def _global_mask(self, x):
        # Upstream draws torch.rand on the CPU generator for a mask that keeps every position
        # (keep_ratio=2). It has no effect, but is drawn so the random streams stay aligned.
        with torch.no_grad():
            f = self._features(x)
        return torch.rand([1, 1, f.size(2), f.size(3)]).to(x.device) < 2

    def loss(self, x_adv, global_mask):
        """Upstream objective: negative sum of pairwise dot products of globally L2-normalised features."""
        f0 = self._features(x_adv)
        f_list = [_feature_post_norm(f0 * global_mask)]
        mask_list = [torch.flatten(torch.ones_like(f0), start_dim=1)]
        for f1, mask in self._view_features(x_adv, f0):
            f_list.append(_feature_post_norm(f1 * global_mask))
            mask_list.append(torch.flatten(mask, start_dim=1) > 0.5)
        cost = 0
        for i, a in enumerate(f_list):
            for j, b in enumerate(f_list):
                if i < j:
                    cost = cost - torch.sum(a * b * mask_list[i] * mask_list[j], dim=[0, 1])
        return cost

    def purify_delta(self, x):
        """Returns the defense perturbation delta (upstream return value)."""
        if self.optimizer != "original":
            return self._purify_new(x)
        global_mask = self._global_mask(x)
        eps = self.eps_v
        delta = torch.zeros_like(x)
        delta.uniform_(-eps, eps)
        delta = torch.max(torch.min(delta, 1 - x), 0 - x)
        for i in range(self.steps):
            delta = delta.detach().requires_grad_(True)
            cost = self.loss(delta + x, global_mask)
            self.model.zero_grad()
            cost.backward()
            g = delta.grad
            g = g / (torch.sum(g ** 2, dim=[0, 1, 2, 3]) ** 0.5 + 1e-10) + \
                torch.normal(mean=torch.zeros_like(g), std=torch.ones_like(g)) * (self.steps - 1 - i) / self.steps
            g = g.sign()
            delta = delta.detach() - self.step_size * g
            delta = torch.clamp(delta, min=-eps, max=eps)
            delta = torch.max(torch.min(delta, 1 - x), 0 - x)
        return delta.detach()

    def _step_sizes(self):
        """Cosine decay from 2*eps_v (a full-range sign step, as upstream's effective step) to eps_v/4."""
        eps, T = self.eps_v, self.steps
        if T == 1:
            return [2 * eps]
        return [eps / 4 + (2 * eps - eps / 4) * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * i / (T - 1))).item())
                for i in range(T)]

    def _purify_new(self, x):
        """P3 optimisers (same objective as upstream, no gradient noise):
          one_step   : start at a random corner of the eps_v box (where upstream's 19 noisy steps end,
                       REPRODUCTION_REFERENCE.md N1), then one sign step of size 2*eps_v
          sign_decay : uniform random start, T sign steps with cosine-decayed step size
          momentum   : as sign_decay, but steps along sign(m), m <- 0.9 m + g / ||g||_1 (per image; MI-FGSM)
        """
        eps = self.eps_v
        keep_all = torch.ones(1, 1, 1, 1, dtype=torch.bool, device=x.device)  # upstream's mask keeps everything
        if self.optimizer == "one_step":
            delta = eps * torch.sign(torch.empty_like(x).uniform_(-1, 1))
        else:
            delta = torch.empty_like(x).uniform_(-eps, eps)
        delta = torch.max(torch.min(delta, 1 - x), 0 - x)
        m = torch.zeros_like(x)
        for eta in self._step_sizes():
            delta = delta.detach().requires_grad_(True)
            cost = self.loss(delta + x, keep_all)
            g = torch.autograd.grad(cost, delta)[0]
            if self.optimizer == "momentum":
                m = 0.9 * m + g / (g.abs().sum(dim=(1, 2, 3), keepdim=True) + 1e-12)
                g = m
            delta = torch.clamp(delta.detach() - eta * torch.sign(g), -eps, eps)
            delta = torch.max(torch.min(delta, 1 - x), 0 - x)
        return delta.detach()

    def __call__(self, x, y=None):
        """Same interface as defenses.OriginalDefense: returns the defended image. Labels are unused."""
        return torch.clamp(x + self.purify_delta(x), 0.0, 1.0)

    @torch.no_grad()
    def error_map(self, x, out_hw=None):
        """Per-pixel equivariance error (new, for P2/P4):
            e_p = 1 - mean over valid views t of cos(f0[:, :, p], f_t[:, :, p])
        cosine over the 19 channels at feature resolution (43x85 for 340x680 input); a rotation
        view counts at p only where its mask is > 0.5. Uses the same transforms (and random draws)
        as the defense. Returns [B, h, w], or [B, *out_hw] after bilinear upsampling. Range [0, 2]."""
        f0 = self._features(x)
        num = torch.zeros(f0.size(0), f0.size(2), f0.size(3), device=x.device)
        den = torch.zeros_like(num)
        for f1, mask in self._view_features(x, f0):
            valid = (mask[:, 0] > 0.5).float()
            num += F.cosine_similarity(f0, f1, dim=1, eps=1e-8) * valid
            den += valid
        e = 1.0 - num / den.clamp_min(1.0)
        if out_hw is not None:
            e = _resize(e.unsqueeze(1), tuple(out_hw)).squeeze(1)
        return e
