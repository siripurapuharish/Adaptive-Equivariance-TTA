"""P1 equivalence test: EquiDefense(optimizer="original") must reproduce the upstream
Equi_Set_Defense_good_inloop for both model presets.

Pass criteria (fixed before running):
  1. delta check, 2 tune100 images under PGD-50, same seed:
     mismatch(new, upstream) <= 2 * mismatch(upstream, upstream rerun) + 1e-5,
     where mismatch = fraction of delta entries differing by > 1e-6. The upstream-vs-rerun
     mismatch measures CUDA non-determinism (atomic adds in upsample / grid_sample backward).
  2. mIoU check, first 20 tune100 images, PGD-50 + defense, seeds paired per batch:
     |mIoU(new) - mIoU(upstream)| <= 0.3.
  3. error_map sanity: shapes [B,43,85] and [B,340,680], finite, within [0, 2].

Run from the repo root:  python tests/test_defense_equivalence.py
"""
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from equi_lib import ROOT_DIR  # noqa: E402  (adds Equi4Rob to sys.path)
from equi_lib.attacks import PGD_PARAMS, pgd  # noqa: E402
from equi_lib.data import load_batch, split_indices  # noqa: E402
from equi_lib.defense import EquiDefense  # noqa: E402
from equi_lib.defenses import OriginalDefense  # noqa: E402
from equi_lib.metrics import ConfusionMeter  # noqa: E402
from equi_lib.model import PRESETS, Normalizer, load_model  # noqa: E402
from equi_lib.seed import set_seed  # noqa: E402

DEVICE = torch.device("cuda")
SEED = 123


def upstream_delta(orig: OriginalDefense, x, y):
    # upstream moves y to the GPU but never uses it; OriginalDefense returns clamp(x + delta),
    # and delta already keeps x + delta in [0, 1], so subtracting x recovers delta
    return orig(x, y) - x


def mismatch(a, b):
    return float(((a - b).abs() > 1e-6).float().mean()), float((a - b).abs().max())


def check_model(tag):
    preset = PRESETS[tag]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), DEVICE)
    norm = Normalizer(DEVICE)
    orig = OriginalDefense(model, norm, preset["eps_v"], preset["num_views"], device=DEVICE)
    new = EquiDefense(model, norm, preset["eps_v"], preset["num_views"], device=DEVICE)
    tune = split_indices("tune100")
    ok = True
    print(f"\n=== {tag} model: eps_v={preset['eps_v'] * 255:.1f}/255, {preset['num_views']} views")

    # 1. delta check
    x, y = load_batch(tune[:2], DEVICE)
    set_seed(SEED)
    x_adv = torch.clamp(x + pgd(model, norm, x, y, **PGD_PARAMS), 0, 1)
    set_seed(SEED); d_up1 = upstream_delta(orig, x_adv, y)
    set_seed(SEED); d_up2 = upstream_delta(orig, x_adv, y)
    set_seed(SEED); d_new = new.purify_delta(x_adv)
    floor, floor_max = mismatch(d_up1, d_up2)
    mm, mm_max = mismatch(d_up1, d_new)
    limit = 2 * floor + 1e-5
    p1 = mm <= limit
    ok &= p1
    print(f"[{'PASS' if p1 else 'FAIL'}] delta: upstream-vs-rerun mismatch {floor:.2e} (max |diff| {floor_max:.2e}); "
          f"new-vs-upstream {mm:.2e} (max |diff| {mm_max:.2e}); limit {limit:.2e}")

    # 2. mIoU check on 20 images, paired seeds
    m_up, m_new = ConfusionMeter(), ConfusionMeter()
    t0 = time.time()
    for b in range(0, 20, 2):
        xb, yb = load_batch(tune[b:b + 2], DEVICE)
        set_seed(SEED + b)
        xa = torch.clamp(xb + pgd(model, norm, xb, yb, **PGD_PARAMS), 0, 1)
        set_seed(SEED + 1000 + b)
        xd_up = orig(xa, yb)
        set_seed(SEED + 1000 + b)
        xd_new = new(xa, yb)
        with torch.no_grad():
            m_up.update(model(norm(xd_up))[0], yb)
            m_new.update(model(norm(xd_new))[0], yb)
    diff = abs(m_up.miou() - m_new.miou())
    p2 = diff <= 0.3
    ok &= p2
    print(f"[{'PASS' if p2 else 'FAIL'}] mIoU (20 images, PGD+def): upstream {m_up.miou():.3f}, new {m_new.miou():.3f}, "
          f"|diff| {diff:.3f} (limit 0.3)  [{time.time() - t0:.0f}s]")

    # 3. error_map sanity
    e_small = new.error_map(x_adv)
    e_full = new.error_map(x_adv, out_hw=(340, 680))
    p3 = (tuple(e_small.shape) == (2, 43, 85) and tuple(e_full.shape) == (2, 340, 680)
          and bool(torch.isfinite(e_full).all()) and float(e_full.min()) >= 0 and float(e_full.max()) <= 2)
    ok &= p3
    e_clean = new.error_map(x)
    print(f"[{'PASS' if p3 else 'FAIL'}] error_map: shapes {tuple(e_small.shape)} / {tuple(e_full.shape)}, "
          f"range [{float(e_full.min()):.3f}, {float(e_full.max()):.3f}]; "
          f"mean e clean {float(e_clean.mean()):.4f} vs attacked {float(e_small.mean()):.4f} (info only)")
    return ok


if __name__ == "__main__":
    results = {tag: check_model(tag) for tag in ("clean", "adv")}
    print("\nOVERALL:", "PASS" if all(results.values()) else "FAIL", results)
    sys.exit(0 if all(results.values()) else 1)
