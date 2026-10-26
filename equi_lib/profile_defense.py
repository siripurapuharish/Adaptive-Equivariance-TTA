"""P3b step 1: where does the defense's time and memory go? Run on an otherwise idle GPU.

    python -m equi_lib.profile_defense --model clean --images 10

Times (torch.cuda.synchronize before every clock read; first batch is a warm-up and is excluded):
  - base forward (no grad), PGD-50 attack, one full upstream-equivalent defense call
  - one defense step split into: base forward with grad, all view forwards, backward, update
  - each view's forward + backward, one at a time with fixed parameters (aspect ratio 1)
Peak GPU memory of a full defense call at batch sizes 1, 2, 4.
Writes results/improvements/p3/profile_<model>.json
"""
import argparse
import json
import os
import time
import warnings

import torch
import torch.nn.functional as F

from . import RESULTS_DIR, ROOT_DIR
from .attacks import PGD_PARAMS, pgd
from .data import load_batch, split_indices
from .defense import EquiDefense
from .model import PRESETS, Normalizer, load_model
from .runner import git_hash
from .seed import set_seed
from learning.transform_geo import rot_img

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(RESULTS_DIR, "p3")

# cnt -> (name, transform) with fixed parameters; mirrors EquiDefense._view_features with ratio 1
def view_transforms(scripted, H, W):
    rs = lambda x, s: F.interpolate(x, size=(int(H * s), int(W * s)), mode="bilinear", align_corners=False)  # noqa: E731
    return [
        ("cnt0 resize 2x", lambda x: rs(x, 2)),
        ("cnt1 resize 0.3x", lambda x: rs(x, 0.3)),
        ("cnt2 resize 2x", lambda x: rs(x, 2)),
        ("cnt3 resize 0.5x", lambda x: rs(x, 0.5)),
        ("cnt4 rotate", lambda x: rot_img(x, 10 / 180 * 3.1415926, x.dtype)),
        ("cnt5 rotate", lambda x: rot_img(x, -10 / 180 * 3.1415926, x.dtype)),
        ("cnt6 jitter+resize 1x", lambda x: rs(scripted(x), 1)),
        ("cnt7 flip+resize 1x", lambda x: torch.flip(rs(x, 1), dims=[3])),
    ]


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--images", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    defense = EquiDefense(model, norm, preset["eps_v"], preset["num_views"], device=device)
    idx_all = split_indices("tune100")[:args.images + 2]   # +2 = one warm-up batch
    set_seed(0)
    nv = preset["num_views"]
    views = view_transforms(defense.scripted, 340, 680)[:nv - 1]

    acc = {k: 0.0 for k in ("base_forward_nograd", "pgd50", "defense_full", "step_base_forward",
                            "step_view_forwards", "step_backward", "step_update")}
    per_view = {name: 0.0 for name, _ in views}
    n_img = 0
    for b in range(0, len(idx_all), 2):
        idx = idx_all[b:b + 2]
        warm = b == 0
        x, y = load_batch(idx, device)
        with torch.no_grad():
            _, t_fwd = timed(lambda: model(norm(x)))
        delta, t_pgd = timed(lambda: pgd(model, norm, x, y, **PGD_PARAMS))
        xa = torch.clamp(x + delta, 0, 1)
        _, t_def = timed(lambda: defense.purify_delta(xa))

        # one defense step, split into parts (same computation as EquiDefense.loss + update)
        gm = defense._global_mask(xa)
        d = (torch.empty_like(xa).uniform_(-defense.eps_v, defense.eps_v)).requires_grad_(True)
        f0, t_b = timed(lambda: defense._features(xa + d))
        vf, t_v = timed(lambda: defense._view_features(xa + d, f0))

        def finish():
            fl = [torch.flatten(f0 * gm, 1)] + [torch.flatten(f1 * gm, 1) for f1, _ in vf]
            fl = [v / (torch.sum(v ** 2, 1, keepdim=True) ** 0.5 + 1e-10) for v in fl]
            cost = 0
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    cost = cost - torch.sum(fl[i] * fl[j])
            cost.backward()
        _, t_bw = timed(finish)

        def update():
            g = d.grad
            g = g / (torch.sum(g ** 2) ** 0.5 + 1e-10) + torch.randn_like(g) * 0.5
            return torch.clamp(d.detach() - g.sign(), -defense.eps_v, defense.eps_v)
        _, t_up = timed(update)

        # each view alone: transform + forward + backward of its feature sum
        pv = {}
        for name, tf in views:
            dd = torch.zeros_like(xa).requires_grad_(True)
            def fb():
                defense._features(tf(xa + dd)).sum().backward()
            _, pv[name] = timed(fb)

        if not warm:
            n = len(idx)
            n_img += n
            for k, v in (("base_forward_nograd", t_fwd), ("pgd50", t_pgd), ("defense_full", t_def),
                         ("step_base_forward", t_b), ("step_view_forwards", t_v), ("step_backward", t_bw),
                         ("step_update", t_up)):
                acc[k] += v
            for name in per_view:
                per_view[name] += pv[name]
        print(f"  batch {b // 2} {'(warm-up)' if warm else ''} defense {t_def / len(idx):.2f} s/img", flush=True)

    per_img = {k: v / n_img for k, v in acc.items()}
    per_view_img = {k: v / n_img for k, v in per_view.items()}
    view_total = sum(per_view_img.values())

    # peak memory of a full defense call at several batch sizes
    peak = {}
    for bs in (1, 2, 4):
        try:
            x, _ = load_batch(idx_all[:bs], device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _, t = timed(lambda: defense.purify_delta(x))
            peak[bs] = {"peak_gb": torch.cuda.max_memory_allocated() / 1e9, "sec_per_img": t / bs}
        except torch.cuda.OutOfMemoryError:
            peak[bs] = "OOM"
            torch.cuda.empty_cache()

    res = {"model": args.model, "num_views": nv, "steps": defense.steps, "images_timed": n_img,
           "sec_per_image": per_img,
           "one_step_sec_per_image": {k: per_img[k] for k in per_img if k.startswith("step_")},
           "per_view_forward_backward_sec_per_image": per_view_img,
           "per_view_share": {k: v / view_total for k, v in per_view_img.items()},
           "peak_memory_by_batch": peak,
           "gpu": torch.cuda.get_device_name(0), "git": git_hash()}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"profile_{args.model}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"\n== Profile {args.model} ({nv} views, {defense.steps} steps, {n_img} images, batch 2)")
    for k, v in per_img.items():
        print(f"  {k:22s} {v:7.3f} s/img")
    print("  per-view forward+backward (share of view cost):")
    for k, v in per_view_img.items():
        print(f"    {k:22s} {v:6.3f} s/img  {100 * v / view_total:5.1f}%")
    print("  peak memory (full defense): " + ", ".join(
        f"bs{bs}: " + (p if isinstance(p, str) else f"{p['peak_gb']:.2f} GB, {p['sec_per_img']:.2f} s/img")
        for bs, p in peak.items()))
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
