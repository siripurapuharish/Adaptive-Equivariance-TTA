"""P4 step 0: calibrate the budget map and the gate on tune100 only (seed 0, cached PGD-50).

    python -m equi_lib.p4_calibrate --model clean

For each signal (equi = down|l2_softmax, entropy) at feature resolution (43x85), and each smoothing size k:
  a   = 50th percentile of the smoothed signal over all cells of clean images
  b   = 90th percentile of the smoothed signal over attacked cells of PGD images
        ("attacked" = the base prediction's argmax at that cell differs from the clean image's, as in P2)
  tau = 95th percentile of the gate score s = mean(unsmoothed signal) over clean images
        (so ~95% of clean tune100 images skip the defense)
Also reports, for information: the fraction of PGD images the gate would defend, and feature-grid AUROCs.
Writes results/improvements/p4/calib_<model>.json (read by equi_lib.adaptive).
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch

from . import ROOT_DIR
from .adaptive import CALIB_DIR, SIGNALS, calib_path, signal_map, smooth_map
from .attacks import PGD_PARAMS, pgd
from .cache import PGDCache
from .data import load_batch, split_indices
from .defense import EquiDefense
from .model import PRESETS, Normalizer, load_model
from .runner import git_hash
from .seed import set_seed
from .stats import auroc

warnings.filterwarnings("ignore")
SMOOTHS = (1, 9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    dfn = EquiDefense(model, norm, preset["eps_v"], preset["num_views"], steps=1, optimizer="one_step", device=device)
    cache = PGDCache(args.model, 0, PGD_PARAMS)
    indices = split_indices("tune100")
    set_seed(0)

    clean_cells = {(sg, k): [] for sg in SIGNALS for k in SMOOTHS}
    adv_cells = {(sg, k): [] for sg in SIGNALS for k in SMOOTHS}
    adv_changed = {(sg, k): [] for sg in SIGNALS for k in SMOOTHS}   # for AUROC: all adv cells + label
    s_clean = {sg: [] for sg in SIGNALS}
    s_adv = {sg: [] for sg in SIGNALS}
    changed_frac = []
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        if cache.has(idx):
            d = torch.clamp(cache.get(idx, device), -PGD_PARAMS["eps"], PGD_PARAMS["eps"])
            d = torch.clamp(d, -x, 1.0 - x)
        else:
            d = pgd(model, norm, x, y, **PGD_PARAMS)
            cache.put(idx, d)
        xa = torch.clamp(x + d, 0.0, 1.0)
        with torch.no_grad():
            changed = dfn._features(x).argmax(1) != dfn._features(xa).argmax(1)   # [B,h,w]
        changed_frac.append(changed.float().mean().item())
        for sg in SIGNALS:
            ec, ea = signal_map(dfn, x, sg), signal_map(dfn, xa, sg)
            s_clean[sg] += ec.flatten(1).mean(1).tolist()
            s_adv[sg] += ea.flatten(1).mean(1).tolist()
            for k in SMOOTHS:
                c, a_ = smooth_map(ec, k), smooth_map(ea, k)
                clean_cells[(sg, k)].append(c.flatten().cpu().numpy())
                adv_cells[(sg, k)].append(a_[changed].cpu().numpy())
                adv_changed[(sg, k)].append((a_.flatten().cpu().numpy(), changed.flatten().cpu().numpy()))
        print(f"  [{min(b + 2, len(indices))}/{len(indices)}]", flush=True)
    cache.save()

    res = {"model": args.model, "split": "tune100", "seed": 0, "num_images": len(indices),
           "attacked_cell_fraction": float(np.mean(changed_frac)), "signals": {}, "git": git_hash()}
    print(f"\n== P4 calibration, {args.model} model (tune100, {len(indices)} images); "
          f"attacked cells {100 * np.mean(changed_frac):.1f}%")
    for sg in SIGNALS:
        tau = float(np.percentile(s_clean[sg], 95))
        entry = {"tau": tau,
                 "gate_defended_fraction_clean": float(np.mean(np.array(s_clean[sg]) >= tau)),
                 "gate_defended_fraction_pgd": float(np.mean(np.array(s_adv[sg]) >= tau)),
                 "image_auroc": auroc(np.r_[s_clean[sg], s_adv[sg]],
                                      np.r_[np.zeros(len(s_clean[sg])), np.ones(len(s_adv[sg]))])}
        print(f"  {sg:8s} tau {tau:.4f} | gate defends clean {100 * entry['gate_defended_fraction_clean']:.0f}% "
              f"/ PGD {100 * entry['gate_defended_fraction_pgd']:.0f}% | image AUROC {entry['image_auroc']:.3f}")
        for k in SMOOTHS:
            cc = np.concatenate(clean_cells[(sg, k)])
            ac = np.concatenate(adv_cells[(sg, k)])
            sc = np.concatenate([a for a, _ in adv_changed[(sg, k)]])
            lb = np.concatenate([l for _, l in adv_changed[(sg, k)]])
            a, bb = float(np.percentile(cc, 50)), float(np.percentile(ac, 90))
            entry[f"smooth{k}"] = {
                "a": a, "b": bb,
                "clean_pct": {p: float(np.percentile(cc, p)) for p in (10, 50, 90, 99)},
                "attacked_pct": {p: float(np.percentile(ac, p)) for p in (10, 50, 90, 99)},
                "pixel_auroc_changed_feature_grid": auroc(sc, lb)}
            print(f"     smooth {k}: a {a:.4f}  b {bb:.4f} | pixel AUROC (feature grid) "
                  f"{entry[f'smooth{k}']['pixel_auroc_changed_feature_grid']:.3f}")
        res["signals"][sg] = entry
    os.makedirs(CALIB_DIR, exist_ok=True)
    with open(calib_path(args.model), "w") as f:
        json.dump(res, f, indent=1)
    print(f"  saved {os.path.relpath(calib_path(args.model), ROOT_DIR)}")


if __name__ == "__main__":
    main()
