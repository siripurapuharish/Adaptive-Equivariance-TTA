"""P3 evidence for finding N7: how low does each optimiser drive the defense's own objective?

    python -m equi_lib.p3_loss_check --model clean --images 10

For each image (tune100, seed 0, clean input and cached PGD-50 input) runs every optimiser and
evaluates the upstream equivariance loss (EquiDefense.loss: negative sum of pairwise dot products
of globally L2-normalised features; minimum -(V choose 2) for V views) at
  - the undefended input (delta = 0)
  - the defended input x + delta for each optimiser.
Every loss evaluation re-seeds the RNG first, so all optimisers are scored on identical random
views. Reported per image (the upstream loss sums over the batch).
Writes results/improvements/p3/loss_check_<model>.json
"""
import argparse
import json
import os
import statistics as st
import warnings

import torch

from . import RESULTS_DIR, ROOT_DIR
from .attacks import PGD_PARAMS, pgd
from .cache import PGDCache
from .data import load_batch, split_indices
from .defense import EquiDefense
from .model import PRESETS, Normalizer, load_model
from .runner import git_hash
from .seed import set_seed

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(RESULTS_DIR, "p3")
OPTIMISERS = {"original": ("original", 20), "one_step": ("one_step", 1),
              "sign_decay3": ("sign_decay", 3), "momentum3": ("momentum", 3)}
EVAL_SEED = 1234


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--images", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    defs = {k: EquiDefense(model, norm, preset["eps_v"], preset["num_views"], steps=s, optimizer=o, device=device)
            for k, (o, s) in OPTIMISERS.items()}
    scorer = defs["original"]
    keep_all = torch.ones(1, 1, 1, 1, dtype=torch.bool, device=device)
    cache = PGDCache(args.model, 0, PGD_PARAMS)
    indices = split_indices("tune100")[:args.images]
    set_seed(0)

    def score(xin):
        set_seed(EVAL_SEED)
        with torch.no_grad():
            return scorer.loss(xin, keep_all).item() / xin.size(0)

    vals = {inp: {k: [] for k in ("undefended", *OPTIMISERS)} for inp in ("clean", "pgd")}
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        if cache.has(idx):
            d = torch.clamp(cache.get(idx, device), -PGD_PARAMS["eps"], PGD_PARAMS["eps"])
            d = torch.clamp(d, -x, 1.0 - x)
        else:
            d = pgd(model, norm, x, y, **PGD_PARAMS)
        for inp, xin in (("clean", x), ("pgd", torch.clamp(x + d, 0.0, 1.0))):
            vals[inp]["undefended"].append(score(xin))
            for k, dfn in defs.items():
                set_seed(b)
                vals[inp][k].append(score(dfn(xin)))
        print(f"  [{min(b + 2, len(indices))}/{len(indices)}]", flush=True)

    v = preset["num_views"]
    res = {"model": args.model, "num_views": v, "loss_minimum": -v * (v - 1) / 2, "num_images": len(indices),
           "split": "tune100", "per_image_mean": {}, "per_image_std": {}, "git": git_hash()}
    print(f"\n== Equivariance loss per image, {args.model} ({v} views, minimum {-v * (v - 1) / 2:.0f})")
    for inp in vals:
        res["per_image_mean"][inp] = {k: st.mean(a) for k, a in vals[inp].items()}
        res["per_image_std"][inp] = {k: st.pstdev(a) for k, a in vals[inp].items()}
        print(f"  {inp:5s} " + " | ".join(f"{k} {st.mean(a):.2f}" for k, a in vals[inp].items()))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"loss_check_{args.model}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
