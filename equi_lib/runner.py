"""Evaluate a model on a fixed split under clean / PGD inputs, with or without a defense.

Example:
    python -m equi_lib.runner --model clean --split test100 --seeds 0 1 2 --modes clean pgd clean+def pgd+def

Writes results/improvements/<tag>_<split>_s<seed>.json per seed and prints mean +/- std over seeds.
"""
import argparse
import json
import os
import subprocess
import time
import warnings
from collections import defaultdict

import numpy as np
import torch

from . import RESULTS_DIR, ROOT_DIR
from .attacks import PGD_PARAMS, pgd
from .averaging import VARIANTS as PREDICT_VARIANTS
from .averaging import predict
from .cache import PGDCache
from .data import load_batch, split_indices
from .defenses import DEFENSE_TAKES_OPTS, DEFENSES
from .metrics import CLASSES, ConfusionMeter
from .model import PRESETS, Normalizer, load_model
from .seed import set_seed

warnings.filterwarnings("ignore")  # upstream code uses deprecated upsample/affine_grid calls

MODES = ("clean", "pgd", "clean+def", "pgd+def")
ATTACK_EPS = PGD_PARAMS["eps"]


def git_hash() -> str:
    """Short commit id, with '-dirty' if tracked files have uncommitted changes."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, text=True).strip()
        changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                          cwd=ROOT_DIR, text=True).strip()
        return head + ("-dirty" if changes else "")
    except Exception:
        return "unknown"


def run_seed(args, seed, model, normalize, defense, indices, device):
    set_seed(seed)
    meters = {m: ConfusionMeter(keep_per_image=args.per_image) for m in args.modes}
    secs = defaultdict(float)
    dstats = defaultdict(list)   # per-image defense stats (adaptive defenses only)
    need_pgd = any(m.startswith("pgd") for m in args.modes)
    attack_seed = seed if args.attack_seed is None else args.attack_seed
    cache = PGDCache(args.model, attack_seed, PGD_PARAMS) if (need_pgd and not args.no_cache) else None

    for b in range(0, len(indices), args.batch_size):
        idx = indices[b:b + args.batch_size]
        x, y = load_batch(idx, device)

        if "clean" in meters:
            t = time.time()
            with torch.no_grad():
                meters["clean"].update(predict(model, normalize, x, args.predict, y.shape[-2:]), y)
            secs["clean"] += time.time() - t

        if need_pgd:
            t = time.time()
            if cache is not None and cache.has(idx):
                # float16 storage can overshoot the budget by ~1e-5; re-project onto it.
                delta = torch.clamp(cache.get(idx, device), -ATTACK_EPS, ATTACK_EPS)
                delta = torch.clamp(delta, -x, 1.0 - x)
            else:
                delta = pgd(model, normalize, x, y, **PGD_PARAMS)
                if cache is not None:
                    cache.put(idx, delta)
            torch.cuda.synchronize()
            secs["pgd_attack"] += time.time() - t
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
            if "pgd" in meters:
                with torch.no_grad():
                    meters["pgd"].update(predict(model, normalize, x_adv, args.predict, y.shape[-2:]), y)

        x_clean_def = x
        if args.clean_noise and "clean+def" in meters:
            # Upstream runs clean+defense rows with a 0-step PGD, which returns its random start:
            # x + U(-eps, eps). See REPRODUCTION_REFERENCE.md Note B.
            noise = torch.empty_like(x).uniform_(-ATTACK_EPS, ATTACK_EPS)
            x_clean_def = torch.clamp(x + noise, 0.0, 1.0)

        for mode, inp in (("clean+def", x_clean_def), ("pgd+def", x_adv if need_pgd else None)):
            if mode in meters:
                t = time.time()
                x_def = defense(inp, y)
                torch.cuda.synchronize()
                secs[mode] += time.time() - t
                if hasattr(defense, "pop_stats"):
                    dstats[mode] += defense.pop_stats()
                t = time.time()
                meters[mode].update(predict(model, normalize, x_def, args.predict, y.shape[-2:]), y)
                torch.cuda.synchronize()
                secs["predict"] += time.time() - t

        done = min(b + args.batch_size, len(indices))
        print(f"  seed {seed} [{done:3d}/{len(indices)}] " +
              " | ".join(f"{m}: {meters[m].miou():.2f}" for m in args.modes), flush=True)

    if cache is not None:
        cache.save()
    n = len(indices)
    out = {
        "miou": {m: meters[m].miou() for m in args.modes},
        "per_class_iou": {m: dict(zip(CLASSES, np.round(meters[m].per_class(), 2).tolist())) for m in args.modes},
        "sec_per_image": {k: v / n for k, v in secs.items()},
    }
    if args.per_image:
        out["_per_image"] = {m: np.stack(meters[m].per_image) for m in args.modes}
    if dstats:
        out["defense_stats"] = {
            m: {"defended_fraction": float(np.mean([d["defended"] for d in v])),
                "mean_eps_frac": float(np.mean([d["mean_eps_frac"] for d in v])),
                "mean_score": float(np.mean([d["score"] for d in v])),
                "per_image": v}
            for m, v in dstats.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--split", default="test100", choices=["full500", "test100", "tune100"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--modes", nargs="+", default=["clean", "pgd"], choices=MODES)
    ap.add_argument("--defense", default="original", choices=sorted(DEFENSES))
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images of the split")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--no-cache", action="store_true", help="always regenerate PGD (and don't save it)")
    ap.add_argument("--clean-noise", action="store_true",
                    help="clean+def defends x + U(-4/255, 4/255) like upstream (Note B); default defends plain x")
    ap.add_argument("--tag", default=None, help="output name prefix (default: <model>_<defense>)")
    ap.add_argument("--eps-v", type=float, default=None,
                    help="defense budget in units of 1/255, overriding the preset (P4 global sweep)")
    ap.add_argument("--attack-seed", type=int, default=None,
                    help="use the cached PGD of this seed for every --seeds run (default: each run's own seed); "
                         "E1 noise oracle: same attacks, different defense randomness")
    ap.add_argument("--predict", default="single", choices=sorted(PREDICT_VARIANTS),
                    help="E2 / P5: average the prediction over views (default: single = the plain prediction)")
    ap.add_argument("--per-image", action="store_true",
                    help="also save per-image confusion matrices to <output>_perimage.npz (E1 oracle)")
    ap.add_argument("--def-opts", default=None,
                    help='JSON options for the adaptive defense, e.g. \'{"mode": "perpixel", "beta": 0.2}\'')
    args = ap.parse_args()

    device = torch.device("cuda")
    preset = dict(PRESETS[args.model])
    if args.eps_v is not None:
        preset["eps_v"] = args.eps_v / 255.0
    def_opts = json.loads(args.def_opts) if args.def_opts else {}
    if def_opts and args.defense not in DEFENSE_TAKES_OPTS:
        ap.error(f"--def-opts is only valid for {sorted(DEFENSE_TAKES_OPTS)}")
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    normalize = Normalizer(device)
    defense = None
    if any(m.endswith("+def") for m in args.modes):
        if args.defense in DEFENSE_TAKES_OPTS:
            def_opts["model_tag"] = args.model
        defense = DEFENSES[args.defense](model, normalize, preset["eps_v"], preset["num_views"], device=device,
                                         **def_opts)

    indices = split_indices(args.split)[:args.limit]
    tag = args.tag or f"{args.model}_{args.defense}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = {k: v for k, v in vars(args).items()}
    config.update(preset=preset, num_images=len(indices), git=git_hash())
    if hasattr(defense, "calib_info"):
        config["calib"] = defense.calib_info

    per_seed = {}
    for seed in args.seeds:
        print(f"== {tag} | {args.split} ({len(indices)} images) | seed {seed}", flush=True)
        res = run_seed(args, seed, model, normalize, defense, indices, device)
        per_seed[seed] = res
        suffix = f"_n{args.limit}" if args.limit else ""
        out = os.path.join(RESULTS_DIR, f"{tag}_{args.split}{suffix}_s{seed}.json")
        per_image = res.pop("_per_image", None)
        if per_image is not None:
            np.savez_compressed(out[:-len(".json")] + "_perimage.npz", indices=np.array(indices),
                                **{m.replace("+", "_"): v for m, v in per_image.items()})
        with open(out, "w") as f:
            json.dump({"config": config, "seed": seed, **res}, f, indent=2)
        print(f"   saved {os.path.relpath(out, ROOT_DIR)}", flush=True)

    print(f"\n== Summary: {tag} | {args.split} | seeds {args.seeds}")
    for m in args.modes:
        vals = np.array([per_seed[s]["miou"][m] for s in args.seeds])
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"  {m:10s} mIoU {vals.mean():6.2f} +/- {std:.2f}   ({', '.join(f'{v:.2f}' for v in vals)})")
    secs = defaultdict(list)
    for s in args.seeds:
        for k, v in per_seed[s]["sec_per_image"].items():
            secs[k].append(v)
    print("  sec/image: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in secs.items()))


if __name__ == "__main__":
    main()
