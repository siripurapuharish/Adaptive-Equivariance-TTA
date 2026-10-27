"""P6b: the remaining attacks against the locked P4c config ("ours") and the original defense.

    python -m equi_lib.p6b_attacks --model clean --target ours --loss segpgd --through none \
        --split test100 --seed 0 --tag segpgd50

Attack (L_inf, eps 4/255, step 1/255, random start, `--steps` iterations):
  --loss     ce | segpgd (Gu et al. 2022) | cospgd (Agnihotri et al. 2023) | noise (random +-eps corners, control)
  --through  none : gradients through the undefended model only; the defense is applied afterwards
                    (the standard "PGD+def" protocol; last iterate, as in attacks.pgd)
             bpda : identity backward through the defense (Athalye 2018), x_def = x + (D(x) - x).detach();
                    keeps per image the iterate with the highest defended CE (as in P6a)
  --eot K    average the BPDA objective over K defense runs (random start / views / jitter differ)
  --fixed-seed  re-seed the RNG before every defense call, so the attacker faces a deterministic defense
                (Croce et al. 2022); the evaluation uses the same seed
Targets: ours = the P4c locked config (clean: gate+perimage up1.5 eps 6; adv: global up1.5 eps 4),
         original = upstream 20-step defense at the paper's eps_v.
Each final x_adv is scored undefended and defended (fresh defense run).
Writes results/improvements/p6/p6b_<tag>_<target>_<model>_<split>[_n<limit>]_s<seed>.json.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from . import RESULTS_DIR, ROOT_DIR
from .adaptive import AdaptiveDefense
from .data import load_batch, split_indices
from .metrics import ConfusionMeter, align_output
from .model import PRESETS, Normalizer, load_model
from .runner import git_hash
from .seed import set_seed

warnings.filterwarnings("ignore")
ATTACK = {"eps": 4.0 / 255.0, "step": 1.0 / 255.0}
OUT_DIR = os.path.join(RESULTS_DIR, "p6")
FIXED_DEF_SEED = 12345
# P4c locked config (IMPLEMENTATION_PLAN.md, P4c "Locked config")
OURS = {"clean": (6.0, {"mode": "gate+perimage", "view_set": "up1.5"}),
        "adv": (4.0, {"mode": "global", "view_set": "up1.5"})}


def pixel_loss(logp, y, kind, t=1, steps=1):
    """Per-image loss [B] from log-softmax output logp [B,C,H,W] (already at label size)."""
    valid = y != 255
    y_safe = torch.where(valid, y, torch.zeros_like(y))
    ce = -logp.gather(1, y_safe.unsqueeze(1)).squeeze(1)
    n = valid.flatten(1).sum(1).clamp_min(1)
    if kind == "ce":
        w = valid.float()
    elif kind == "segpgd":
        lam = (t - 1) / (2 * steps)
        correct = (logp.argmax(1) == y) & valid
        w = (1 - lam) * correct.float() + lam * (valid & ~correct).float()
    elif kind == "cospgd":
        # CosPGD: weight each pixel's CE by cos(softmax, one-hot(y)) (detached), so pixels that are
        # still confidently correct get more weight and already-flipped pixels less.
        p = logp.exp()
        cos = p.gather(1, y_safe.unsqueeze(1)).squeeze(1) / p.norm(dim=1).clamp_min(1e-12)
        w = cos.detach() * valid
    else:
        raise ValueError(kind)
    return (ce * w).flatten(1).sum(1) / n


def ce_per_image(logp, y):
    return pixel_loss(logp, y, "ce")


class Defended:
    """Calls a defense, optionally with a fixed RNG seed (deterministic defense)."""

    def __init__(self, dfn, fixed):
        self.dfn, self.fixed = dfn, fixed

    def __call__(self, x, y):
        if self.fixed:
            set_seed(FIXED_DEF_SEED)
        out = self.dfn(x.detach(), y).detach()   # defenses take their own gradient step: no no_grad here
        if hasattr(self.dfn, "pop_stats"):
            self._last = self.dfn.pop_stats()
        else:
            self._last = [{"defended": True, "mean_eps_frac": 1.0, "score": float("nan")}] * x.size(0)
        return out


def attack(model, norm, dfd, x, y, loss, through, steps, eot, eps, step):
    x = x.detach()
    if loss == "noise":
        d = eps * torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype).mul(2).sub(1)
        return torch.clamp(d, -x, 1.0 - x)
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, -x, 1.0 - x)
    best_delta, best = delta.clone(), torch.full((x.size(0),), -float("inf"), device=x.device)

    def objective(delta, t):
        xa = x + delta
        if through == "none":
            logp = align_output(model(norm(xa))[0], y.shape[-2:])
            return pixel_loss(logp, y, loss, t, steps), None
        obj, sel = 0.0, 0.0
        for _ in range(eot):
            out = dfd(xa, y)
            xd = xa + (out - xa).detach()
            logp = align_output(model(norm(xd))[0], y.shape[-2:])
            obj = obj + pixel_loss(logp, y, loss, t, steps) / eot
            sel = sel + ce_per_image(logp, y).detach() / eot
        return obj, sel

    for t in range(1, steps + 1):
        delta = delta.detach().requires_grad_(True)
        obj, sel = objective(delta, t)
        if sel is not None:
            better = sel > best
            best = torch.where(better, sel, best)
            best_delta[better] = delta.detach()[better]
        g = torch.autograd.grad(obj.sum(), delta)[0]
        delta = delta.detach() + step * g.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -x, 1.0 - x)
    if through == "none":
        return delta.detach()
    with torch.enable_grad():
        _, sel = objective(delta.detach(), steps)
    better = sel > best
    best_delta[better] = delta.detach()[better]
    return best_delta.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--target", choices=["ours", "original"], required=True)
    ap.add_argument("--loss", choices=["ce", "segpgd", "cospgd", "noise"], default="ce")
    ap.add_argument("--through", choices=["none", "bpda"], default="bpda")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eot", type=int, default=1)
    ap.add_argument("--fixed-seed", action="store_true")
    ap.add_argument("--split", default="test100", choices=["tune100", "test100", "full500"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.through == "none" and (args.eot > 1 or args.fixed_seed):
        ap.error("--eot / --fixed-seed only make sense with --through bpda")

    device = torch.device("cuda")
    preset = dict(PRESETS[args.model])
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    if args.target == "ours":
        eps_v, opts = OURS[args.model]
        eps_v /= 255.0
        dfn = AdaptiveDefense(model, norm, eps_v, preset["num_views"], device=device, model_tag=args.model, **opts)
    else:
        from .defenses import OriginalDefense
        eps_v, opts = preset["eps_v"], {"target": "original"}
        dfn = OriginalDefense(model, norm, eps_v, preset["num_views"], device=device)
    dfd = Defended(dfn, args.fixed_seed)
    indices = split_indices(args.split)
    if args.limit:
        indices = indices[:args.limit]
    print(f"== P6b {args.tag} | {args.target} | {args.model} | {args.split} ({len(indices)} images) | "
          f"seed {args.seed} | loss {args.loss} through {args.through} steps {args.steps} eot {args.eot} "
          f"fixed {args.fixed_seed} | eps_v {eps_v * 255:.1f}/255 {opts}", flush=True)

    set_seed(args.seed)
    meters = {k: ConfusionMeter() for k in ("undefended", "defended")}
    stats, t0 = [], time.time()
    for b in range(0, len(indices), args.batch_size):
        idx = indices[b:b + args.batch_size]
        x, y = load_batch(idx, device)
        d = attack(model, norm, dfd, x, y, args.loss, args.through, args.steps, args.eot,
                   ATTACK["eps"], ATTACK["step"])
        xa = torch.clamp(x + d, 0.0, 1.0)
        x_def = dfd(xa, y)
        stats += dfd._last
        with torch.no_grad():
            meters["undefended"].update(model(norm(xa))[0], y)
            meters["defended"].update(model(norm(x_def))[0], y)
        done = min(b + args.batch_size, len(indices))
        print(f"  [{done:3d}/{len(indices)}] " + " | ".join(f"{k} {m.miou():.2f}" for k, m in meters.items()) +
              f" | gate defended {np.mean([s['defended'] for s in stats]):.2f}", flush=True)
    r = {k: m.miou() for k, m in meters.items()}
    r.update(defended_fraction=float(np.mean([s["defended"] for s in stats])),
             mean_eps_frac=float(np.mean([s["mean_eps_frac"] for s in stats])),
             sec_per_image=(time.time() - t0) / len(indices), per_image=stats)
    print(f"  => undefended {r['undefended']:.2f} | defended {r['defended']:.2f} | gate defended "
          f"{r['defended_fraction']:.2f} | {r['sec_per_image']:.1f} s/img", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_n{args.limit}" if args.limit else ""
    out = os.path.join(OUT_DIR, f"p6b_{args.tag}_{args.target}_{args.model}_{args.split}{suffix}_s{args.seed}.json")
    config = dict(vars(args), preset=preset, eps_v=eps_v, def_opts=opts, attack=ATTACK, git=git_hash())
    with open(out, "w") as f:
        json.dump({"config": config, "results": r}, f, indent=1)
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
