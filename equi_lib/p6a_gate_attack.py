"""P6a: gate-aware adaptive attack on the gated defense (IMPLEMENTATION_PLAN.md, P6a).

    python -m equi_lib.p6a_gate_attack --model clean --split tune100 --limit 50 --seed 0 \
        --lams 0 1 10 100 1000 --def-opts '{"mode": "gate+perimage"}' --tag equigate

The attacker knows the whole pipeline: gate score s, threshold tau, budget rule and defense.
Each of the `steps` iterations (L_inf, eps 4/255, step 1/255, random start):
  1. run the real pipeline on x_adv (gate decision + defense, no grad)  -> out
  2. BPDA: x_def = x_adv + (out - x_adv).detach()  (identity backward through the defense)
  3. objective per image = CE(model(x_def), y) - lam * relu(s(x_adv) - target),
     target = tau - 0.1 * |tau|  (just below the gate threshold; the hinge stops pushing once evaded).
     s is the gate score computed WITH gradient (same formula as the defense's gate).
  4. ascend sign(grad); keep per image the iterate with the highest pipeline CE seen so far.
lam = 0 is plain BPDA through the gated pipeline (no evasion term).

Each final x_adv is scored three ways:
  undefended : model(x_adv)
  pipeline   : the attacked defense (gate + budget), fresh run
  always_on  : one_step at global eps_v (no gate). Shows what evading the gate costs the attacker.
Writes results/improvements/p6/p6a_<tag>_<split>[_n<limit>]_s<seed>.json.
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
from .defense import LAYER_IND, _resize
from .metrics import ConfusionMeter, align_output
from .model import PRESETS, Normalizer, load_model
from .runner import git_hash
from .seed import set_seed

warnings.filterwarnings("ignore")
ATTACK = {"eps": 4.0 / 255.0, "step": 1.0 / 255.0}
OUT_DIR = os.path.join(RESULTS_DIR, "p6")


def gate_score_grad(model, normalize, x, signal):
    """Differentiable gate score s [B]: same formula as adaptive.signal_map(...).mean, but with gradient."""
    z0 = model(normalize(x))[LAYER_IND]
    if signal == "equi":
        zd = model(normalize(_resize(x, (int(x.size(2) * 0.5), int(x.size(3) * 0.5)))))[LAYER_IND]
        zv = _resize(zd, (z0.size(2), z0.size(3)))
        m = (F.softmax(z0, 1) - F.softmax(zv, 1)).norm(dim=1)
    elif signal in ("entropy", "neg_entropy"):
        logp = F.log_softmax(z0, 1)
        h = -(logp.exp() * logp).sum(1)
        m = h if signal == "entropy" else -h
    else:
        raise ValueError(signal)
    return m.flatten(1).mean(1)


def per_image_ce(logp, y):
    valid = y != 255
    y_safe = torch.where(valid, y, torch.zeros_like(y))
    ce = -logp.gather(1, y_safe.unsqueeze(1)).squeeze(1) * valid
    return ce.flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)


def gate_aware_attack(model, normalize, dfn, x, y, lam, steps, eps, step):
    """dfn: an AdaptiveDefense, or any defense callable (e.g. defenses.OriginalDefense) when lam == 0."""
    if lam > 0 and not hasattr(dfn, "tau"):
        raise ValueError("lam > 0 needs a gated AdaptiveDefense")
    target = dfn.tau - 0.1 * abs(dfn.tau) if lam > 0 else None
    x = x.detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, -x, 1.0 - x)
    best_delta = delta.clone()
    best_ce = torch.full((x.size(0),), -float("inf"), device=x.device)

    def forward(delta):
        xa = x + delta
        out = dfn(xa.detach(), y).detach()   # the defense takes its own gradient step, so no no_grad here
        if hasattr(dfn, "pop_stats"):
            dfn.pop_stats()
        xd = xa + (out - xa).detach()
        ce = per_image_ce(align_output(model(normalize(xd))[0], y.shape[-2:]), y)
        return xa, ce

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        xa, ce = forward(delta)
        better = ce.detach() > best_ce
        best_ce = torch.where(better, ce.detach(), best_ce)
        best_delta[better] = delta.detach()[better]
        obj = ce
        if lam > 0:
            s = gate_score_grad(model, normalize, xa, dfn.gate_signal)
            obj = ce - lam * F.relu(s - target)
        g = torch.autograd.grad(obj.sum(), delta)[0]
        delta = delta.detach() + step * g.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -x, 1.0 - x)
    _, ce = forward(delta)
    ce = ce.detach()
    better = ce > best_ce
    best_delta[better] = delta[better]
    return best_delta.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--split", default="tune100", choices=["tune100", "test100", "full500"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lams", type=float, nargs="+", default=[0, 1, 10, 100, 1000])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eps-v", type=float, default=None, help="defense budget in 1/255 (default: preset)")
    ap.add_argument("--def-opts", default='{"mode": "gate+perimage"}')
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--target", default="adaptive", choices=["adaptive", "original"],
                    help="defense in the attack loop: the adaptive pipeline (--def-opts) or the upstream "
                         "20-step defense (lam must be 0; 'pipeline' then means the original defense)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    preset = dict(PRESETS[args.model])
    if args.eps_v is not None:
        preset["eps_v"] = args.eps_v / 255.0
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    opts = json.loads(args.def_opts)
    if args.target == "original":
        from .defenses import OriginalDefense
        if any(l > 0 for l in args.lams):
            ap.error("--target original needs --lams 0")
        dfn = OriginalDefense(model, norm, preset["eps_v"], preset["num_views"], device=device)
        opts = {"target": "original"}
    else:
        dfn = AdaptiveDefense(model, norm, preset["eps_v"], preset["num_views"], device=device,
                              model_tag=args.model, **opts)
    always_on = AdaptiveDefense(model, norm, preset["eps_v"], preset["num_views"], device=device,
                                model_tag=args.model, mode="global", view_set=opts.get("view_set", "full"))
    indices = split_indices(args.split)[:args.limit]
    tau = getattr(dfn, "tau", float("nan"))
    print(f"== P6a {args.tag} | {args.model} | {args.split} ({len(indices)} images) | seed {args.seed} | "
          f"opts {opts} | tau {tau:.4f} | steps {args.steps}", flush=True)

    results = {}
    for lam in args.lams:
        set_seed(args.seed)
        meters = {k: ConfusionMeter() for k in ("undefended", "pipeline", "always_on")}
        stats, t0 = [], time.time()
        for b in range(0, len(indices), args.batch_size):
            idx = indices[b:b + args.batch_size]
            x, y = load_batch(idx, device)
            d = gate_aware_attack(model, norm, dfn, x, y, lam, args.steps, ATTACK["eps"], ATTACK["step"])
            xa = torch.clamp(x + d, 0.0, 1.0)
            x_pipe = dfn(xa, y).detach()       # defenses take a gradient step: call outside no_grad
            stats += dfn.pop_stats() if hasattr(dfn, "pop_stats") else \
                [{"defended": True, "mean_eps_frac": 1.0, "score": float("nan")}] * x.size(0)
            x_on = always_on(xa).detach()
            always_on.pop_stats()
            with torch.no_grad():
                meters["undefended"].update(model(norm(xa))[0], y)
                meters["pipeline"].update(model(norm(x_pipe))[0], y)
                meters["always_on"].update(model(norm(x_on))[0], y)
            done = min(b + args.batch_size, len(indices))
            print(f"  lam {lam:g} [{done:3d}/{len(indices)}] " +
                  " | ".join(f"{k} {m.miou():.2f}" for k, m in meters.items()) +
                  f" | defended {np.mean([s['defended'] for s in stats]):.2f}", flush=True)
        r = {k: m.miou() for k, m in meters.items()}
        r.update(defended_fraction=float(np.mean([s["defended"] for s in stats])),
                 mean_eps_frac=float(np.mean([s["mean_eps_frac"] for s in stats])),
                 mean_score=float(np.mean([s["score"] for s in stats])),
                 sec_per_image=(time.time() - t0) / len(indices), per_image=stats)
        results[f"{lam:g}"] = r
        print(f"  => lam {lam:g}: undefended {r['undefended']:.2f} | pipeline {r['pipeline']:.2f} | "
              f"always_on {r['always_on']:.2f} | gate defended {r['defended_fraction']:.2f} | "
              f"mean eps frac {r['mean_eps_frac']:.2f} | {r['sec_per_image']:.1f} s/img", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_n{args.limit}" if args.limit else ""
    out = os.path.join(OUT_DIR, f"p6a_{args.tag}_{args.split}{suffix}_s{args.seed}.json")
    config = dict(vars(args), preset=preset, tau=tau, attack=ATTACK, git=git_hash())
    with open(out, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=1)
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}")
    print("\n  lam      undefended  pipeline  always_on  defended  eps_frac")
    for lam, r in results.items():
        print(f"  {lam:8s} {r['undefended']:9.2f} {r['pipeline']:9.2f} {r['always_on']:9.2f} "
              f"{r['defended_fraction']:9.2f} {r['mean_eps_frac']:9.2f}")


if __name__ == "__main__":
    main()
