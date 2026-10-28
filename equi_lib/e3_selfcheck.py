"""E3 / P5b B1: does restored equivariance mean restored correctness? (IMPLEMENTATION_PLAN.md, P5b)

    python -m equi_lib.e3_selfcheck --model clean

On the first 50 test100 images (seed 0), with ours (the P4c locked config), for three inputs:
  clean  : x
  pgd    : cached PGD-50 (seed 0), the standard non-adaptive attack
  bpda   : BPDA-50 through ours, generated with the P6b attack loop (p6b_attacks.attack, same order and seed);
           saved to results/improvements/cache/bpda50_<model>_test100_n50_s0.pt and reused
the defended image x_def is scored with a held-out consistency error: softmax L2 between the prediction on x_def
and the prediction on a view the defense never optimises, mapped back (feature grid 43x85):
  clean model: horizontal flip and resize 0.75x (its 7 defense views have no flip and no 0.75x)
  adv model  : resize 0.75x (its defense views include a flip)
The per-pixel error is the mean over the held-out views. For reference, the gate signal (0.5x, NOT held out:
the defense has a 0.5x view) is computed on x_def too.

Reported (pre-registered in IMPLEMENTATION_PLAN.md, P5b):
  - Spearman rho between the image score (mean held-out error) and the image's defended mIoU, per input / pooled
  - pixel AUROC of the held-out error (upsampled to label size) for "defended prediction wrong" (valid labels)
  - image AUROC of the score for bpda vs clean and pgd vs clean ("is the failure visible?")
  "Proxy holds" on an input type: rho <= -0.5 and pixel AUROC >= 0.80. "Failure visible under BPDA": image AUROC
  (bpda vs clean) >= 0.90.
Writes results/improvements/e3/e3_<model>_test100_n50_s0.json.
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
from .attacks import PGD_PARAMS
from .cache import CACHE_DIR, PGDCache
from .data import load_batch, split_indices
from .defense import _resize
from .metrics import NUM_CLASSES, ConfusionMeter, align_output
from .model import PRESETS, Normalizer, load_model
from .p6b_attacks import ATTACK, OURS, Defended, attack, ce_per_image
from .runner import git_hash
from .seed import set_seed
from .stats import auroc

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(RESULTS_DIR, "e3")
HELD_OUT = {"clean": ("flip", 0.75), "adv": (0.75,)}
N_IMG, SEED, EVAL_SEED = 50, 0, 100
PIX_SAMPLE = 400_000


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def image_miou(h):
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.diag(h) / (h.sum(0) + h.sum(1) - np.diag(h))
    return float(np.nanmean(iou) * 100)


def view_error_grad(inner, x, views):
    """Mean softmax-L2 error [B, h, w] between x's prediction and each view's prediction mapped back."""
    z0 = inner._features(x)
    p0, hw = F.softmax(z0, 1), (z0.size(2), z0.size(3))
    errs = []
    for v in views:
        if v == "flip":
            zv = torch.flip(inner._features(torch.flip(x, dims=[3])), dims=[3])
        else:
            zv = _resize(inner._features(_resize(x, (int(x.size(2) * v), int(x.size(3) * v)))), hw)
        errs.append((p0 - F.softmax(zv, 1)).norm(dim=1))
    return torch.stack(errs).mean(0)


@torch.no_grad()
def view_error(inner, x, views):
    return view_error_grad(inner, x, views)


def selfcheck_aware_attack(model, norm, dfd, inner, views, x, y, lam, steps=50):
    """E3b: BPDA through the defense, maximising CE(defended) - lam * mean held-out error of the defended image.
    The held-out error is differentiated exactly through its forward passes; the purifier uses the identity
    backward (x_def = x_adv + (D(x_adv) - x_adv).detach()). Keeps per image the iterate with the highest objective."""
    eps, step = ATTACK["eps"], ATTACK["step"]
    x = x.detach()
    delta = torch.clamp(torch.empty_like(x).uniform_(-eps, eps), -x, 1.0 - x)
    best_delta, best = delta.clone(), torch.full((x.size(0),), -float("inf"), device=x.device)

    def objective(delta):
        xa = x + delta
        out = dfd(xa, y)
        xd = xa + (out - xa).detach()
        logp = align_output(model(norm(xd))[0], y.shape[-2:])
        return ce_per_image(logp, y) - lam * view_error_grad(inner, xd, views).flatten(1).mean(1)

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        obj = objective(delta)
        better = obj.detach() > best
        best = torch.where(better, obj.detach(), best)
        best_delta[better] = delta.detach()[better]
        g = torch.autograd.grad(obj.sum(), delta)[0]
        delta = torch.clamp(torch.clamp(delta.detach() + step * g.sign(), -eps, eps), -x, 1.0 - x)
    obj = objective(delta.detach()).detach()   # no torch.no_grad: the defense takes its own gradient step
    better = obj > best
    best_delta[better] = delta.detach()[better]
    return best_delta.detach()


def bpda_deltas(model_tag, model, norm, dfd, indices, device, lam=0.0, inner=None):
    """lam == 0: the plain BPDA-50 of P6b (p6b_attacks.attack). lam > 0: the E3b self-check-aware attack."""
    name = f"bpda50_{model_tag}" if lam == 0 else f"bpda50sc{lam:g}_{model_tag}"
    path = os.path.join(CACHE_DIR, f"{name}_test100_n{len(indices)}_s{SEED}.pt")
    if os.path.exists(path):
        return torch.load(path), None
    os.makedirs(CACHE_DIR, exist_ok=True)
    set_seed(SEED)                       # same order as p6b_attacks.main: attack a batch, then defend it
    meters = {k: ConfusionMeter() for k in ("undefended", "defended")}
    deltas, t0 = {}, time.time()
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        if lam == 0:
            d = attack(model, norm, dfd, x, y, "ce", "bpda", 50, 1, ATTACK["eps"], ATTACK["step"])
        else:
            d = selfcheck_aware_attack(model, norm, dfd, inner, HELD_OUT[model_tag], x, y, lam)
        xa = torch.clamp(x + d, 0.0, 1.0)
        x_def = dfd(xa, y)
        with torch.no_grad():
            meters["undefended"].update(model(norm(xa))[0], y)
            meters["defended"].update(model(norm(x_def))[0], y)
        for i, k in enumerate(idx):
            deltas[int(k)] = d[i].half().cpu()
        tag = "BPDA-50" if lam == 0 else f"BPDA-50 self-check-aware lam {lam:g}"
        print(f"  {tag} [{b + len(idx):2d}/{len(indices)}] undefended {meters['undefended'].miou():.2f} "
              f"defended {meters['defended'].miou():.2f}", flush=True)
    torch.save(deltas, path)
    check = {k: m.miou() for k, m in meters.items()}
    check["sec_per_image"] = (time.time() - t0) / len(indices)
    return deltas, check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--tag", default=None, help="output name prefix (default: e3, or e3b with --lams)")
    ap.add_argument("--lams", type=float, nargs="*", default=[],
                    help="E3b: also evaluate self-check-aware BPDA-50 with these lambdas (written to e3b_<model>_...)")
    args = ap.parse_args()
    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    eps_v, opts = OURS[args.model]
    dfn = AdaptiveDefense(model, norm, eps_v / 255.0, preset["num_views"], device=device, model_tag=args.model, **opts)
    dfd = Defended(dfn, False)
    indices = split_indices("test100")[:N_IMG]
    bpda, bpda_check = bpda_deltas(args.model, model, norm, dfd, indices, device)
    aware = {lam: bpda_deltas(args.model, model, norm, dfd, indices, device, lam, dfn.inner) for lam in args.lams}
    pgd_cache = PGDCache(args.model, SEED, PGD_PARAMS)
    eps = ATTACK["eps"]

    set_seed(EVAL_SEED)
    inputs = ["clean", "pgd", "bpda"] + [f"sc{lam:g}" for lam in args.lams]
    rec = {k: {"score": [], "gate_score": [], "miou": [], "acc": []} for k in inputs}
    pix = {k: ([], []) for k in rec}
    meters = {k: ConfusionMeter() for k in rec}
    rng = np.random.default_rng(0)
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        d_pgd = torch.clamp(torch.clamp(pgd_cache.get(idx, device), -eps, eps), -x, 1.0 - x)
        d_bpda = torch.stack([bpda[int(k)] for k in idx]).to(device).float()
        d_bpda = torch.clamp(torch.clamp(d_bpda, -eps, eps), -x, 1.0 - x)
        cases = [("clean", x), ("pgd", torch.clamp(x + d_pgd, 0, 1)), ("bpda", torch.clamp(x + d_bpda, 0, 1))]
        for lam, (dl, _) in aware.items():
            d_sc = torch.stack([dl[int(k)] for k in idx]).to(device).float()
            cases.append((f"sc{lam:g}", torch.clamp(x + torch.clamp(torch.clamp(d_sc, -eps, eps), -x, 1.0 - x), 0, 1)))
        for name, inp in cases:
            x_def = dfd(inp, y)
            with torch.no_grad():
                logp = align_output(model(norm(x_def))[0], y.shape[-2:])
                meters[name].update(logp, y)
                pred = logp.argmax(1)
                e = view_error(dfn.inner, x_def, HELD_OUT[args.model])
                g = dfn.inner.consistency_map(x_def)
                e_up = F.interpolate(e.unsqueeze(1), size=y.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            for i in range(x.size(0)):
                valid = (y[i] != 255)
                wrong = (pred[i] != y[i]) & valid
                h = np.bincount(NUM_CLASSES * y[i][valid].cpu().numpy() + pred[i][valid].cpu().numpy(),
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
                rec[name]["score"].append(float(e[i].mean()))
                rec[name]["gate_score"].append(float(g[i].mean()))
                rec[name]["miou"].append(image_miou(h))
                rec[name]["acc"].append(float(1 - wrong.sum() / valid.sum()))
                sv, wv = e_up[i][valid].cpu().numpy(), wrong[valid].cpu().numpy()
                take = rng.choice(sv.size, size=min(sv.size, PIX_SAMPLE // N_IMG), replace=False)
                pix[name][0].append(sv[take])
                pix[name][1].append(wv[take])
        print(f"  [{b + len(idx):2d}/{len(indices)}] defended mIoU " +
              " | ".join(f"{k} {m.miou():.2f}" for k, m in meters.items()), flush=True)

    res = {"model": args.model, "held_out_views": [str(v) for v in HELD_OUT[args.model]], "num_images": len(indices),
           "defended_miou": {k: m.miou() for k, m in meters.items()}, "bpda_generation_check": bpda_check,
           "per_input": {}, "git": git_hash()}
    print(f"\n== E3 self-check, {args.model} model (test100 first {len(indices)}, seed {SEED}); held-out views "
          f"{res['held_out_views']}")
    print(f"  {'input':6s} {'def mIoU':>8s} {'score mean':>10s} {'gate score':>10s} {'rho(score,mIoU)':>16s} "
          f"{'pixel AUROC':>11s}  proxy holds")
    for k in rec:
        s, mi = np.array(rec[k]["score"]), np.array(rec[k]["miou"])
        rho = spearman(s, mi)
        pa = auroc(np.concatenate(pix[k][0]), np.concatenate(pix[k][1]))
        holds = rho <= -0.5 and pa >= 0.80
        res["per_input"][k] = dict(rec[k], rho=rho, pixel_auroc=pa, proxy_holds=bool(holds))
        print(f"  {k:6s} {meters[k].miou():8.2f} {s.mean():10.4f} {np.mean(rec[k]['gate_score']):10.4f} "
              f"{rho:16.3f} {pa:11.3f}  {holds}")
    s_all = np.concatenate([rec[k]["score"] for k in rec])
    m_all = np.concatenate([rec[k]["miou"] for k in rec])
    res["pooled_rho"] = spearman(s_all, m_all)
    res["aware_generation_check"] = {f"sc{lam:g}": chk for lam, (_, chk) in aware.items()}
    for att in inputs[1:]:
        sc = np.r_[rec["clean"]["score"], rec[att]["score"]]
        lb = np.r_[np.zeros(len(rec["clean"]["score"])), np.ones(len(rec[att]["score"]))]
        res[f"image_auroc_{att}_vs_clean"] = auroc(sc, lb)
    res["failure_visible_under_bpda"] = bool(res["image_auroc_bpda_vs_clean"] >= 0.90)
    print(f"  pooled rho {res['pooled_rho']:.3f} | image AUROC pgd vs clean {res['image_auroc_pgd_vs_clean']:.3f} | "
          f"bpda vs clean {res['image_auroc_bpda_vs_clean']:.3f} -> failure visible under BPDA: "
          f"{res['failure_visible_under_bpda']}")
    for lam in args.lams:
        k = f"sc{lam:g}"
        print(f"  E3b self-check-aware lam {lam:g}: defended mIoU {meters[k].miou():.2f} | held-out error "
              f"{np.mean(rec[k]['score']):.4f} | image AUROC vs clean {res[f'image_auroc_{k}_vs_clean']:.3f}")
    if bpda_check:
        print(f"  BPDA generation check (p6b seed 0 on the same images: see IMPLEMENTATION_PLAN.md P6b): {bpda_check}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.tag or ('e3b' if args.lams else 'e3')}_{args.model}_test100_n{len(indices)}_s{SEED}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
