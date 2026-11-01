"""E5: trust map, a per-pixel "don't trust" output for the v2 system (IMPLEMENTATION_PLAN.md, E5).

    python -m equi_lib.e5_trust --model clean --phase tune     # pick our signal + threshold on tune100
    python -m equi_lib.e5_trust --model clean --phase test     # report on test100 (first 50), incl. attacks

The defense and the prediction are the locked v2 ones and are not changed; the trust map is an extra output.
Signals (per pixel, higher = less trust, computed on the defended image, upsampled to label size):
  msp      1 - max softmax of the v2 prediction                         (baseline)
  entropy  entropy of the v2 prediction                                 (baseline)
  heldout  E3 held-out-view error (clean model flip + 0.75x; adv 0.75x)
  avgdis   disagreement of the 4 averaging views (mean softmax L2 to their mean)
  fresh    held-out error on views used by neither the defense nor the averaging (0.625x, 0.875x)
Tune: ours = the equivariance signal with the lowest mean AURC over clean + PGD; threshold t = 95th percentile
of the signal on clean valid pixels. Test: clean, PGD-50, BPDA-50 through v2, trust-aware BPDA-50 (lam 1/10/100).
Writes results/improvements/e5/e5_{tune,test}_<model>....json; attacks cached in results/improvements/cache/.
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
from .attacks import PGD_PARAMS, pgd
from .averaging import _logp, predict_logp
from .cache import CACHE_DIR, PGDCache
from .data import load_batch, split_indices
from .e3_selfcheck import HELD_OUT, view_error_grad
from .metrics import NUM_CLASSES, ConfusionMeter
from .model import PRESETS, Normalizer, load_model
from .p6b_attacks import ATTACK, Defended, attack, ce_per_image
from .runner import git_hash
from .seed import set_seed
from .stats import auroc

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(RESULTS_DIR, "e5")
# locked v2 configs (IMPLEMENTATION_PLAN.md, v2 confirmation): eps_v in 1/255, AdaptiveDefense options
V2 = {"clean": (6.0, {"mode": "gate+perimage", "view_set": "up1.5"}),
      "adv": (4.0, {"mode": "global", "view_set": "up1.5-up"})}
SIGNALS = ("msp", "entropy", "heldout", "avgdis", "fresh")
CANDIDATES = ("heldout", "avgdis", "fresh")
FRESH = (0.625, 0.875)
LAMS = (1.0, 10.0, 100.0)
COVERAGES = (0.9, 0.8, 0.7, 0.5)
FLAG_PCT = 95.0
PER_IMG = 40_000
N_TEST, SEED, EVAL_SEED = 50, 0, 100


# ---------------------------------------------------------------- signals

def _up(e, hw):
    return F.interpolate(e.unsqueeze(1), size=hw, mode="bilinear", align_corners=False)[:, 0]


def trust_maps(model, norm, inner, x, out_hw, names, model_tag):
    """v2 prediction (mean log-softmax over identity, flip, 0.75x, 1.25x; same order and arithmetic as
    averaging.predict_logp 'avg_rs') and the requested signals [B, H, W] at out_hw. Differentiable."""
    out_hw = tuple(out_hw)
    ps = [_logp(model, norm, x, out_hw),
          torch.flip(_logp(model, norm, torch.flip(x, dims=[3]), out_hw), dims=[3])]
    for s in (0.75, 1.25):
        xs = F.interpolate(x, size=(int(x.size(2) * s), int(x.size(3) * s)), mode="bilinear", align_corners=False)
        ps.append(_logp(model, norm, xs, out_hw))
    logp = ps[0]
    for p in ps[1:]:
        logp = logp + p
    logp = logp / len(ps)
    maps = {}
    if "msp" in names or "entropy" in names:
        q = F.softmax(logp, 1)
        if "msp" in names:
            maps["msp"] = 1.0 - q.max(1)[0]
        if "entropy" in names:
            maps["entropy"] = -(q * F.log_softmax(logp, 1)).sum(1)
    if "avgdis" in names:
        qs = [F.softmax(p, 1) for p in ps]
        qm = sum(qs) / len(qs)
        maps["avgdis"] = torch.stack([(q - qm).norm(dim=1) for q in qs]).mean(0)
    if "heldout" in names:
        maps["heldout"] = _up(view_error_grad(inner, x, HELD_OUT[model_tag]), out_hw)
    if "fresh" in names:
        maps["fresh"] = _up(view_error_grad(inner, x, FRESH), out_hw)
    return logp, maps


# ---------------------------------------------------------------- attacks

def trust_aware_attack(model, norm, dfd, inner, model_tag, signal, x, y, lam, steps=50):
    """BPDA-50 through v2 maximising CE(v2 prediction) - lam * mean trust signal of the defended image.
    Identity backward through the purifier; exact gradient through the averaging and the signal.
    Keeps per image the iterate with the highest objective (as e3_selfcheck.selfcheck_aware_attack)."""
    eps, step = ATTACK["eps"], ATTACK["step"]
    x = x.detach()
    delta = torch.clamp(torch.empty_like(x).uniform_(-eps, eps), -x, 1.0 - x)
    best_delta, best = delta.clone(), torch.full((x.size(0),), -float("inf"), device=x.device)

    def objective(delta):
        xa = x + delta
        out = dfd(xa, y)
        xd = xa + (out - xa).detach()
        logp, maps = trust_maps(model, norm, inner, xd, y.shape[-2:], (signal,), model_tag)
        return ce_per_image(logp, y) - lam * maps[signal].flatten(1).mean(1)

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


def attack_deltas(model_tag, model, norm, dfd, inner, indices, device, lam=0.0, signal=None):
    """lam == 0: BPDA-50 through v2 (p6b_attacks.attack with predict avg_rs). lam > 0: trust-aware BPDA-50."""
    name = "e5_bpda50v2" if lam == 0 else f"e5_bpda50v2_{signal}{lam:g}"
    path = os.path.join(CACHE_DIR, f"{name}_{model_tag}_test100_n{len(indices)}_s{SEED}.pt")
    if os.path.exists(path):
        return torch.load(path), None
    os.makedirs(CACHE_DIR, exist_ok=True)
    set_seed(SEED)
    meter, deltas, t0 = ConfusionMeter(), {}, time.time()
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        if lam == 0:
            d = attack(model, norm, dfd, x, y, "ce", "bpda", 50, 1, ATTACK["eps"], ATTACK["step"], predict="avg_rs")
        else:
            d = trust_aware_attack(model, norm, dfd, inner, model_tag, signal, x, y, lam)
        x_def = dfd(torch.clamp(x + d, 0.0, 1.0), y)
        with torch.no_grad():
            meter.update(predict_logp(model, norm, x_def, "avg_rs", y.shape[-2:]), y)
        for i, k in enumerate(idx):
            deltas[int(k)] = d[i].half().cpu()
        print(f"  {name} [{b + len(idx):2d}/{len(indices)}] defended v2 mIoU {meter.miou():.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    torch.save(deltas, path)
    return deltas, {"defended_v2_miou": meter.miou(), "sec_per_image": (time.time() - t0) / len(indices)}


# ---------------------------------------------------------------- metrics

def aurc(sig, wrong):
    """Area under the risk-coverage curve: mean error rate over all prefixes of the most-trusted pixels."""
    o = np.argsort(sig, kind="stable")
    return float((np.cumsum(wrong[o]) / np.arange(1, o.size + 1)).mean())


def hist_metrics(gt, pred):
    h = np.bincount(NUM_CLASSES * gt.astype(np.int64) + pred, minlength=NUM_CLASSES ** 2)
    h = h.reshape(NUM_CLASSES, NUM_CLASSES)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.diag(h) / (h.sum(0) + h.sum(1) - np.diag(h))
    return float(np.trace(h) / max(h.sum(), 1) * 100), float(np.nanmean(iou) * 100)


def summarise(rec, thresholds):
    """rec: {'gt','pred', signal: values} over sampled valid pixels of one input type."""
    gt, pred = rec["gt"], rec["pred"]
    wrong = gt != pred
    acc, miou = hist_metrics(gt, pred)
    out = {"pixels": int(gt.size), "error_rate": float(wrong.mean() * 100), "acc": acc, "miou": miou, "signals": {}}
    for s in SIGNALS:
        v = rec[s]
        o = np.argsort(v, kind="stable")
        cov = {}
        for c in COVERAGES:
            k = o[:int(round(c * v.size))]
            a, m = hist_metrics(gt[k], pred[k])
            cov[f"{int(c * 100)}"] = {"acc": a, "miou": m}
        d = {"aurc": aurc(v, wrong) * 100, "pixel_auroc": auroc(v, wrong), "mean": float(v.mean()), "coverage": cov}
        if s in thresholds:
            flag = v > thresholds[s]
            d["at_t"] = {"t": thresholds[s], "flagged": float(flag.mean() * 100),
                         "silent_error": float((wrong & ~flag).mean() * 100),
                         "caught_error": float((wrong & flag).mean() * 100)}
        out["signals"][s] = d
    return out


# ---------------------------------------------------------------- evaluation loop

def evaluate(model_tag, model, norm, dfd, dfn, indices, device, inputs_fn):
    """inputs_fn(idx, x) -> [(name, attacked input)]. Returns per-input sampled pixel records, mIoU, gate stats."""
    set_seed(EVAL_SEED)
    rng = np.random.default_rng(0)
    recs, meters, gate = {}, {}, {}
    t0 = time.time()
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        for name, inp in inputs_fn(idx, x):
            x_def = dfd(inp, y)
            gate.setdefault(name, []).extend(int(d["defended"]) for d in getattr(dfd, "_last", []) or [])
            with torch.no_grad():
                logp, maps = trust_maps(model, norm, dfn.inner, x_def, y.shape[-2:], SIGNALS, model_tag)
                if b == 0 and name == "clean":   # the v2 prediction here must be exactly averaging's avg_rs
                    ref = predict_logp(model, norm, x_def, "avg_rs", y.shape[-2:])
                    diff = float((ref - logp).abs().max())
                    assert diff < 1e-4, f"trust_maps prediction differs from averaging avg_rs (max {diff})"
            meters.setdefault(name, ConfusionMeter()).update(logp, y)
            pred = logp.argmax(1)
            r = recs.setdefault(name, {k: [] for k in ("gt", "pred") + SIGNALS})
            for i in range(x.size(0)):
                valid = torch.nonzero((y[i] != 255).flatten()).squeeze(1).cpu().numpy()
                take = torch.from_numpy(np.sort(rng.choice(valid, size=min(PER_IMG, valid.size), replace=False)))
                take = take.to(device)
                r["gt"].append(y[i].flatten()[take].cpu().numpy().astype(np.uint8))
                r["pred"].append(pred[i].flatten()[take].cpu().numpy().astype(np.uint8))
                for s in SIGNALS:
                    r[s].append(maps[s][i].flatten()[take].float().cpu().numpy())
        print(f"  eval [{b + len(idx):3d}/{len(indices)}] v2 mIoU " +
              " | ".join(f"{k} {m.miou():.2f}" for k, m in meters.items()) + f" ({time.time() - t0:.0f} s)",
              flush=True)
    recs = {n: {k: np.concatenate(v) for k, v in r.items()} for n, r in recs.items()}
    return recs, {n: m.miou() for n, m in meters.items()}, {n: float(np.mean(g)) if g else None
                                                            for n, g in gate.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--phase", choices=["tune", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images (output gets _n<N>)")
    args = ap.parse_args()
    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    eps_v, opts = V2[args.model]
    dfn = AdaptiveDefense(model, norm, eps_v / 255.0, preset["num_views"], device=device, model_tag=args.model,
                          **opts)
    dfd = Defended(dfn, False)
    eps = ATTACK["eps"]
    pgd_cache = PGDCache(args.model, SEED, PGD_PARAMS)
    os.makedirs(OUT_DIR, exist_ok=True)

    def pgd_input(idx, x):
        if pgd_cache.has(idx):
            d = torch.clamp(pgd_cache.get(idx, device), -eps, eps)
        else:
            _, y = load_batch(idx, device)
            d = pgd(model, norm, x, y, **PGD_PARAMS)
        return torch.clamp(x + torch.clamp(d, -x, 1.0 - x), 0.0, 1.0)

    if args.phase == "tune":
        indices = split_indices("tune100")[:args.limit]
        recs, miou, gate = evaluate(args.model, model, norm, dfd, dfn, indices, device,
                                    lambda idx, x: [("clean", x), ("pgd", pgd_input(idx, x))])
        res = {"model": args.model, "split": "tune100", "num_images": len(indices), "v2_miou": miou,
               "gate_defended_fraction": gate, "per_input": {}, "git": git_hash()}
        for n, r in recs.items():
            res["per_input"][n] = summarise(r, {})
        mean_aurc = {s: float(np.mean([res["per_input"][n]["signals"][s]["aurc"] for n in recs])) for s in SIGNALS}
        ours = min(CANDIDATES, key=lambda s: mean_aurc[s])
        res.update(mean_aurc=mean_aurc, ours=ours,
                   thresholds={s: float(np.percentile(recs["clean"][s], FLAG_PCT)) for s in SIGNALS})
        print(f"\n== E5 tune, {args.model} model (tune100, {len(indices)} images): v2 mIoU {miou}")
        print(f"  {'signal':8s} " + " ".join(f"{n + ' AURC':>11s} {n + ' AUROC':>12s}" for n in recs) +
              f" {'mean AURC':>10s}")
        for s in SIGNALS:
            print(f"  {s:8s} " + " ".join(f"{res['per_input'][n]['signals'][s]['aurc']:11.2f} "
                                          f"{res['per_input'][n]['signals'][s]['pixel_auroc']:12.3f}" for n in recs) +
                  f" {mean_aurc[s]:10.2f}")
        print(f"  ours = {ours} (lowest mean AURC among {CANDIDATES}); t(ours) = {res['thresholds'][ours]:.4f}, "
              f"t(msp) = {res['thresholds']['msp']:.4f}")
        out = os.path.join(OUT_DIR, f"e5_tune_{args.model}{'_n%d' % args.limit if args.limit else ''}.json")
    else:
        tune_path = os.path.join(OUT_DIR, f"e5_tune_{args.model}.json")
        with open(tune_path) as f:
            tune = json.load(f)
        ours = tune["ours"]
        thr = {s: tune["thresholds"][s] for s in (ours, "msp")}
        indices = split_indices("test100")[:args.limit or N_TEST]
        attacks = {"bpda": attack_deltas(args.model, model, norm, dfd, dfn.inner, indices, device)}
        for lam in LAMS:
            attacks[f"tr{lam:g}"] = attack_deltas(args.model, model, norm, dfd, dfn.inner, indices, device, lam, ours)

        def inputs(idx, x):
            cases = [("clean", x), ("pgd", pgd_input(idx, x))]
            for name, (dl, _) in attacks.items():
                d = torch.stack([dl[int(k)] for k in idx]).to(device).float()
                cases.append((name, torch.clamp(x + torch.clamp(torch.clamp(d, -eps, eps), -x, 1.0 - x), 0, 1)))
            return cases

        recs, miou, gate = evaluate(args.model, model, norm, dfd, dfn, indices, device, inputs)
        res = {"model": args.model, "split": "test100", "num_images": len(indices), "ours": ours, "thresholds": thr,
               "v2_miou": miou, "gate_defended_fraction": gate,
               "attack_generation_check": {k: chk for k, (_, chk) in attacks.items()},
               "per_input": {n: summarise(r, thr) for n, r in recs.items()}, "git": git_hash()}
        P = res["per_input"]
        attacked = ["bpda"] + [f"tr{lam:g}" for lam in LAMS]
        worst = max(attacked, key=lambda n: P[n]["signals"][ours]["at_t"]["silent_error"])
        res["C1_aurc_ours_lt_msp_on_bpda"] = bool(P["bpda"]["signals"][ours]["aurc"] < P["bpda"]["signals"]["msp"]["aurc"])
        res["C2_worst_input"] = worst
        res["C2_worst_silent_error"] = P[worst]["signals"][ours]["at_t"]["silent_error"]
        res["C2_bpda_error_rate"] = P["bpda"]["error_rate"]
        res["C2_no_silent_failure"] = bool(res["C2_worst_silent_error"] <= 0.5 * res["C2_bpda_error_rate"])
        print(f"\n== E5 test, {args.model} model (test100 first {len(indices)}, seed {SEED}); ours = {ours}")
        print(f"  {'input':6s} {'v2 mIoU':>7s} {'err%':>6s} | " + " ".join(f"{s + ' AURC':>12s}" for s in SIGNALS) +
              f" | {'flag% ours':>10s} {'silent% ours':>12s} {'flag% msp':>9s} {'silent% msp':>11s}")
        for n, p in P.items():
            so, sm = p["signals"][ours]["at_t"], p["signals"]["msp"]["at_t"]
            print(f"  {n:6s} {miou[n]:7.2f} {p['error_rate']:6.2f} | " +
                  " ".join(f"{p['signals'][s]['aurc']:12.2f}" for s in SIGNALS) +
                  f" | {so['flagged']:10.2f} {so['silent_error']:12.2f} {sm['flagged']:9.2f} {sm['silent_error']:11.2f}")
        print(f"  C1 (AURC ours < msp on BPDA): {res['C1_aurc_ours_lt_msp_on_bpda']}")
        print(f"  C2 (worst silent error {res['C2_worst_silent_error']:.2f} on {worst} <= 0.5 x BPDA error "
              f"{res['C2_bpda_error_rate']:.2f}): {res['C2_no_silent_failure']}")
        out = os.path.join(OUT_DIR, f"e5_test_{args.model}_n{len(indices)}_s{SEED}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
