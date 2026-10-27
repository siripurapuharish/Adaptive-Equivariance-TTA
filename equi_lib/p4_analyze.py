"""Applies the pre-registered P4 decision rule (IMPLEMENTATION_PLAN.md, P4) to the tune100 results.

    python -m equi_lib.p4_analyze [--split tune100]

Reads results/improvements/p4_<model>_<variant>_<split>_s{0,1,2}.json and prints:
  - per variant: clean+def / PGD+def mean +/- std over seeds, mean eps fraction, defended fraction, sec/img
  - placement test: perpixel minus perimage and minus shuffled (paired per seed: same PGD attacks per seed)
  - global-curve test: perpixel's distance above the global eps_v sweep curve (piecewise linear between
    sweep points, including eps 0 = undefended; upper envelope; no extrapolation)
  - clean-model gate test
Writes results/improvements/p4/decision_<split>.json.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from . import RESULTS_DIR, ROOT_DIR

THR = 1.5
OUT_DIR = os.path.join(RESULTS_DIR, "p4")


def load(model, variant, split):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"p4_{model}_{variant}_{split}_s*.json")))
    if not files:
        return None
    rows = [json.load(open(f)) for f in files]
    seeds = [r["seed"] for r in rows]
    out = {"seeds": seeds, "git": sorted({r["config"]["git"] for r in rows})}
    for m in ("clean", "pgd", "clean+def", "pgd+def"):
        if m in rows[0]["miou"]:
            out[m] = np.array([r["miou"][m] for r in rows])
    out["sec"] = {k: float(np.mean([r["sec_per_image"][k] for r in rows])) for k in rows[0]["sec_per_image"]}
    if "defense_stats" in rows[0]:
        out["stats"] = {m: {k: float(np.mean([r["defense_stats"][m][k] for r in rows]))
                            for k in ("defended_fraction", "mean_eps_frac")} for m in rows[0]["defense_stats"]}
    return out


def ms(a):
    return f"{a.mean():6.2f} +/- {a.std(ddof=1) if len(a) > 1 else 0:.2f}"


def envelope(points, xq, x_key, y_key):
    """Max y on the polyline (points in eps order) at x = xq; None if xq is outside every segment."""
    best = None
    for p, q in zip(points[:-1], points[1:]):
        x0, x1, y0, y1 = p[x_key], q[x_key], p[y_key], q[y_key]
        lo, hi = min(x0, x1), max(x0, x1)
        if lo <= xq <= hi:
            y = y0 if x1 == x0 else y0 + (y1 - y0) * (xq - x0) / (x1 - x0)
            best = y if best is None else max(best, y)
    return best


def paired(a, b, metric):
    d = a[metric] - b[metric]
    return float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune100")
    args = ap.parse_args()
    report = {}
    for model in ("adv", "clean"):
        variants = sorted({re.match(rf"p4_{model}_(.+)_{args.split}_s\d+\.json", os.path.basename(f)).group(1)
                           for f in glob.glob(os.path.join(RESULTS_DIR, f"p4_{model}_*_{args.split}_s*.json"))})
        R = {v: load(model, v, args.split) for v in variants}
        if not R:
            continue
        print(f"\n===== {model} model | {args.split} | mean +/- std over seeds")
        print(f"  {'variant':22s} {'clean+def':>16s} {'PGD+def':>16s} {'eps frac (clean/PGD)':>21s} "
              f"{'defended':>12s} {'s/img':>6s}  seeds")
        for v, r in R.items():
            st = r.get("stats", {})
            ef = "/".join(f"{st[m]['mean_eps_frac']:.2f}" for m in ("clean+def", "pgd+def") if m in st) or "1.00/1.00"
            df = "/".join(f"{st[m]['defended_fraction']:.2f}" for m in ("clean+def", "pgd+def") if m in st) or "1/1"
            print(f"  {v:22s} {ms(r['clean+def']):>16s} {ms(r['pgd+def']):>16s} {ef:>21s} {df:>12s} "
                  f"{r['sec'].get('pgd+def', 0):6.2f}  {r['seeds']} {r['git']}")
        rep = {"variants": {v: {"clean+def": r["clean+def"].tolist(), "pgd+def": r["pgd+def"].tolist(),
                                "stats": r.get("stats"), "git": r["git"]} for v, r in R.items()}}

        # global curve (eps 0 = undefended, from the global run that recorded clean/pgd)
        sweep = sorted((int(m.group(1)), v) for v in R for m in [re.fullmatch(r"global(\d+)", v)] if m)
        pts = []
        base = next((R[v] for _, v in sweep if "clean" in R[v]), None)
        if base is not None:
            pts.append({"eps": 0, "c": float(base["clean"].mean()), "p": float(base["pgd"].mean())})
        pts += [{"eps": e, "c": float(R[v]["clean+def"].mean()), "p": float(R[v]["pgd+def"].mean())} for e, v in sweep]
        print("  global curve (eps/255: clean+def, PGD+def): " +
              ", ".join(f"{p['eps']}: {p['c']:.2f}, {p['p']:.2f}" for p in pts))
        rep["global_curve"] = pts

        for ours in [v for v in ("perpixel", "gate_perpixel", "perimage", "perpixel_entropy") if v in R]:
            c, p = float(R[ours]["clean+def"].mean()), float(R[ours]["pgd+def"].mean())
            yp, xc = envelope(pts, c, "c", "p"), envelope(pts, p, "p", "c")
            gp = None if yp is None else p - yp
            gc = None if xc is None else c - xc
            beats = (gp is not None and gp > THR) or (gc is not None and gc > THR)
            print(f"  curve test {ours:18s}: PGD+def above curve at same clean+def: "
                  f"{'n/a' if gp is None else f'{gp:+.2f}'} | clean+def above curve at same PGD+def: "
                  f"{'n/a' if gc is None else f'{gc:+.2f}'} -> {'BEATS curve' if beats else 'does not beat curve'}")
            rep[f"curve_{ours}"] = {"pgd_gain": gp, "clean_gain": gc, "beats": bool(beats)}

        if all(v in R for v in ("perpixel", "perimage", "shuffled")):
            res = {}
            for ctrl in ("perimage", "shuffled"):
                dc, sc = paired(R["perpixel"], R[ctrl], "clean+def")
                dp, sp = paired(R["perpixel"], R[ctrl], "pgd+def")
                win = (dp > THR and dc >= -THR) or (dc > THR and dp >= -THR)
                res[ctrl] = {"d_clean": dc, "d_clean_std": sc, "d_pgd": dp, "d_pgd_std": sp, "win": bool(win)}
                print(f"  placement: perpixel - {ctrl:9s} clean+def {dc:+.2f} (+/-{sc:.2f})  "
                      f"PGD+def {dp:+.2f} (+/-{sp:.2f})  -> {'WIN' if win else 'no win'}")
            res["placement_matters"] = bool(all(res[c]["win"] for c in ("perimage", "shuffled")))
            print(f"  ==> placement matters: {res['placement_matters']}")
            rep["placement"] = res

        if model == "clean" and all(v in R for v in ("gate_perpixel", "perpixel")):
            dc, _ = paired(R["gate_perpixel"], R["perpixel"], "clean+def")
            dp, _ = paired(R["gate_perpixel"], R["perpixel"], "pgd+def")
            ok = dp >= -THR and dc > 0
            print(f"  gate test: gate+perpixel - perpixel: clean+def {dc:+.2f}, PGD+def {dp:+.2f} -> "
                  f"{'gate OK' if ok else 'gate not OK'}")
            rep["gate"] = {"d_clean": dc, "d_pgd": dp, "ok": bool(ok)}
        report[model] = rep

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"decision_{args.split}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  saved {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
