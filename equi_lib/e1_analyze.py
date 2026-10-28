"""E1 analysis: applies the two pre-registered rules (IMPLEMENTATION_PLAN.md, E1) to the tune100 runs.

    python -m equi_lib.e1_analyze

Rule 1 (fixed subset): a subset replaces base if it beats base by > 1.5 on PGD+def or clean+def while losing
<= 1.5 on the other, or is within +/-1.5 on both at <= 0.75x base's PGD+def sec/img. Paired over seeds 0-2.
Rule 2 (per-image oracle): per image, pick the subset with the highest per-image mIoU (seed 0 runs, same seed-0
attacks); pool the chosen confusion matrices. Noise oracle: the same best-of-K pick over K reruns of base on the
same attacks (defense seeds 0..K-1). Go to E1b only if oracle - noise oracle > 1.5 on PGD+def.
Writes results/improvements/e1/e1_decision_tune100.json.
"""
import json
import os

import numpy as np

from . import RESULTS_DIR

THR = 1.5
E1 = os.path.join(RESULTS_DIR, "e1")
SUBSETS = {"clean": ["base", "norot", "norotaddflip", "addflip", "nodown", "noup"],
           "adv": ["base", "norot", "nojit", "noflip", "nodown", "noup"]}
LABEL = {"base": "base (up1.5)", "norot": "-rot", "norotaddflip": "-rot +flip", "addflip": "+flip",
         "nodown": "-down", "noup": "-up", "nojit": "-jit", "noflip": "-flip"}


def per_class_iou(h):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(h) / (h.sum(0) + h.sum(1) - np.diag(h))


def miou(h):
    return float(np.nanmean(per_class_iou(h)) * 100)


def load_json(model, name, seed):
    with open(os.path.join(E1, f"e1_{model}_{name}_tune100_s{seed}.json")) as f:
        return json.load(f)


def load_pi(model, name, seed, mode):
    z = np.load(os.path.join(E1, f"e1_{model}_{name}_tune100_s{seed}_perimage.npz"))
    return z["indices"], z[mode].astype(np.int64)


def oracle(runs):
    """runs: list of [N,19,19] per-image confusions on the same images. Returns (oracle mIoU, mean single mIoU,
    fraction of images where each run is picked)."""
    per = np.array([[miou(h) for h in r] for r in runs])           # [K, N] per-image mIoU (nan-mean over classes)
    pick = np.nanargmax(np.nan_to_num(per, nan=-1.0), axis=0)
    chosen = np.stack([runs[k][i] for i, k in enumerate(pick)])
    return miou(chosen.sum(0)), float(np.mean([miou(r.sum(0)) for r in runs])), \
        np.bincount(pick, minlength=len(runs)) / len(pick)


def main():
    report = {}
    for model in ("clean", "adv"):
        print(f"\n===== E1 {model} model | tune100 | seeds 0-2 | ours config with each view subset")
        R = {}
        for name in SUBSETS[model]:
            rows = [load_json(model, name, s) for s in (0, 1, 2)]
            R[name] = {m: np.array([r["miou"][m] for r in rows]) for m in ("clean+def", "pgd+def")}
            R[name]["sec"] = float(np.mean([r["sec_per_image"]["pgd+def"] for r in rows]))
            R[name]["views"] = rows[0]["config"]["def_opts"]
        b = R["base"]
        print(f"  {'subset':12s} {'clean+def':>15s} {'PGD+def':>15s} {'d clean':>8s} {'d PGD':>8s} {'s/img':>6s} "
              f"{'speed':>6s}  rule 1")
        rep = {"subsets": {}}
        adopt = []
        for name, r in R.items():
            dc = float((r["clean+def"] - b["clean+def"]).mean())
            dp = float((r["pgd+def"] - b["pgd+def"]).mean())
            ratio = r["sec"] / b["sec"]
            better = (dp > THR and dc >= -THR) or (dc > THR and dp >= -THR)
            faster = abs(dc) <= THR and abs(dp) <= THR and ratio <= 0.75
            ok = name != "base" and (better or faster)
            if ok:
                adopt.append((dp, name))
            print(f"  {LABEL[name]:12s} {r['clean+def'].mean():7.2f} +/- {r['clean+def'].std(ddof=1):.2f} "
                  f"{r['pgd+def'].mean():7.2f} +/- {r['pgd+def'].std(ddof=1):.2f} {dc:+8.2f} {dp:+8.2f} "
                  f"{r['sec']:6.3f} {ratio:6.2f}x  {'ADOPT-able' if ok else ('-' if name == 'base' else 'no')}")
            rep["subsets"][name] = {"clean+def": r["clean+def"].tolist(), "pgd+def": r["pgd+def"].tolist(),
                                    "d_clean": dc, "d_pgd": dp, "sec": r["sec"], "speed_ratio": ratio,
                                    "qualifies": bool(ok), "def_opts": r["views"]}
        rep["rule1_choice"] = max(adopt)[1] if adopt else "base"
        print(f"  ==> rule 1: {'adopt ' + LABEL[rep['rule1_choice']] if adopt else 'keep base'}")

        modes = ["pgd_def"] + (["clean_def"] if model == "adv" else [])
        for mode in modes:
            sub, idx0 = [], None
            for name in SUBSETS[model]:
                idx, h = load_pi(model, name, 0, mode)
                idx0 = idx if idx0 is None else idx0
                assert np.array_equal(idx, idx0)
                sub.append(h)
            noise = [load_pi(model, "base", 0, mode)[1]] + [load_pi(model, "base_noise", s, mode)[1] for s in (1, 2, 3, 4, 5)]
            o_sub, m_sub, frac = oracle(sub)
            o_noise, m_noise, _ = oracle(noise)
            gain = o_sub - o_noise
            go = gain > THR
            print(f"  oracle {mode:9s}: subsets best-of-{len(sub)} {o_sub:.2f} (mean single run {m_sub:.2f}) | "
                  f"noise best-of-{len(noise)} {o_noise:.2f} (mean single {m_noise:.2f}) | gain over noise "
                  f"{gain:+.2f}" + (f" -> {'GO to E1b' if go else 'no-go'}" if mode == "pgd_def" else " (info)"))
            print("    picked: " + ", ".join(f"{LABEL[n]} {100 * f:.0f}%" for n, f in zip(SUBSETS[model], frac)))
            rep[f"oracle_{mode}"] = {"subset_oracle": o_sub, "subset_mean": m_sub, "noise_oracle": o_noise,
                                     "noise_mean": m_noise, "gain": gain, "go": bool(go) if mode == "pgd_def" else None,
                                     "picked": dict(zip(SUBSETS[model], frac.tolist()))}
        report[model] = rep
    out = os.path.join(E1, "e1_decision_tune100.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
