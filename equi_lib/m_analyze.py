"""Apply the pre-registered M1 / M2b / M3 rules (IMPLEMENTATION_PLAN.md, "Mid-review improvements")
to the tune100 runs in results/improvements/m_*_tune100_s{0,1,2}.json.

    python -m equi_lib.m_analyze

Writes results/improvements/mid/m_decision_tune100.json.
"""
import glob
import json
import os

import numpy as np

from . import RESULTS_DIR

THR = 1.5
SEEDS = (0, 1, 2)
OUT = os.path.join(RESULTS_DIR, "mid", "m_decision_tune100.json")


def load(tag):
    files = [os.path.join(RESULTS_DIR, f"{tag}_tune100_s{s}.json") for s in SEEDS]
    if not all(os.path.exists(f) for f in files):
        return None
    runs = [json.load(open(f)) for f in files]
    r = {}
    for m in ("clean+def", "pgd+def"):
        v = np.array([x["miou"][m] for x in runs])
        r[m] = (float(v.mean()), float(v.std(ddof=1)), v.tolist())
    r["sec_pgd+def"] = float(np.mean([x["sec_per_image"]["pgd+def"] for x in runs]))
    return r


def diff(a, b, m):
    """Paired (same seeds) mean difference a - b."""
    return float(np.mean(np.array(a[m][2]) - np.array(b[m][2])))


def row(name, r, base=None):
    s = f"  {name:34s} clean+def {r['clean+def'][0]:6.2f} ± {r['clean+def'][1]:.2f} | " \
        f"PGD+def {r['pgd+def'][0]:6.2f} ± {r['pgd+def'][1]:.2f} | {r['sec_pgd+def']:.2f} s/img"
    if base is not None:
        s += f" | Δ {diff(r, base, 'clean+def'):+.2f} / {diff(r, base, 'pgd+def'):+.2f}"
    print(s)


def main():
    out = {"threshold": THR, "models": {}}
    verdict = {}
    for model in ("clean", "adv"):
        base = load(f"m_{model}_base")
        if base is None:
            print(f"{model}: base run missing")
            continue
        print(f"\n== {model} model (tune100, 3 seeds; Δ = paired difference vs base: clean+def / PGD+def)")
        row("base", base)
        res = {"base": base}
        for tag in ("noup", "up15", "lw_equi", "lw_conf", "lw_invconf", "entgate_perimage", "entgate", "equigate"):
            r = load(f"m_{model}_{tag}")
            if r is not None:
                res[tag] = r
                row(tag, r, base if not tag.endswith("gate") else None)
        out["models"][model] = res

    # M1: fastest view set within THR of full on both metrics, on both models
    m1 = {}
    for vs in ("noup", "up15"):
        ok = []
        for model, res in out["models"].items():
            if vs in res:
                ok.append(all(abs(diff(res[vs], res["base"], m)) <= THR or diff(res[vs], res["base"], m) > 0
                              for m in ("clean+def", "pgd+def")))
        m1[vs] = bool(ok) and all(ok) and len(ok) == 2
    passing = [vs for vs in ("noup", "up15") if m1[vs]]   # noup is the faster one
    verdict["M1"] = {"per_variant_pass": m1, "adopt": passing[0] if passing else "full"}

    # M2b: equivariance gate vs entropy gate (clean model)
    c = out["models"].get("clean", {})
    m2b = {}
    for ours, theirs in (("base", "entgate_perimage"), ("equigate", "entgate")):
        if ours in c and theirs in c:
            d = {m: diff(c[ours], c[theirs], m) for m in ("clean+def", "pgd+def")}
            if any(v > THR for v in d.values()):
                v = "equivariance gate wins: wording stands"
            elif all(v < -THR for v in d.values()):
                v = "entropy gate wins both: rename the contribution"
            else:
                v = "tie: the wording rests on P6a"
            m2b[f"{ours} vs {theirs}"] = {"diff_equi_minus_entropy": d, "verdict": v}
    verdict["M2b"] = m2b

    # M3: equi weighting beats uniform by > THR on PGD+def, loses <= THR on clean+def, on >= 1 model
    m3 = {}
    for model, res in out["models"].items():
        for w in ("lw_equi", "lw_conf", "lw_invconf"):
            if w in res:
                d = {m: diff(res[w], res["base"], m) for m in ("clean+def", "pgd+def")}
                m3[f"{model} {w}"] = {"diff_vs_uniform": d,
                                      "pass": d["pgd+def"] > THR and d["clean+def"] >= -THR}
    verdict["M3"] = {"runs": m3, "adopt_equi": any(v["pass"] for k, v in m3.items() if k.endswith("lw_equi"))}

    out["verdict"] = verdict
    print("\n== Verdicts (pre-registered rules)")
    print(json.dumps(verdict, indent=1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {os.path.relpath(OUT, os.path.dirname(RESULTS_DIR))}")


if __name__ == "__main__":
    main()
