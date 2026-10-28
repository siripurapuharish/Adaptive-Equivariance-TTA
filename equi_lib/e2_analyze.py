"""E2 / P5 analysis: prediction averaging vs the plain prediction (IMPLEMENTATION_PLAN.md, P5 rule).

    python -m equi_lib.e2_analyze

Rule (pre-registered): adopt the best variant if it beats `single` by > 1.5 on PGD+def or clean+def while losing
<= 1.5 on the other (paired over seeds 0-2, tune100). The undefended rows (clean, pgd) show plain test-time
augmentation without the defense.
Writes results/improvements/e2/e2_decision_tune100.json.
"""
import json
import os

import numpy as np

from . import RESULTS_DIR

THR = 1.5
E2 = os.path.join(RESULTS_DIR, "e2")
MODES = ("clean", "pgd", "clean+def", "pgd+def")


def load(model, variant):
    rows = [json.load(open(os.path.join(E2, f"e2_{model}_{variant}_tune100_s{s}.json"))) for s in (0, 1, 2)]
    out = {m: np.array([r["miou"][m] for r in rows]) for m in MODES}
    out["sec_predict"] = float(np.mean([r["sec_per_image"].get("predict", 0.0) for r in rows]))
    return out


def main():
    report = {}
    for model in ("clean", "adv"):
        R = {v: load(model, v) for v in ("single", "avg_flip", "avg_rs")}
        s = R["single"]
        print(f"\n===== E2 {model} model | tune100 | seeds 0-2 (Δ = paired difference vs single)")
        print(f"  {'variant':9s} " + " ".join(f"{m:>16s}" for m in MODES) + "  predict s/img  rule")
        rep, best = {}, None
        for v, r in R.items():
            d = {m: float((r[m] - s[m]).mean()) for m in MODES}
            ok = v != "single" and ((d["pgd+def"] > THR and d["clean+def"] >= -THR) or
                                    (d["clean+def"] > THR and d["pgd+def"] >= -THR))
            if ok and (best is None or d["pgd+def"] > best[0]):
                best = (d["pgd+def"], v)
            cells = " ".join(f"{r[m].mean():6.2f} ({d[m]:+5.2f})" if v != "single" else f"{r[m].mean():6.2f} ± {r[m].std(ddof=1):.2f}"
                             for m in MODES)
            print(f"  {v:9s} {cells}  {r['sec_predict']:.3f}  {'-' if v == 'single' else ('ADOPT-able' if ok else 'no')}")
            rep[v] = {m: r[m].tolist() for m in MODES} | {"delta": d, "qualifies": bool(ok), "sec_predict": r["sec_predict"]}
        rep["choice"] = best[1] if best else "single"
        print(f"  ==> rule: {'adopt ' + best[1] if best else 'keep single'}")
        report[model] = rep
    out = os.path.join(E2, "e2_decision_tune100.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
