"""v2 confirmation: applies the pre-registered v2 rules (IMPLEMENTATION_PLAN.md, "v2 confirmation") on test100.

    python -m equi_lib.v2_analyze

References (plain prediction): P4c locked config (`p4c_<model>_ours_test100`) and its full-view row
(`p4c_<model>_ours_full_test100`). Candidates: avg_rs, -up, -up + avg_rs (`v2/v2_<model>_<name>_test100`).
BPDA-50 through the averaging: `p6/p6b_bpda50avgrs_ours_<model>_test100_n50_s0` vs plain `p6b_bpda50_ours_...`.
Writes results/improvements/v2/v2_decision_test100.json.
"""
import json
import os

import numpy as np

from . import RESULTS_DIR

THR = 1.5
MODES = ("clean", "pgd", "clean+def", "pgd+def")


def load(rel):
    rows = [json.load(open(os.path.join(RESULTS_DIR, rel.format(s=s)))) for s in (0, 1, 2)]
    out = {m: np.array([r["miou"][m] for r in rows]) for m in rows[0]["miou"]}
    out["sec_def"] = float(np.mean([r["sec_per_image"].get("pgd+def", 0.0) for r in rows]))
    out["sec_predict"] = float(np.mean([r["sec_per_image"].get("predict", 0.0) for r in rows]))
    return out


def bpda(name, model):
    return json.load(open(os.path.join(RESULTS_DIR, "p6", f"p6b_{name}_ours_{model}_test100_n50_s0.json")))["results"]


def d(a, b, m):
    return float((a[m] - b[m]).mean())


def main():
    report = {}
    for model in ("clean", "adv"):
        ref = load(f"p4c_{model}_ours_test100_s{{s}}.json")
        full = load(f"p4c_{model}_ours_full_test100_s{{s}}.json")
        C = {n: load(f"v2/v2_{model}_{n}_test100_s{{s}}.json") for n in ("avgrs", "noup", "noup_avgrs")}
        b_plain, b_avg = bpda("bpda50", model), bpda("bpda50avgrs", model)
        print(f"\n===== v2 {model} model | test100 | seeds 0-2 (paired Δ vs the P4c locked config / vs its full-view row)")
        rows = {"P4c locked (ref)": ref, "P4c full views": full, **C}
        for n, r in rows.items():
            print(f"  {n:17s} clean+def {r['clean+def'].mean():6.2f} ± {r['clean+def'].std(ddof=1):.2f}  "
                  f"PGD+def {r['pgd+def'].mean():6.2f} ± {r['pgd+def'].std(ddof=1):.2f}  "
                  f"Δref {d(r, ref, 'clean+def'):+.2f}/{d(r, ref, 'pgd+def'):+.2f}  "
                  f"Δfull {d(r, full, 'clean+def'):+.2f}/{d(r, full, 'pgd+def'):+.2f}  "
                  f"s/img def {r['sec_def']:.3f} + predict {r['sec_predict']:.3f}")
        a = C["avgrs"]
        r1 = d(a, ref, "pgd+def") > THR and d(a, ref, "clean+def") >= -THR
        r2 = b_avg["defended"] >= b_plain["defended"] - THR
        avg_ok = bool(r1 and r2)
        n = C["noup"]
        noup_ok = bool(all(abs(d(n, x, m)) <= THR for x in (ref, full) for m in ("clean+def", "pgd+def")))
        both_ok = avg_ok and noup_ok
        print(f"  BPDA-50 through avg_rs: defended {b_avg['defended']:.2f} vs plain BPDA-50 {b_plain['defended']:.2f} "
              f"(Δ {b_avg['defended'] - b_plain['defended']:+.2f})")
        print(f"  ==> avg_rs: rule 1 {r1}, rule 2 {r2} -> {'ADOPT' if avg_ok else 'not adopted'} | "
              f"-up: {'ADOPT' if noup_ok else 'not adopted (fast mode)'} | -up + avg_rs: {'ADOPT' if both_ok else 'no'}")
        report[model] = {
            "rows": {k: {m: v[m].tolist() for m in MODES if m in v} |{"sec_def": v["sec_def"], "sec_predict": v["sec_predict"]}
                     for k, v in rows.items()},
            "bpda50_plain": b_plain["defended"], "bpda50_through_avgrs": b_avg["defended"],
            "adopt_avgrs": avg_ok, "adopt_noup": noup_ok, "adopt_noup_avgrs": both_ok,
            "final": ("up1.5-up" if noup_ok else "up1.5") + (" + avg_rs" if avg_ok else "")}
    out = os.path.join(RESULTS_DIR, "v2", "v2_decision_test100.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
