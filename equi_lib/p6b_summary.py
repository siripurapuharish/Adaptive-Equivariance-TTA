"""Summarise the P6b attack runs (results/improvements/p6/p6b_*.json and the test100 gate-aware runs).

    python -m equi_lib.p6b_summary

Prints a markdown table (mean ± std over the seeds present) and writes results/improvements/p6/p6b_summary.json.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from . import RESULTS_DIR

P6 = os.path.join(RESULTS_DIR, "p6")
NAMES = {"segpgd50": "SegPGD-50 (transfer, 100 img)", "cospgd50": "CosPGD-50 (transfer, 100 img)",
         "noise": "random-noise control (100 img)", "bpda10": "BPDA-10 (50 img)", "bpda50": "BPDA-50 (50 img)",
         "bpda10eot4": "BPDA-10 + EOT-4 (50 img)", "bpda50fixed": "BPDA-50, fixed defense seed (50 img)",
         "segpgd50bpda": "SegPGD-50 via BPDA (50 img)",
         "ce50_n50": "PGD-50 (transfer, same 50 img)", "segpgd50_n50": "SegPGD-50 (transfer, same 50 img)",
         "cospgd50_n50": "CosPGD-50 (transfer, same 50 img)",
         "gateaware_lam1": "gate-aware BPDA-50, λ = 1 (50 img)", "gateaware_lam10": "gate-aware BPDA-50, λ = 10 (50 img)"}
TRANSFER = {"ce50", "segpgd50", "cospgd50"}
ADAPTIVE = {"bpda10", "bpda50", "bpda10eot4", "bpda50fixed", "segpgd50bpda", "gateaware_lam1", "gateaware_lam10"}


def fmt(v):
    return f"{np.mean(v):.2f}" + (f" ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else "")


def main():
    rows = defaultdict(lambda: defaultdict(list))   # (model, target, attack) -> metric -> values
    for f in glob.glob(os.path.join(P6, "p6b_*_test100*_s*.json")):
        m = re.match(r"p6b_(\w+?)_(ours|original)_(clean|adv)_test100(_n\d+)?_s", os.path.basename(f))
        if not m or m.group(1) == "equivcheck":
            continue
        tag = m.group(1)
        if m.group(4) and tag in TRANSFER:      # transfer attacks re-run on the 50-image subset
            tag += m.group(4)
        r = json.load(open(f))["results"]
        k = (m.group(3), m.group(2), tag)
        for key in ("undefended", "defended", "defended_fraction"):
            rows[k][key].append(r[key])
    for f in glob.glob(os.path.join(P6, "p6a_gateaware_lam*_test100_n50_s*.json")):
        tag = re.match(r"p6a_(gateaware_lam\d+)_", os.path.basename(f)).group(1)
        r = next(iter(json.load(open(f))["results"].values()))
        k = ("clean", "ours", tag)
        rows[k]["undefended"].append(r["undefended"])
        rows[k]["defended"].append(r["pipeline"])
        rows[k]["defended_fraction"].append(r["defended_fraction"])

    out = {}
    print("| Model | Defense | Attack | seeds | undefended | **defended** | gate defends |")
    print("|:--|:--|:--|:--:|:--:|:--:|:--:|")
    for model in ("clean", "adv"):
        for target in ("ours", "original"):
            worst = None
            for tag in NAMES:
                v = rows.get((model, target, tag))
                if not v:
                    continue
                print(f"| {model} | {target} | {NAMES[tag]} | {len(v['defended'])} | {fmt(v['undefended'])} | "
                      f"**{fmt(v['defended'])}** | {np.mean(v['defended_fraction']):.2f} |")
                out[f"{model}/{target}/{tag}"] = {k: list(map(float, x)) for k, x in v.items()}
                if tag in ADAPTIVE and (worst is None or np.mean(v["defended"]) < worst[1]):
                    worst = (tag, float(np.mean(v["defended"])))
            if worst:
                out[f"{model}/{target}/worst_adaptive"] = worst
                print(f"| {model} | {target} | **worst adaptive attack run** | | | **{worst[1]:.2f}** ({worst[0]}) | |")
    with open(os.path.join(P6, "p6b_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
