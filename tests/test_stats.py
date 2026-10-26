"""Checks equi_lib.stats.auroc against brute-force pair counting (ties count 1/2).
Run from the repo root:  python tests/test_stats.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equi_lib.stats import auroc  # noqa: E402


def brute(s, y):
    pos, neg = s[y], s[~y]
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


rng = np.random.default_rng(0)
for trial in range(200):
    n = int(rng.integers(2, 300))
    s = rng.integers(0, 6, n).astype(float) if trial % 2 else rng.normal(size=n)  # half the trials have many ties
    y = rng.random(n) < rng.uniform(0.05, 0.95)
    if y.all() or not y.any():
        continue
    assert abs(auroc(s, y) - brute(s, y)) < 1e-12, (trial, auroc(s, y), brute(s, y))
assert auroc([0.1, 0.9], [False, True]) == 1.0 and auroc([0.9, 0.1], [False, True]) == 0.0
assert auroc([0.5, 0.5], [False, True]) == 0.5
assert np.isnan(auroc([1.0, 2.0], [True, True]))
print("PASS auroc matches brute force on 200 random cases (with and without ties)")
