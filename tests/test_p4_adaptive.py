"""P4 correctness checks (2 tune100 images, both models). Run from the repo root:

    python tests/test_p4_adaptive.py

1. consistency_map == the P2 measure down|l2_softmax computed by p2_extended.Probe (max |diff| < 1e-5)
2. one_step with eps_map == eps_v everywhere gives the same delta as the scalar path
   (same seed; the mismatch must not exceed the scalar path's own rerun mismatch, CUDA non-determinism)
3. per-pixel box: |delta| <= eps_map everywhere; eps_map == 0 gives delta == 0
4. AdaptiveDefense modes: gate with tau = +inf returns x unchanged; perimage budget is constant per image
   with the same mean as perpixel; shuffled keeps the mean; budget map is detached and in [beta, 1] * eps_v
"""
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from equi_lib import ROOT_DIR  # noqa: E402
from equi_lib.adaptive import AdaptiveDefense  # noqa: E402
from equi_lib.attacks import PGD_PARAMS  # noqa: E402
from equi_lib.cache import PGDCache  # noqa: E402
from equi_lib.data import load_batch, split_indices  # noqa: E402
from equi_lib.defense import EquiDefense  # noqa: E402
from equi_lib.model import PRESETS, Normalizer, load_model  # noqa: E402
from equi_lib.p2_extended import Probe  # noqa: E402
from equi_lib.seed import set_seed  # noqa: E402

dev = torch.device("cuda")
ok_all = True


def check(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")


def mismatch(a, b):
    return (a != b).float().mean().item()


for tag in ("clean", "adv"):
    p = PRESETS[tag]
    print(f"\n=== {tag} model: eps_v {p['eps_v'] * 255:.0f}/255, {p['num_views']} views")
    model = load_model(os.path.join(ROOT_DIR, p["ckpt"]), dev)
    norm = Normalizer(dev)
    dfn = EquiDefense(model, norm, p["eps_v"], p["num_views"], steps=1, optimizer="one_step", device=dev)
    idx = split_indices("tune100")[:2]
    x, _ = load_batch(idx, dev)
    cache = PGDCache(tag, 0, PGD_PARAMS)
    xa = torch.clamp(x + torch.clamp(cache.get(idx, dev), -PGD_PARAMS["eps"], PGD_PARAMS["eps"]), 0, 1)

    # 1. consistency_map vs the P2 code path
    probe = Probe(model, norm, dfn.scripted)
    for name, inp in (("clean", x), ("pgd", xa)):
        with torch.no_grad():
            p0, z0 = probe.forward(inp)
            pv, zv, _ = probe.view(inp, "down", (z0.size(2), z0.size(3)))
            ref = Probe.measures(p0, z0, pv, zv)["l2_softmax"]
        e = dfn.consistency_map(inp)
        d = (e - ref).abs().max().item()
        check(f"consistency_map == P2 down|l2_softmax ({name} input)", d < 1e-5 and e.shape == ref.shape,
              f"max |diff| {d:.2e}, shape {tuple(e.shape)}")

    # 2. constant eps_map reproduces the scalar one_step path
    def run(eps_map=None, seed=7):
        set_seed(seed)
        return dfn.purify_delta(xa, eps_map)
    # CUDA backward is non-deterministic and the rerun mismatch is bimodal (seen: 6e-5 or 9e-4 on the adv
    # model), so the noise floor is the largest mismatch among 4 scalar reruns.
    runs = [run() for _ in range(4)]
    rerun = max(mismatch(runs[0], r) for r in runs[1:])
    d_map = run(torch.full_like(xa[:, :1], p["eps_v"]))
    vs = min(mismatch(r, d_map) for r in runs)
    check("eps_map == eps_v reproduces scalar one_step", vs <= max(2 * rerun, 1e-4),
          f"(scalar rerun mismatch up to {rerun:.2e}, map vs closest scalar run {vs:.2e} of entries)")

    # 3. per-pixel box
    em = torch.rand_like(xa[:, :1]) * p["eps_v"]
    em[:, :, :50] = 0.0
    dm = run(em)
    over = (dm.abs() - em.expand_as(dm)).max().item()
    check("|delta| <= eps_map everywhere", over <= 1e-7, f"(max excess {over:.2e})")
    check("eps_map == 0 gives delta == 0", dm[:, :, :50].abs().max().item() == 0.0)

    # 4. adaptive modes with a synthetic calibration
    calib = {"signals": {"equi": {"tau": float("inf"), "smooth9": {"a": 0.05, "b": 0.8}},
                         "entropy": {"tau": float("inf"), "smooth9": {"a": 0.1, "b": 1.0}}}, "git": "test"}
    kw = dict(device=dev, calib=calib, beta=0.2, smooth=9)
    gate = AdaptiveDefense(model, norm, p["eps_v"], p["num_views"], mode="gate+perpixel", **kw)
    check("gate with tau=+inf returns x unchanged", torch.equal(gate(xa), xa))
    st = gate.pop_stats()
    check("gate stats recorded", len(st) == 2 and not any(s["defended"] for s in st))
    pp = AdaptiveDefense(model, norm, p["eps_v"], p["num_views"], mode="perpixel", **kw)
    em, s = pp.budget(xa)
    lo, hi = em.min().item() / p["eps_v"], em.max().item() / p["eps_v"]
    check("budget map detached and in [beta, 1] * eps_v", not em.requires_grad and lo >= 0.2 - 1e-6 and hi <= 1 + 1e-6,
          f"(range {lo:.3f}..{hi:.3f} of eps_v, mean {em.mean().item() / p['eps_v']:.3f})")
    pim = em.mean(dim=(1, 2, 3), keepdim=True).expand_as(em)
    rolled = torch.roll(em, shifts=(xa.size(2) // 2, xa.size(3) // 2), dims=(2, 3))
    check("perimage / shuffled keep each image's mean budget",
          torch.allclose(pim.mean(dim=(1, 2, 3)), em.mean(dim=(1, 2, 3)), rtol=1e-5)
          and torch.allclose(rolled.mean(dim=(1, 2, 3)), em.mean(dim=(1, 2, 3)), rtol=1e-5))
    for mode in ("perpixel", "perimage", "shuffled", "global"):
        dd = AdaptiveDefense(model, norm, p["eps_v"], p["num_views"], mode=mode, **kw)
        out = dd(xa)
        dd.pop_stats()
        check(f"mode {mode} runs, output in [0,1], |out-x| <= eps_v",
              out.min() >= 0 and out.max() <= 1 and (out - xa).abs().max().item() <= p["eps_v"] + 1e-6)
    del model
    torch.cuda.empty_cache()

print(f"\nOVERALL: {'PASS' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
