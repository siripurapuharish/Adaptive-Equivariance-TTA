"""P3 day-5 check: is the exact gradient through the unrolled defense the same as BPDA's?

Each defense step is delta <- clamp(delta - s * sign(g), ...); sign() has zero derivative almost
everywhere, so d(defended)/d(x) should be the identity except where the [0, 1] box clamp is
active. If so, the exact "full-gradient" attack planned for P6 equals BPDA, and P6 needs a
smoothed surrogate instead.

Computes, on one tune100 image under PGD-50 (clean model, 7 views, eps_v 6/255, original
optimizer with its noise, T steps):
  g_exact : d task_loss(model(x + delta_T(x))) / d x, autograd through all T steps with
            create_graph=True (second-order terms through the defense gradient included)
  g_bpda  : d task_loss / d (defended image), treating the defense as identity
Reports cosine(g_exact, g_bpda) over all pixels and over pixels where the box clamp is inactive.
Note: create_graph=True fails because grid_sample (rotation views) has no double-backward in PyTorch;
that is reported, and the first-order result is exact because sign() has zero derivative.

Run from the repo root:  python tests/check_unrolled_gradient.py
"""
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from equi_lib import ROOT_DIR  # noqa: E402
from equi_lib.attacks import PGD_PARAMS, pgd  # noqa: E402
from equi_lib.data import load_batch, split_indices  # noqa: E402
from equi_lib.defense import EquiDefense  # noqa: E402
from equi_lib.metrics import align_output  # noqa: E402
from equi_lib.model import PRESETS, Normalizer, load_model  # noqa: E402
from equi_lib.seed import set_seed  # noqa: E402

T = 3
dev = torch.device("cuda")
p = PRESETS["clean"]
model = load_model(os.path.join(ROOT_DIR, p["ckpt"]), dev)
norm = Normalizer(dev)
dfn = EquiDefense(model, norm, p["eps_v"], p["num_views"], steps=T, device=dev)
x, y = load_batch(split_indices("tune100")[:1], dev)
set_seed(0)
x_adv = torch.clamp(x + pgd(model, norm, x, y, **PGD_PARAMS), 0, 1).detach()


def task_loss(img):
    return F.nll_loss(align_output(model(norm(img))[0], y.shape[-2:]), y, ignore_index=255)


def unrolled(xin, create_graph):
    """Original-optimizer defense, differentiable w.r.t. xin (same math as EquiDefense.purify_delta)."""
    gm = dfn._global_mask(xin.detach())
    eps = dfn.eps_v
    delta = torch.empty_like(xin).uniform_(-eps, eps)
    delta = torch.max(torch.min(delta, 1 - xin), 0 - xin)
    for i in range(T):
        cost = dfn.loss(xin + delta, gm)
        g = torch.autograd.grad(cost, delta, create_graph=create_graph, retain_graph=True)[0] \
            if delta.requires_grad else torch.zeros_like(delta)
        g = g / (torch.sum(g ** 2) ** 0.5 + 1e-10) + torch.randn_like(g) * (T - 1 - i) / T
        delta = torch.clamp(delta - dfn.step_size * torch.sign(g), -eps, eps)
        delta = torch.max(torch.min(delta, 1 - xin), 0 - xin)
    return xin + delta


results = {}
for cg in (True, False):
    set_seed(1)
    xin = x_adv.clone().requires_grad_(True)
    x_def = unrolled(xin, create_graph=cg)
    try:
        g_exact = torch.autograd.grad(task_loss(x_def), xin)[0]
    except RuntimeError as e:
        # rotation views use grid_sample, which has no double-backward in PyTorch
        print(f"create_graph={cg}: second-order gradient not computable ({str(e).splitlines()[0]})")
        continue
    x_def_leaf = x_def.detach().requires_grad_(True)
    g_bpda = torch.autograd.grad(task_loss(x_def_leaf), x_def_leaf)[0]
    inside = ((x_def.detach() > 0) & (x_def.detach() < 1))   # box clamp inactive
    cos_all = F.cosine_similarity(g_exact.flatten(), g_bpda.flatten(), dim=0).item()
    cos_in = F.cosine_similarity(g_exact[inside], g_bpda[inside], dim=0).item()
    rel = ((g_exact - g_bpda)[inside].norm() / g_bpda[inside].norm()).item()
    results[cg] = (cos_all, cos_in, rel, inside.float().mean().item(), g_exact.clone())
    print(f"create_graph={cg}: cosine all pixels {cos_all:.6f} | clamp-inactive pixels {cos_in:.6f} "
          f"(rel. diff {rel:.2e}, {100 * inside.float().mean().item():.2f}% of entries) | "
          f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
if len(results) == 2:
    same = torch.allclose(results[True][4], results[False][4], rtol=1e-4, atol=1e-10)
    print(f"g_exact identical with and without second-order terms: {same}")
else:
    print("Second-order terms only enter through sign(g), whose derivative is 0 almost everywhere, "
          "so the create_graph=False gradient is the exact gradient.")
