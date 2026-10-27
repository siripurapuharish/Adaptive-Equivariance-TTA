"""Slide figures for the chosen P2 measure `down|l2_softmax` (forward passes only; PIL, no matplotlib).

    python -m equi_lib.p2_figures --model adv

Uses exactly the P2 code path (p2_extended.Probe, view "down", measure "l2_softmax"),
tune100, seed 0, cached PGD-50. Writes to results/improvements/p2/:
  fig_<model>_examples.png : 5 rows of  image | error map (clean input) | error map (PGD input) |
                             pixels whose prediction PGD changed (white). Same colour scale everywhere
                             (black = 0 ... white = 99th percentile of attacked-pixel error).
  fig_<model>_hist.png     : histograms, clean (blue) vs PGD (red): per-pixel error (left), per-image mean (right)
  fig_<model>_hist.json    : the histogram bins and counts, to redraw the charts elsewhere
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import RESULTS_DIR, ROOT_DIR
from .attacks import PGD_PARAMS, pgd
from .cache import PGDCache
from .data import load_batch, split_indices
from .defense import EquiDefense
from .model import PRESETS, Normalizer, load_model
from .p2_extended import Probe, resize
from .runner import git_hash
from .seed import set_seed

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(RESULTS_DIR, "p2")
BLUE, RED = (40, 110, 200), (215, 50, 40)


def heat(e, vmax):
    """[H,W] float -> [H,W,3] uint8, black -> red -> yellow -> white."""
    t = np.clip(e / vmax, 0, 1)
    r = np.clip(3 * t, 0, 1)
    g = np.clip(3 * t - 1, 0, 1)
    b = np.clip(3 * t - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def draw_hist(draw, box, edges, series, title):
    """series: list of (counts normalised to max 1, colour). Draws outlined step bars in box."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(0, 0, 0))
    draw.text((x0 + 4, y0 - 14), title, fill=(0, 0, 0))
    n = len(edges) - 1
    w = (x1 - x0) / n
    for counts, col in series:
        pts = []
        for i, c in enumerate(counts):
            top = y1 - c * (y1 - y0 - 6)
            pts += [(x0 + i * w, top), (x0 + (i + 1) * w, top)]
        draw.line([(x0, y1)] + pts + [(x1, y1)], fill=col, width=2)
    draw.text((x0, y1 + 3), f"{edges[0]:.2f}", fill=(0, 0, 0))
    draw.text((x1 - 30, y1 + 3), f"{edges[-1]:.2f}", fill=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--images", type=int, default=100, help="images used for the histograms")
    args = ap.parse_args()

    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    probe = Probe(model, norm, EquiDefense(model, norm, preset["eps_v"], preset["num_views"], device=device).scripted)
    cache = PGDCache(args.model, 0, PGD_PARAMS)
    indices = split_indices("tune100")[:args.images]
    set_seed(0)

    def emap(x, hw_full):
        with torch.no_grad():
            p0, z0 = probe.forward(x)
            pv, zv, _ = probe.view(x, "down", (z0.size(2), z0.size(3)))
            e = Probe.measures(p0, z0, pv, zv)["l2_softmax"]
            pred = resize(z0, hw_full).argmax(1)
            return resize(e.unsqueeze(1), hw_full).squeeze(1), pred

    rng = np.random.default_rng(0)
    pix = {"clean": [], "pgd": []}
    img = {"clean": [], "pgd": []}
    examples = []
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        hw = tuple(y.shape[-2:])
        if cache.has(idx):
            d = torch.clamp(cache.get(idx, device), -PGD_PARAMS["eps"], PGD_PARAMS["eps"])
            d = torch.clamp(d, -x, 1.0 - x)
        else:
            d = pgd(model, norm, x, y, **PGD_PARAMS)
        xa = torch.clamp(x + d, 0.0, 1.0)
        e_c, p_c = emap(x, hw)
        e_a, p_a = emap(xa, hw)
        for i in range(len(idx)):
            for k, e in (("clean", e_c[i]), ("pgd", e_a[i])):
                flat = e.flatten().cpu().numpy()
                pix[k].append(flat[rng.choice(flat.size, 2000, replace=False)])
                img[k].append(float(e.mean()))
            if len(examples) < 5 and b % 4 == 0 and i == 0:   # split positions 0, 4, 8, 12, 16
                examples.append((idx[i], (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                                 e_c[i].cpu().numpy(), e_a[i].cpu().numpy(), (p_c[i] != p_a[i]).cpu().numpy()))
        print(f"  [{min(b + 2, len(indices))}/{len(indices)}]", flush=True)
    vmax = float(np.percentile(np.concatenate(pix["pgd"]), 99))

    # example grid, half resolution (340x170 per panel)
    rows = []
    for _, im, ec, ea, ch in examples:
        ch3 = np.repeat((ch * 255).astype(np.uint8)[..., None], 3, -1)
        rows.append(np.concatenate([im, heat(ec, vmax), heat(ea, vmax), ch3], 1))
    grid = Image.fromarray(np.concatenate(rows, 0))
    grid = grid.resize((grid.width // 2, grid.height // 2), Image.BILINEAR)
    canvas = Image.new("RGB", (grid.width, grid.height + 20), (255, 255, 255))
    canvas.paste(grid, (0, 20))
    d = ImageDraw.Draw(canvas)
    for c, t in enumerate(("input", "error, clean input", "error, PGD-50 input", "pixels PGD changed")):
        d.text((c * grid.width // 4 + 6, 4), t, fill=(0, 0, 0))
    out1 = os.path.join(OUT_DIR, f"fig_{args.model}_examples.png")
    canvas.save(out1)

    # histograms
    allpix = np.concatenate(pix["clean"] + pix["pgd"])
    e_pix = np.linspace(0, float(np.percentile(allpix, 99.5)), 61)
    e_img = np.linspace(min(img["clean"] + img["pgd"]), max(img["clean"] + img["pgd"]), 26)
    h = {k: {"pixel": np.histogram(np.concatenate(pix[k]), e_pix)[0].tolist(),
             "image": np.histogram(img[k], e_img)[0].tolist()} for k in pix}
    W, H = 900, 300
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((10, 6), f"{args.model} model, tune100 ({len(indices)} images): clean (blue) vs PGD-50 (red), "
                    f"down-scale softmax-L2 error", fill=(0, 0, 0))
    for j, (key, edges, title) in enumerate((("pixel", e_pix, "per-pixel error (density)"),
                                              ("image", e_img, "per-image mean error (count)"))):
        if key == "pixel":
            series = [(np.array(h[k][key]) / sum(h[k][key]), col) for k, col in (("clean", BLUE), ("pgd", RED))]
            top = max(s.max() for s, _ in series)
            series = [(s / top, col) for s, col in series]
        else:
            top = max(max(h["clean"][key]), max(h["pgd"][key]))
            series = [(np.array(h[k][key]) / top, col) for k, col in (("clean", BLUE), ("pgd", RED))]
        draw_hist(d, (30 + j * 440, 50, 410 + j * 440, 270), edges, series, title)
    out2 = os.path.join(OUT_DIR, f"fig_{args.model}_hist.png")
    im.save(out2)
    with open(os.path.join(OUT_DIR, f"fig_{args.model}_hist.json"), "w") as f:
        json.dump({"pixel_edges": e_pix.tolist(), "image_edges": e_img.tolist(), "counts": h,
                   "pixels_per_image": 2000, "num_images": len(indices), "example_indices": [e[0] for e in examples],
                   "colour_vmax": vmax, "measure": "down|l2_softmax", "git": git_hash()}, f, indent=1)
    print(f"saved {os.path.relpath(out1, ROOT_DIR)}, {os.path.relpath(out2, ROOT_DIR)}")


if __name__ == "__main__":
    main()
