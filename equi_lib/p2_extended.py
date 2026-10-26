"""Extended P2 table (IMPLEMENTATION_PLAN.md, P2): which (view, error measure) pair separates
attacked from clean pixels, for each attack? Forward passes only, apart from attack generation.

    python -m equi_lib.p2_extended --model adv --split tune100 --seed 0

Views (deterministic, each mapped back onto the base feature grid 43x85):
  flip (horizontal), down (0.5x), up (2x), rot10 (+10 deg, masked corners),
  shift (crop 16 px top / 24 px left = exactly 2 / 3 feature cells), jitter (ColorJitter + RandomGrayscale)
Measures per pixel (base output vs mapped-back view output):
  cos_logit (1 - cosine of 19 logits; the P1 error_map measure), l2_softmax, sym_kl,
  label_flip (argmax differs), cos_penult (1 - cosine of the 512-ch features before the final 1x1 conv)
Also: the defense's own view set with cos_logit (= EquiDefense.error_map, sanity check against P2),
and per-pixel softmax entropy (baseline, both directions).
Attacks: PGD-50 (eps 4/255, cached), SegPGD-50 (eps 4/255), low-eps PGD-50 (eps 2/255, step 0.5/255).
Pixel AUROC target: "prediction changed by the attack" on attacked images, on a fixed random
sample of 10,000 pixels per image (same pixels for every score; pixels outside a view's valid
area, i.e. rot10 corners and the shift border, are excluded for that view). Image AUROC: mean score per image,
clean vs attacked.
"""
import argparse
import json
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import RESULTS_DIR, ROOT_DIR
from .attacks import PGD_PARAMS, pgd, segpgd
from .cache import PGDCache
from .data import load_batch, split_indices
from .defense import EquiDefense
from .metrics import ConfusionMeter, align_output
from .model import PRESETS, Normalizer, load_model
from .runner import git_hash
from .seed import set_seed
from .stats import auroc
from learning.transform_geo import rot_img

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(RESULTS_DIR, "p2")
VIEWS = ("flip", "down", "up", "rot10", "shift", "jitter")
MEASURES = ("cos_logit", "l2_softmax", "sym_kl", "label_flip", "cos_penult")
LOW_EPS = {"eps": 2.0 / 255.0, "step": 0.5 / 255.0, "steps": 50}
SHIFT_PX = (16, 24)          # multiples of the 8x stride: exact (2, 3) feature-cell shift
ROT_THETA = 10 / 180 * np.pi
PIXELS_PER_IMAGE = 10000


def resize(t, size):
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False)


class Probe:
    def __init__(self, model, normalize, scripted):
        self.model, self.normalize, self.scripted = model, normalize, scripted

    def forward(self, x):
        pen = self.model.base(self.normalize(x))     # [B,512,h,w]
        return pen, self.model.seg(pen)              # logits [B,19,h,w] (= DRNSeg output [1])

    def view(self, x, name, hw):
        """Returns (penultimate, logits, valid[B,h,w]) of the view, mapped back to the base grid hw."""
        h, w = hw
        if name == "flip":
            p, z = self.forward(torch.flip(x, dims=[3]))
            return torch.flip(p, dims=[3]), torch.flip(z, dims=[3]), None
        if name in ("down", "up"):
            s = 0.5 if name == "down" else 2.0
            p, z = self.forward(resize(x, (int(x.size(2) * s), int(x.size(3) * s))))
            return resize(p, hw), resize(z, hw), None
        if name == "jitter":
            p, z = self.forward(self.scripted(x))
            return p, z, None
        if name == "rot10":
            p, z = self.forward(rot_img(x, ROT_THETA, x.dtype))
            p, z = rot_img(p, -ROT_THETA, p.dtype), rot_img(z, -ROT_THETA, z.dtype)
            ones = torch.ones(x.size(0), 1, h, w, device=x.device)
            m = rot_img(rot_img(ones, ROT_THETA, ones.dtype), -ROT_THETA, ones.dtype)[:, 0] > 0.5
            return p, z, m
        if name == "shift":
            dy, dx = SHIFT_PX
            p, z = self.forward(x[:, :, dy:, dx:])
            cy, cx = dy // 8, dx // 8
            P = torch.zeros(p.size(0), p.size(1), h, w, device=x.device)
            Z = torch.zeros(z.size(0), z.size(1), h, w, device=x.device)
            m = torch.zeros(x.size(0), h, w, dtype=torch.bool, device=x.device)
            hh, ww = min(p.size(2), h - cy), min(p.size(3), w - cx)
            P[:, :, cy:cy + hh, cx:cx + ww] = p[:, :, :hh, :ww]
            Z[:, :, cy:cy + hh, cx:cx + ww] = z[:, :, :hh, :ww]
            m[:, cy:cy + hh, cx:cx + ww] = True
            return P, Z, m
        raise ValueError(name)

    @staticmethod
    def measures(p0, z0, pv, zv):
        q0, qv = F.softmax(z0, 1), F.softmax(zv, 1)
        lq0, lqv = F.log_softmax(z0, 1), F.log_softmax(zv, 1)
        return {
            "cos_logit": 1 - F.cosine_similarity(z0, zv, dim=1, eps=1e-8),
            "l2_softmax": (q0 - qv).norm(dim=1),
            "sym_kl": 0.5 * ((q0 * (lq0 - lqv)).sum(1) + (qv * (lqv - lq0)).sum(1)),
            "label_flip": (z0.argmax(1) != zv.argmax(1)).float(),
            "cos_penult": 1 - F.cosine_similarity(p0, pv, dim=1, eps=1e-8),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--split", default="tune100", choices=["tune100", "test100"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    defense = EquiDefense(model, norm, preset["eps_v"], preset["num_views"], device=device)
    probe = Probe(model, norm, defense.scripted)
    cache = PGDCache(args.model, args.seed, PGD_PARAMS)
    indices = split_indices(args.split)[:args.limit]
    set_seed(args.seed)
    attacks = ("pgd", "segpgd", "pgd_loweps")

    # score name -> attack -> list of arrays ; "clean" stores clean-image values
    pix = defaultdict(lambda: defaultdict(list))
    pix_valid = defaultdict(lambda: defaultdict(list))  # rot10 / shift have invalid border pixels
    img = defaultdict(lambda: defaultdict(list))
    labels = defaultdict(list)
    meters = {a: ConfusionMeter() for a in ("clean", *attacks)}
    t0 = time.time()

    def score_maps(x, hw_full):
        """All (view, measure) score maps + defense-set error map + entropy, upsampled to full res.
        Returns dict name -> [B,H,W] tensor and a validity dict name -> [B,H,W] bool (for image means)."""
        with torch.no_grad():
            p0, z0 = probe.forward(x)
            hw = (z0.size(2), z0.size(3))
            out, valid = {}, {}
            for v in VIEWS:
                pv, zv, m = probe.view(x, v, hw)
                for meas, s in Probe.measures(p0, z0, pv, zv).items():
                    s = torch.where(m, s, torch.zeros_like(s)) if m is not None else s
                    out[f"{v}|{meas}"] = resize(s.unsqueeze(1), hw_full).squeeze(1)
                    mm = m if m is not None else torch.ones_like(s, dtype=torch.bool)
                    valid[f"{v}|{meas}"] = resize(mm.float().unsqueeze(1), hw_full).squeeze(1) > 0.5
            out["defense_set|cos_logit"] = defense.error_map(x, out_hw=hw_full)
            valid["defense_set|cos_logit"] = torch.ones_like(out["defense_set|cos_logit"], dtype=torch.bool)
            logits = align_output(model(norm(x))[0], hw_full)
            logp = torch.log_softmax(logits, 1)
            out["entropy"] = -(logp.exp() * logp).sum(1)
            valid["entropy"] = torch.ones_like(out["entropy"], dtype=torch.bool)
            return out, valid, logits.argmax(1)

    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        hw_full = tuple(y.shape[-2:])
        # fixed pixel sample per image (same for every score), seeded by image index
        samp = torch.stack([torch.from_numpy(np.random.default_rng(i).choice(hw_full[0] * hw_full[1], PIXELS_PER_IMAGE,
                                                                             replace=False)) for i in idx]).to(device)

        def sample(t):  # [B,H,W] -> [B*K] numpy
            return t.flatten(1).gather(1, samp).cpu().numpy().astype(np.float32).ravel()

        s_clean, v_clean, pred_clean = score_maps(x, hw_full)
        with torch.no_grad():
            meters["clean"].update(model(norm(x))[0], y)
        for k, s in s_clean.items():
            pix[k]["clean"].append(sample(s))
            for i in range(len(idx)):
                img[k]["clean"].append(float(s[i][v_clean[k][i]].mean()))

        if cache.has(idx):
            d_pgd = torch.clamp(cache.get(idx, device), -PGD_PARAMS["eps"], PGD_PARAMS["eps"])
            d_pgd = torch.clamp(d_pgd, -x, 1.0 - x)
        else:
            d_pgd = pgd(model, norm, x, y, **PGD_PARAMS)
            cache.put(idx, d_pgd)
        deltas = {"pgd": d_pgd, "segpgd": segpgd(model, norm, x, y, **PGD_PARAMS),
                  "pgd_loweps": pgd(model, norm, x, y, **LOW_EPS)}
        for a in attacks:
            xa = torch.clamp(x + deltas[a], 0.0, 1.0)
            s_adv, v_adv, pred_adv = score_maps(xa, hw_full)
            with torch.no_grad():
                meters[a].update(model(norm(xa))[0], y)
            labels[a].append(sample((pred_adv != pred_clean).float()) > 0.5)
            for k, s in s_adv.items():
                pix[k][a].append(sample(s))
                pix_valid[k][a].append(sample(v_adv[k].float()) > 0.5)
                for i in range(len(idx)):
                    img[k][a].append(float(s[i][v_adv[k][i]].mean()))
        print(f"  [{min(b + 2, len(indices)):3d}/{len(indices)}] {time.time() - t0:.0f}s", flush=True)
    cache.save()

    table = {}
    for k in pix:
        row = {}
        for a in attacks:
            ok = np.concatenate(pix_valid[k][a])
            lab = np.concatenate(labels[a])[ok]
            sc = np.concatenate(pix[k][a])[ok]
            img_scores = np.r_[img[k]["clean"], img[k][a]]
            img_lab = np.r_[np.zeros(len(img[k]["clean"]), bool), np.ones(len(img[k][a]), bool)]
            row[a] = {"pixel_auroc_changed": auroc(sc, lab), "image_auroc": auroc(img_scores, img_lab)}
        table[k] = row
    rates = {a: float(np.concatenate(labels[a]).mean()) for a in attacks}
    res = {"table": table, "changed_fraction_sampled": rates,
           "undefended_miou": {a: meters[a].miou() for a in meters},
           "config": {"model": args.model, "split": args.split, "seed": args.seed, "num_images": len(indices),
                      "pixels_per_image": PIXELS_PER_IMAGE, "views": VIEWS, "measures": MEASURES,
                      "attacks": {"pgd": PGD_PARAMS, "segpgd": PGD_PARAMS, "pgd_loweps": LOW_EPS},
                      "shift_px": SHIFT_PX, "rot_deg": 10, "preset": preset, "git": git_hash()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"p2ext_{args.model}_{args.split}_s{args.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"\n== Extended P2 {args.model} | {args.split} ({len(indices)} images) | seed {args.seed}")
    print("  undefended mIoU: " + ", ".join(f"{a} {m:.2f}" for a, m in res["undefended_miou"].items()))
    print("  changed fraction (sampled): " + ", ".join(f"{a} {r:.3f}" for a, r in rates.items()))
    print(f"  {'score':28s} " + " ".join(f"{a + ' px/img':>22s}" for a in attacks))
    for k in sorted(table, key=lambda k: -table[k]["pgd"]["pixel_auroc_changed"]):
        print(f"  {k:28s} " + " ".join(f"{table[k][a]['pixel_auroc_changed']:10.3f}/{table[k][a]['image_auroc']:<10.3f}"
                                       for a in attacks))
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
