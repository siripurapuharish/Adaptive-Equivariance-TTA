"""P2 go/no-go: does the per-pixel equivariance error separate clean from PGD-attacked inputs?

    python -m equi_lib.p2_separation --model clean --split tune100 --seed 0

Thresholds come from IMPLEMENTATION_PLAN.md and are fixed in advance:
  image level: AUROC of mean(e) for clean (neg) vs attacked (pos) images      >= 0.90  (gate)
  pixel level: AUROC of e_p on attacked images for "prediction changed by the
               attack" (pred(x_adv) != pred(x)), over all pixels             >= 0.70  (per-pixel budget)
Also reported, not used for the decision:
  - "became wrong": valid-label pixels that were correct on x and wrong on x_adv
  - pooled: changed pixels of attacked images (pos) vs all pixels of clean images (neg)
  - the entropy-only baseline (per-pixel softmax entropy instead of e), as the plan requires
Error maps use the defense's own view set (EquiDefense.error_map), upsampled to 340x680.
Writes results/improvements/p2/p2_<model>_<split>_s<seed>.json and example PNGs.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
from PIL import Image

from . import RESULTS_DIR, ROOT_DIR
from .attacks import PGD_PARAMS, pgd
from .cache import PGDCache
from .data import load_batch, split_indices
from .defense import EquiDefense
from .metrics import align_output
from .model import PRESETS, Normalizer, load_model
from .runner import git_hash
from .seed import set_seed
from .stats import auroc

warnings.filterwarnings("ignore")
IMAGE_THRESHOLD, PIXEL_THRESHOLD = 0.90, 0.70
OUT_DIR = os.path.join(RESULTS_DIR, "p2")


def entropy(logits):
    logp = torch.log_softmax(logits, dim=1)
    return -(logp.exp() * logp).sum(1)


def to_u8(a, vmax):
    return (np.clip(a / vmax, 0, 1) * 255).astype(np.uint8)


def save_example(path, x, e_clean, e_adv, changed, vmax):
    """Panel (half resolution): image | e on clean | e on attacked | pixels changed by the attack."""
    img = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = lambda a: np.repeat(a[..., None], 3, axis=2)  # noqa: E731
    panel = np.concatenate([img, gray(to_u8(e_clean, vmax)), gray(to_u8(e_adv, vmax)),
                            gray(changed.astype(np.uint8) * 255)], axis=1)
    Image.fromarray(panel).resize((panel.shape[1] // 2, panel.shape[0] // 2), Image.BILINEAR).save(path)


def histogram(values, bins):
    h, _ = np.histogram(values, bins=bins)
    return h.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--split", default="tune100", choices=["tune100", "test100", "full500"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()
    if args.split != "tune100":
        print("WARNING: the plan runs P2 on tune100 only; other splits are for checking, not for decisions.")

    device = torch.device("cuda")
    preset = PRESETS[args.model]
    model = load_model(os.path.join(ROOT_DIR, preset["ckpt"]), device)
    norm = Normalizer(device)
    defense = EquiDefense(model, norm, preset["eps_v"], preset["num_views"], device=device)
    cache = PGDCache(args.model, args.seed, PGD_PARAMS)
    indices = split_indices(args.split)[:args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(args.seed)

    img_e = {"clean": [], "adv": []}
    img_h = {"clean": [], "adv": []}
    px = {k: [] for k in ("e_adv", "h_adv", "e_clean", "h_clean", "changed", "valid", "clean_ok", "adv_ok")}
    examples, t0 = 0, time.time()
    for b in range(0, len(indices), 2):
        idx = indices[b:b + 2]
        x, y = load_batch(idx, device)
        if cache.has(idx):
            delta = torch.clamp(cache.get(idx, device), -PGD_PARAMS["eps"], PGD_PARAMS["eps"])
            delta = torch.clamp(delta, -x, 1.0 - x)
        else:
            delta = pgd(model, norm, x, y, **PGD_PARAMS)
            cache.put(idx, delta)
        x_adv = torch.clamp(x + delta, 0.0, 1.0)

        with torch.no_grad():
            lc = align_output(model(norm(x))[0], y.shape[-2:])
            la = align_output(model(norm(x_adv))[0], y.shape[-2:])
        pc, pa = lc.argmax(1), la.argmax(1)
        hc, ha = entropy(lc), entropy(la)
        ec = defense.error_map(x, out_hw=y.shape[-2:])
        ea = defense.error_map(x_adv, out_hw=y.shape[-2:])
        valid = y != 255

        for i in range(len(idx)):
            img_e["clean"].append(float(ec[i].mean())); img_e["adv"].append(float(ea[i].mean()))
            img_h["clean"].append(float(hc[i].mean())); img_h["adv"].append(float(ha[i].mean()))
        px["e_adv"].append(ea.cpu().numpy().astype(np.float32).ravel())
        px["h_adv"].append(ha.cpu().numpy().astype(np.float32).ravel())
        px["e_clean"].append(ec.cpu().numpy().astype(np.float32).ravel())
        px["h_clean"].append(hc.cpu().numpy().astype(np.float32).ravel())
        px["changed"].append((pa != pc).cpu().numpy().ravel())
        px["valid"].append(valid.cpu().numpy().ravel())
        px["clean_ok"].append(((pc == y) & valid).cpu().numpy().ravel())
        px["adv_ok"].append(((pa == y) & valid).cpu().numpy().ravel())

        for i in range(len(idx)):
            if examples < args.examples:
                vmax = float(np.percentile(ea[i].cpu().numpy(), 99))
                save_example(os.path.join(OUT_DIR, f"p2_{args.model}_{args.split}_s{args.seed}_ex{examples}_img{idx[i]}.png"),
                             x[i], ec[i].cpu().numpy(), ea[i].cpu().numpy(), (pa[i] != pc[i]).cpu().numpy(), vmax)
                examples += 1
        print(f"  [{min(b + 2, len(indices)):3d}/{len(indices)}] {time.time() - t0:.0f}s", flush=True)
    cache.save()

    P = {k: np.concatenate(v) for k, v in px.items()}
    lab_img = np.r_[np.zeros(len(img_e["clean"]), bool), np.ones(len(img_e["adv"]), bool)]
    became_wrong_region = P["clean_ok"]                       # valid pixels correct on clean
    pooled_scores_e = np.r_[P["e_adv"][P["changed"]], P["e_clean"]]
    pooled_scores_h = np.r_[P["h_adv"][P["changed"]], P["h_clean"]]
    pooled_labels = np.r_[np.ones(int(P["changed"].sum()), bool), np.zeros(P["e_clean"].size, bool)]

    res = {
        "image_auroc": {"equi_error": auroc(np.r_[img_e["clean"], img_e["adv"]], lab_img),
                        "entropy": auroc(np.r_[img_h["clean"], img_h["adv"]], lab_img)},
        "pixel_auroc_changed": {"equi_error": auroc(P["e_adv"], P["changed"]),
                                "entropy": auroc(P["h_adv"], P["changed"])},
        "pixel_auroc_became_wrong": {"equi_error": auroc(P["e_adv"][became_wrong_region], ~P["adv_ok"][became_wrong_region]),
                                     "entropy": auroc(P["h_adv"][became_wrong_region], ~P["adv_ok"][became_wrong_region])},
        "pixel_auroc_pooled_changed_vs_clean": {"equi_error": auroc(pooled_scores_e, pooled_labels),
                                                "entropy": auroc(pooled_scores_h, pooled_labels)},
        "rates": {"changed_fraction_all_pixels": float(P["changed"].mean()),
                  "became_wrong_fraction_of_clean_correct": float((~P["adv_ok"][became_wrong_region]).mean())},
        "image_scores": {"equi_error": img_e, "entropy": img_h},
        "mean_pixel_e": {"clean": float(P["e_clean"].mean()), "adv": float(P["e_adv"].mean()),
                         "adv_changed": float(P["e_adv"][P["changed"]].mean()) if P["changed"].any() else None,
                         "adv_unchanged": float(P["e_adv"][~P["changed"]].mean()) if (~P["changed"]).any() else None},
    }
    hi = float(np.percentile(np.r_[P["e_clean"], P["e_adv"]], 99.9))
    bins = np.linspace(0.0, hi, 51)
    res["pixel_histograms_e"] = {"bin_edges": bins.tolist(), "clean": histogram(P["e_clean"], bins),
                                 "adv_changed": histogram(P["e_adv"][P["changed"]], bins),
                                 "adv_unchanged": histogram(P["e_adv"][~P["changed"]], bins)}
    # Attacked inputs can have LOWER entropy (confidently wrong), so also report the reversed
    # direction; a fair baseline may use either sign, chosen on tune100.
    for k in ("image_auroc", "pixel_auroc_changed", "pixel_auroc_became_wrong", "pixel_auroc_pooled_changed_vs_clean"):
        res[k]["entropy_reversed"] = 1.0 - res[k]["entropy"]
    img_pass = res["image_auroc"]["equi_error"] >= IMAGE_THRESHOLD
    px_pass = res["pixel_auroc_changed"]["equi_error"] >= PIXEL_THRESHOLD
    res["decision"] = {"image_pass": bool(img_pass), "pixel_pass": bool(px_pass),
                       "thresholds": {"image": IMAGE_THRESHOLD, "pixel": PIXEL_THRESHOLD}}
    res["config"] = {"model": args.model, "split": args.split, "seed": args.seed, "num_images": len(indices),
                     "preset": preset, "pgd": PGD_PARAMS, "git": git_hash(),
                     "error_map": "1 - mean_t cos(f0, f_t) over 19 channels at 43x85, rotation views masked, "
                                  "bilinear to 340x680; views = defense view set"}
    out = os.path.join(OUT_DIR, f"p2_{args.model}_{args.split}_s{args.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"\n== P2 {args.model} | {args.split} ({len(indices)} images) | seed {args.seed}")
    for k in ("image_auroc", "pixel_auroc_changed", "pixel_auroc_became_wrong", "pixel_auroc_pooled_changed_vs_clean"):
        print(f"  {k:38s} equi {res[k]['equi_error']:.4f} | entropy {res[k]['entropy']:.4f} "
              f"(reversed {res[k]['entropy_reversed']:.4f})")
    print(f"  changed fraction {res['rates']['changed_fraction_all_pixels']:.3f}; "
          f"became-wrong fraction {res['rates']['became_wrong_fraction_of_clean_correct']:.3f}")
    print(f"  DECISION: image {'PASS' if img_pass else 'FAIL'} (>= {IMAGE_THRESHOLD}), "
          f"pixel {'PASS' if px_pass else 'FAIL'} (>= {PIXEL_THRESHOLD})")
    print(f"  saved {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
