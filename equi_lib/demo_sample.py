"""One demo-style figure for the clean model with the final v2 config (what a Gradio demo would show).

    python -m equi_lib.demo_sample [--pos 5]

Row 1 = clean input, row 2 = PGD-50 input (eps 4/255).
Columns: input | ground truth | prediction, no defense | equivariance error map + gate decision | v2 prediction.
Writes results/improvements/p7/demo_sample_clean.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from equi_lib import RESULTS_DIR  # noqa: E402
from equi_lib.adaptive import AdaptiveDefense  # noqa: E402
from equi_lib.attacks import PGD_PARAMS, pgd  # noqa: E402
from equi_lib.averaging import predict  # noqa: E402
from equi_lib.cache import PGDCache  # noqa: E402
from equi_lib.data import load_batch, split_indices  # noqa: E402
from equi_lib.metrics import ConfusionMeter  # noqa: E402
from equi_lib.model import PRESETS, Normalizer, load_model  # noqa: E402
from equi_lib.seed import set_seed  # noqa: E402

PALETTE = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 80, 100), (0, 0, 230), (119, 11, 32)], dtype=np.uint8)


def colour(lbl):
    out = np.zeros(lbl.shape + (3,), np.uint8)
    ok = lbl != 255
    out[ok] = PALETTE[lbl[ok]]
    return out


def heat(e, vmax):
    t = np.clip(e / vmax, 0, 1)
    rgb = np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], -1)
    return (rgb * 255).astype(np.uint8)


def miou(logp, y):
    m = ConfusionMeter()
    m.update(logp, y)
    return m.miou()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=5, help="position in test100")
    args = ap.parse_args()
    device = torch.device("cuda")
    preset = PRESETS["clean"]
    model = load_model(os.path.join(ROOT, preset["ckpt"]), device)
    norm = Normalizer(device)
    dfn = AdaptiveDefense(model, norm, 6.0 / 255, preset["num_views"], device=device, model_tag="clean",
                          mode="gate+perimage", view_set="up1.5")
    idx = [split_indices("test100")[args.pos]]
    x, y = load_batch(idx, device)
    hw = y.shape[-2:]
    set_seed(0)
    cache = PGDCache("clean", 0, PGD_PARAMS)
    if cache.has(idx):
        d = torch.clamp(cache.get(idx, device), -PGD_PARAMS["eps"], PGD_PARAMS["eps"])
        d = torch.clamp(d, -x, 1.0 - x)
    else:
        d = pgd(model, norm, x, y, **PGD_PARAMS)
    xa = torch.clamp(x + d, 0.0, 1.0)

    rows, maps = [], []
    for name, inp in (("clean", x), ("PGD-50", xa)):
        e = F.interpolate(dfn.inner.consistency_map(inp).unsqueeze(1), size=hw, mode="bilinear",
                          align_corners=False)[0, 0].cpu().numpy()
        x_def = dfn(inp, y)
        st = dfn.pop_stats()[0]
        with torch.no_grad():
            p_plain = predict(model, norm, inp, "single", hw)
            p_v2 = predict(model, norm, x_def, "avg_rs", hw)
        rows.append(dict(name=name, img=(inp[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                         plain=colour(p_plain.argmax(1)[0].cpu().numpy()), plain_miou=miou(p_plain, y),
                         v2=colour(p_v2.argmax(1)[0].cpu().numpy()), v2_miou=miou(p_v2, y),
                         score=st["score"], defended=st["defended"]))
        maps.append(e)
    vmax = float(np.percentile(maps[1], 99))
    gt = colour(y[0].cpu().numpy())

    pw, ph, top, gap = 408, 204, 46, 6
    W = 5 * pw + 6 * gap
    H = 40 + 2 * (top + ph) + 3 * gap + 40
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    try:
        f_big, f = ImageFont.truetype("arialbd.ttf", 20), ImageFont.truetype("arial.ttf", 15)
    except OSError:
        f_big = f = ImageFont.load_default()
    dr.text((gap, 8), f"Clean DRN-D-22, Cityscapes val #{idx[0]}  |  gate threshold tau = {dfn.tau:.3f}  |  "
                      f"final v2 defense (eps_v 6/255, one step, averaging over 4 views)", font=f_big, fill=(0, 0, 0))
    for r, (row, e) in enumerate(zip(rows, maps)):
        y0 = 40 + gap + r * (top + ph + gap)
        gate = "ON -> defend" if row["defended"] else "OFF -> skip defense"
        titles = [f"{row['name']} input",
                  "ground truth",
                  f"no defense: mIoU {row['plain_miou']:.1f}",
                  f"equivariance error  s = {row['score']:.3f}\ngate {gate}",
                  f"ours (v2): mIoU {row['v2_miou']:.1f}"]
        panels = [row["img"], gt, row["plain"], heat(e, vmax), row["v2"]]
        for c, (t, p) in enumerate(zip(titles, panels)):
            x0 = gap + c * (pw + gap)
            col = (200, 30, 30) if (c == 3 and row["defended"]) else (0, 0, 0)
            dr.multiline_text((x0 + 2, y0), t, font=f, fill=col, spacing=2)
            canvas.paste(Image.fromarray(p).resize((pw, ph), Image.BILINEAR), (x0, y0 + top - 4))
    dr.text((gap, H - 32), "The PGD image looks identical to the clean one (change <= 4/255 per pixel) but fools the "
                           "plain model. The error map (bright = model disagrees with its own 0.5x-downscaled view) "
                           "lights up, so the gate switches the defense on.", font=f, fill=(60, 60, 60))
    out = os.path.join(RESULTS_DIR, "p7", "demo_sample_clean.png")
    canvas.save(out)
    for row in rows:
        print(f"{row['name']}: plain {row['plain_miou']:.2f} | v2 {row['v2_miou']:.2f} | s {row['score']:.3f} "
              f"| defended {row['defended']}")
    print("saved", out)


if __name__ == "__main__":
    main()
