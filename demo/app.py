"""Gradio demo: an invisible attack breaks the model; the equivariance defense notices and repairs it.

    python -m demo.app                 # opens http://127.0.0.1:7860
    python -m demo.app --calibrate     # once: self-check threshold from tune100 clean images

Layout follows the "Story view" design (claude.ai artifact "Gradio Demo Preview"). Everything shown is computed
live by the same equi_lib code that produced the reported results. Methods come from demo/methods.json, so a new
defense is one more entry there. PGD-50 and BPDA-50 perturbations are reused from results/improvements/cache when
they exist (PGD seed 0: all 500 val images; BPDA-50 through v2: the first 50 test100 images), else computed.
"""
import argparse
import base64
import html
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from equi_lib.attacks import PGD_PARAMS, pgd  # noqa: E402
from equi_lib.averaging import predict  # noqa: E402
from equi_lib.cache import CACHE_DIR, PGDCache  # noqa: E402
from equi_lib.data import IMAGE_PATHS, load_batch, split_indices  # noqa: E402
from equi_lib.defenses import DEFENSE_TAKES_OPTS, DEFENSES  # noqa: E402
from equi_lib.e3_selfcheck import HELD_OUT, view_error  # noqa: E402
from equi_lib.metrics import ConfusionMeter  # noqa: E402
from equi_lib.model import PRESETS, Normalizer, load_model  # noqa: E402
from equi_lib.p6b_attacks import ATTACK, Defended, attack  # noqa: E402
from equi_lib.seed import set_seed  # noqa: E402

warnings.filterwarnings("ignore")
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS_PATH = os.path.join(DEMO_DIR, "methods.json")
CALIB_PATH = os.path.join(DEMO_DIR, "selfcheck_calib.json")
H, W = 340, 680
EPS = ATTACK["eps"]
ERR_VMAX = 1.2          # error-map colour scale (softmax L2 is in [0, sqrt 2])
SEED = 0

PALETTE = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 80, 100), (0, 0, 230), (119, 11, 32)], dtype=np.uint8)
MODEL_LABELS = {"Clean-trained": "clean", "Adv-trained": "adv"}
ATTACK_LABELS = {"None": "none", "PGD-50": "pgd", "SegPGD-50": "segpgd", "BPDA-50 (adaptive)": "bpda"}
EXAMPLES = split_indices("test100")
DEFAULT_EXAMPLE = 25

# Headline numbers: full500, seed 42 (IMPLEMENTATION_PLAN.md, P7 'full500 outcome'); v2 seeds 0-2 agree within 0.6.
HEADLINE = {
    "clean": [("59.21", "clean accuracy with defense (paper 35.66)"),
              ("31.66", "mIoU under PGD-50 (paper 29.38)"),
              ("0.41 s", "per image, vs ≈7.2 s for the paper")],
    "adv": [("49.97", "clean accuracy with defense (paper 44.06)"),
            ("30.12", "mIoU under PGD-50 (paper 28.48)"),
            ("0.35 s", "per image, vs ≈8.3 s for the paper")],
}
# Adaptive-attack numbers for the verdict (test100 first 50, seed 0: E5; image AUROC of the held-out check on v2)
BPDA_FACTS = {"clean": {"ours": "3.85", "auroc": "0.84"}, "adv": {"ours": "25.36", "auroc": "0.94"}}


# ---------------------------------------------------------------- engine

class Engine:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.norm = Normalizer(self.device)
        with open(METHODS_PATH) as f:
            cfg = json.load(f)
        self.models, self.methods = {}, {}
        for tag in ("clean", "adv"):
            preset = PRESETS[tag]
            model = load_model(os.path.join(ROOT, preset["ckpt"]), self.device)
            self.models[tag] = model
            self.methods[tag] = []
            for m in cfg[tag]:
                eps_v = m["eps_v"] / 255.0 if "eps_v" in m else preset["eps_v"]
                kw = dict(m.get("def_opts", {}))
                if m["defense"] in DEFENSE_TAKES_OPTS:
                    kw["model_tag"] = tag
                dfn = DEFENSES[m["defense"]](model, self.norm, eps_v, preset["num_views"], device=self.device, **kw)
                self.methods[tag].append(dict(m, dfn=dfn))
        self.calib = json.load(open(CALIB_PATH)) if os.path.exists(CALIB_PATH) else {}
        self._bpda = {}

    def target(self, tag):
        ms = self.methods[tag]
        return next((m for m in ms if m.get("attack_target")), next((m for m in ms if m.get("highlight")), ms[-1]))

    def bpda_cache(self, tag):
        if tag not in self._bpda:
            p = os.path.join(CACHE_DIR, f"e5_bpda50v2_{tag}_test100_n50_s0.pt")
            self._bpda[tag] = torch.load(p) if os.path.exists(p) else {}
        return self._bpda[tag]

    def pred(self, tag, x, variant="single"):
        with torch.no_grad():
            return predict(self.models[tag], self.norm, x, variant, (H, W))

    def make_attack(self, tag, kind, x, y, idx):
        model = self.models[tag]
        if kind == "none":
            return x, "none", 0.0
        t0 = time.time()
        if kind == "pgd":
            cache = PGDCache(tag, SEED, PGD_PARAMS)
            if idx is not None and cache.has([idx]):
                d, src = torch.clamp(cache.get([idx], self.device), -EPS, EPS), "cached"
            else:
                d, src = pgd(model, self.norm, x, y, **PGD_PARAMS), "live"
        elif kind == "segpgd":
            d, src = attack(model, self.norm, None, x, y, "segpgd", "none", 50, 1, EPS, ATTACK["step"]), "live"
        else:
            tgt = self.target(tag)
            cache = self.bpda_cache(tag)
            if idx is not None and idx in cache and tgt.get("attack_target"):
                d, src = cache[idx].unsqueeze(0).to(self.device).float(), "cached"
            else:
                d = attack(model, self.norm, Defended(tgt["dfn"], False), x, y, "ce", "bpda", 50, 1, EPS,
                           ATTACK["step"], predict=tgt.get("predict", "single"))
                src = "live"
        d = torch.clamp(torch.clamp(d, -EPS, EPS), -x, 1.0 - x)
        torch.cuda.synchronize()
        return torch.clamp(x + d, 0.0, 1.0), src, time.time() - t0

    def run_method(self, tag, m, inp, y):
        dfn = m["dfn"]
        torch.cuda.synchronize()
        t0 = time.time()
        x_def = dfn(inp, y)
        logp = self.pred(tag, x_def, m.get("predict", "single"))
        torch.cuda.synchronize()
        sec = time.time() - t0
        stats = dfn.pop_stats()[-1] if hasattr(dfn, "pop_stats") else None
        return x_def, logp, sec, stats

    def selfcheck(self, tag, m, x_def):
        inner = getattr(m["dfn"], "inner", None)
        if inner is None:
            return None
        with torch.no_grad():
            return float(view_error(inner, x_def, HELD_OUT[tag]).mean())


def calibrate():
    """Self-check threshold per model: 95th percentile of the held-out score of v2-defended tune100 clean images."""
    eng = Engine()
    out = {}
    for tag in ("clean", "adv"):
        m = eng.target(tag)
        set_seed(SEED)
        scores = []
        for i in split_indices("tune100"):
            x, y = load_batch([i], eng.device)
            x_def, _, _, _ = eng.run_method(tag, m, x, y)
            scores.append(eng.selfcheck(tag, m, x_def))
        out[tag] = {"t": float(np.percentile(scores, 95)), "pct": 95, "n": len(scores), "method": m["name"],
                    "views": [str(v) for v in HELD_OUT[tag]], "clean_median": float(np.median(scores))}
        print(f"{tag}: threshold {out[tag]['t']:.4f} (median clean {out[tag]['clean_median']:.4f})")
    with open(CALIB_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {os.path.relpath(CALIB_PATH, ROOT)}")


# ---------------------------------------------------------------- images

def to_tensor(img):
    """HxWx3 uint8 -> [1,3,340,680] in [0,1]; centre-crops to 2:1 first."""
    im = Image.fromarray(img).convert("RGB")
    w, h = im.size
    if w / h > 2:
        nw = int(h * 2)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / 2)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    im = im.resize((W, H), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def load_example(i):
    return (np.asarray(Image.open(os.path.join(ROOT, "cityscapes_dataset_subsampled", IMAGE_PATHS[i])).convert("RGB")))


def colour(lbl):
    out = np.zeros(lbl.shape + (3,), np.uint8)
    ok = lbl != 255
    out[ok] = PALETTE[lbl[ok]]
    return out


def heat(e):
    t = np.clip(e / ERR_VMAX, 0, 1)
    return (np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], -1) * 255).astype(np.uint8)


def uri(arr, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, quality=90)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def img_np(x):
    return (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def score(logp, y):
    m = ConfusionMeter()
    m.update(logp, y)
    return m.miou()


# ---------------------------------------------------------------- page pieces

E = html.escape


def fig(title, src, alt, extra="", cls="", foot=""):
    return (f'<figure class="card {cls}"><figcaption>{title}{extra}</figcaption>'
            f'<img src="{src}" alt="{E(alt)}">{foot}</figure>')


def metric_row(val, label, sec, slowest, accent=False, danger=False):
    pct = max(2.0, 100.0 * sec / max(slowest, 1e-6))
    cls = "accent" if accent else ("danger" if danger else "")
    return (f'<div class="metric"><span class="mval"><span class="mlbl">{label}</span>'
            f'<span class="mnum {cls}">{val}</span></span>'
            f'<div class="bar"><div class="{ "fill accent" if accent else "fill"}" style="width:{pct:.1f}%"></div></div>'
            f'<span class="mtime {cls}">{sec:.2f} s</span></div>')


def build_page(r):
    tag, a, gt = r["tag"], r["attack"], r["gt"] is not None
    lbl = "mIoU" if gt else "agree %"
    fmt = (lambda v: f"{v:.1f}")
    attacked = a != "none"
    slowest = max([r["plain_sec"]] + [m["sec"] for m in r["methods"]])
    hl = next((m for m in r["methods"] if m["highlight"]), r["methods"][-1])
    paper = next((m for m in r["methods"] if m["defense"] == "original"), None)
    atk_name = {"none": "no attack", "pgd": "PGD-50", "segpgd": "SegPGD-50", "bpda": "BPDA-50"}[a]

    # ---- narration (story steps)
    scene = (f"Cityscapes val #{r['idx']}. The model labels every pixel: road, car, tree, sign."
             if gt else "Your image, centre-cropped and resized to 680×340. There is no ground truth, so scores are "
                        "agreement with the model's own prediction on the clean image.")
    if a == "none":
        n2 = f"No attack: the input stays clean. No defense scores {fmt(r['plain_score'])} {lbl}."
    elif a == "bpda":
        n2 = (f"BPDA-50 knows our defense and attacks straight through it ({r['atk_src']}). "
              f"No defense: {fmt(r['clean_score'])} → {fmt(r['plain_score'])} {lbl}.")
    else:
        n2 = (f"{atk_name} changes each pixel by at most 4/255, invisible to us ({r['atk_src']}). "
              f"No defense: {fmt(r['clean_score'])} → {fmt(r['plain_score'])} {lbl}.")
    st = hl.get("stats")
    if st is None or not hl["gated"]:
        n3 = (f"{hl['name']} compares the model with its own 0.5× view (error map) and runs one defense step on "
              f"every image, then averages 4 views.")
    elif st["defended"]:
        n3 = (f"The model disagrees with its own 0.5× view: gate score {st['score']:.3f} ≥ τ {r['tau']:.3f}, so the "
              f"gate switches on. One step plus 4-view averaging.")
    else:
        n3 = (f"The model agrees with its own 0.5× view: gate score {st['score']:.3f} < τ {r['tau']:.3f}, so the "
              f"defense is skipped. No time spent, no accuracy lost.")
    n4 = (f"{hl['name']}: {fmt(hl['score'])} {lbl} in {hl['sec']:.2f} s" +
          (f"; paper: {fmt(paper['score'])} in {paper['sec']:.1f} s (≈{paper['sec'] / max(hl['sec'], 1e-3):.0f}× "
           f"slower)." if paper else "."))
    steps = [("Scene", scene), ("Attack", n2), ("Detect &amp; defend", n3), ("Compare", n4)]
    story = "".join(f'<li class="step" style="animation-delay:{0.15 + 0.55 * i:.2f}s"><span class="chip">'
                    f'<span class="mono">{i + 1}</span> {t}</span><span class="say">{E(s)}</span></li>'
                    for i, (t, s) in enumerate(steps))

    # ---- left column: input -> attacked
    left = fig("Original image", uri(r["x_img"], "JPEG"), "Original image")
    if attacked:
        foot = (f'<div class="pert"><img src="{uri(r["pert_img"], "JPEG")}" alt="The perturbation, amplified">'
                f'<span>the change, amplified ×25 · max 4/255 per pixel</span></div>')
        left += (f'<div class="arrow danger"><span>+{atk_name}</span></div>' +
                 fig("Attacked image", uri(r["adv_img"], "JPEG"), "Attacked image; looks identical", cls="danger",
                     foot=foot))
    else:
        left += '<div class="arrow"><span>no attack</span></div>' + fig("Input (no attack)", uri(r["x_img"], "JPEG"),
                                                                        "Unmodified input")

    # ---- middle column: the paths
    plain_danger = attacked and r["plain_score"] < 0.5 * r["clean_score"]
    mid = fig("No defense", uri(r["plain_img"]), "Prediction without defense", cls="path",
              foot=metric_row(fmt(r["plain_score"]), lbl, r["plain_sec"], slowest, danger=plain_danger))
    for m in r["methods"]:
        extra = (f'<span class="mono small">≈{slowest / max(m["sec"], 1e-3):.0f}× faster</span>'
                 if m["highlight"] and paper and m is not paper else f'<span class="sub">{E(m["note"])}</span>')
        mid += fig(E(m["name"]), uri(m["img"]), f"Prediction with {m['name']}", extra=extra,
                   cls="path hl" if m["highlight"] else "path",
                   foot=metric_row(fmt(m["score"]), lbl, m["sec"], slowest, accent=m["highlight"]))

    # ---- right column: ground truth, error map + gate, self-check
    right = ""
    if gt:
        right += fig("Ground truth", uri(r["gt_img"]), "Ground-truth segmentation",
                     extra='<span class="sub">answer key</span>')
    if hl["gated"] and st is not None:
        badge = ('<span class="badge on">GATE ON → DEFEND</span>' if st["defended"]
                 else '<span class="badge off">GATE OFF</span>')
        cap = (f"Score {st['score']:.3f} {'≥' if st['defended'] else '<'} τ {r['tau']:.3f}: "
               f"{'looks attacked, defend.' if st['defended'] else 'looks clean, skip the defense.'}")
    else:
        badge, cap = '<span class="badge on">NO GATE</span>', "This model always runs one defense step."
    right += fig("EqVar error map", uri(r["err_img"]), "Error map: bright where the model disagrees with its 0.5× view",
                 extra=badge, cls="hlb", foot=f'<p class="cap">{E(cap)} Bright = the model disagrees with its own '
                                             f'0.5× view.</p>')
    sc = r.get("selfcheck")
    if sc is not None:
        if sc["t"] is not None:
            alarm = sc["score"] > sc["t"]
            sbadge = '<span class="badge alarm">ALARM</span>' if alarm else '<span class="badge off">OK</span>'
            stext = (f"Held-out views ({', '.join(sc['views'])}) on the defended image: score {sc['score']:.3f} vs "
                     f"clean limit {sc['t']:.3f}. " +
                     ("The defended image still disagrees with views the defense never used: don't trust this "
                      "output." if alarm else "Consistent with clean images."))
        else:
            sbadge, stext = "", f"Held-out score {sc['score']:.3f} (run `python -m demo.app --calibrate` for a limit)."
        right += (f'<div class="card"><div class="fc">Self-check{sbadge}</div><p class="cap">{E(stext)}</p></div>')

    # ---- verdict
    if a == "bpda":
        f = BPDA_FACTS[tag]
        alarm = sc is not None and sc["t"] is not None and sc["score"] > sc["t"]
        vlabel, vk = ("ALARM" if alarm else "LIMITATION"), "bad"
        verdict = (f"Shown openly: an attacker who knows the defense breaks it ({hl['name']} {fmt(hl['score'])} {lbl} "
                   f"here; {f['ours']} mIoU on 50 test images). The self-check " +
                   ("flags this image. " if alarm else "does not flag this image. ") +
                   f"On 50 test images it separates such attacks from clean images with AUROC {f['auroc']}, but an "
                   f"attacker who also targets the check can hide from it." +
                   (" The paper's defense here faces an attack aimed at ours." if paper else ""))
    else:
        vlabel, vk = "RESULT", "good"
        if a == "none":
            verdict = (f"Clean image: {hl['name']} keeps {fmt(hl['score'])} {lbl} (no defense {fmt(r['plain_score'])}"
                       + (f", paper {fmt(paper['score'])}" if paper else "") + ").")
        else:
            verdict = (f"{atk_name}: no defense {fmt(r['plain_score'])} → {hl['name']} {fmt(hl['score'])} {lbl} in "
                       f"{hl['sec']:.2f} s" + (f"; the paper's defense {fmt(paper['score'])} in {paper['sec']:.1f} s"
                                               if paper else "") + ".")
    stats = "".join(f'<div class="stat"><span class="sv">{v}</span><span class="sl">{E(l)}</span></div>'
                    for v, l in HEADLINE[tag])
    note = ("Headline numbers: 500 Cityscapes validation images, seed 42 (ours over seeds 0–2 agrees within 0.6). "
            "Times on this page are measured live on this GPU.")
    return f"""
<div class="eqd">
  <ol class="story">{story}</ol>
  <section class="stage" aria-label="What happens to the image">
    <div class="col">{left}</div>
    <div class="col fan">{mid}</div>
    <div class="col">{right}</div>
  </section>
  <section class="verdict reveal" aria-label="Verdict">
    <div class="vbox {vk}"><span class="vlab">{vlabel}</span><p>{E(verdict)}</p></div>
    <div class="stats">{stats}</div>
    <p class="note">{E(note)}</p>
  </section>
</div>"""


# ---------------------------------------------------------------- run

def run(engine, img, idx, model_label, attack_label, progress):
    tag, a = MODEL_LABELS[model_label], ATTACK_LABELS[attack_label]
    dev = engine.device
    set_seed(SEED)
    if idx is not None:
        x, y = load_batch([idx], dev)
    else:
        if img is None:
            raise ValueError("Pick an example or upload an image first.")
        x, y = to_tensor(img).to(dev), None
    clean_logp = engine.pred(tag, x)
    y_eval = y if y is not None else clean_logp.argmax(1)            # uploads: score agreement with the clean prediction
    progress(0.1, desc=f"Attack: {attack_label}")
    inp, src, _ = engine.make_attack(tag, a, x, y_eval, idx)
    torch.cuda.synchronize()
    t0 = time.time()
    plain_logp = engine.pred(tag, inp)
    torch.cuda.synchronize()
    plain_sec = time.time() - t0

    def sc(logp):
        return score(logp, y_eval) if y is not None else float((logp.argmax(1) == y_eval).float().mean() * 100)

    r = {"tag": tag, "attack": a, "idx": idx, "gt": y, "atk_src": src, "plain_sec": plain_sec,
         "clean_score": sc(clean_logp), "plain_score": sc(plain_logp), "methods": []}
    hl_dfn = None
    for k, m in enumerate(engine.methods[tag]):
        progress(0.3 + 0.6 * k / len(engine.methods[tag]), desc=f"Defense: {m['name']}")
        x_def, logp, sec, stats = engine.run_method(tag, m, inp, y_eval)
        gated = bool(m.get("def_opts", {}).get("mode", "").startswith("gate"))
        r["methods"].append({"name": m["name"], "note": m.get("note", ""), "defense": m["defense"],
                             "highlight": bool(m.get("highlight")), "gated": gated, "stats": stats, "sec": sec,
                             "score": sc(logp), "img": colour(logp.argmax(1)[0].cpu().numpy().astype(np.int64))})
        if m.get("highlight"):
            hl_dfn = m
            calib = engine.calib.get(tag, {})
            s = engine.selfcheck(tag, m, x_def)
            if s is not None:
                r["selfcheck"] = {"score": s, "t": calib.get("t"), "views": [str(v) for v in HELD_OUT[tag]]}
    progress(0.95, desc="Drawing")
    r["tau"] = getattr(hl_dfn["dfn"], "tau", float("nan")) if hl_dfn else float("nan")
    with torch.no_grad():
        err = hl_dfn["dfn"].inner.consistency_map(inp) if hl_dfn and hasattr(hl_dfn["dfn"], "inner") else None
    if err is not None:
        err = F.interpolate(err.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        r["err_img"] = heat(err)
    else:
        r["err_img"] = np.zeros((H, W, 3), np.uint8)
    r["x_img"], r["adv_img"] = img_np(x), img_np(inp)
    r["pert_img"] = np.clip(128 + (r["adv_img"].astype(np.int16) - r["x_img"].astype(np.int16)) * 25, 0,
                            255).astype(np.uint8)
    r["plain_img"] = colour(plain_logp.argmax(1)[0].cpu().numpy().astype(np.int64))
    if y is not None:
        r["gt_img"] = colour(y[0].cpu().numpy())
    return build_page(r)


# ---------------------------------------------------------------- UI

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:'
         'wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">')
# dark by default (the design's default theme); ?__theme=light switches
FORCE_DARK = """() => { const u = new URL(window.location); if (!u.searchParams.get('__theme')) {
  u.searchParams.set('__theme', 'dark'); window.location.replace(u.href); } }"""

CSS = """
:root, .eqroot { --bg:#f6f5f2; --panel:#ffffff; --stage:#fbfaf8; --grid:rgba(23,24,27,.06); --setupBg:#fff6ee;
  --setupBd:#f1dccb; --optBg:#ffffff; --border:#dedcd5; --border2:#a8a59c; --text:#17181b; --muted:#55585f;
  --accent:#c2410c; --onAccent:#ffffff; --accentInk:#9a3412; --accentSoft:#fff1e6; --line:#7d818a; --track:#ecebe6;
  --danger:#b91c1c; --infoInk:#1e3a8a; --infoSoft:#e6eefc; --shadow:0 1px 3px rgba(23,24,27,.08);
  --glow:0 0 0 4px rgba(194,65,12,.10),0 8px 24px rgba(194,65,12,.12); --badBg:#fdecec; --badBd:#b91c1c; --badTag:#991b1b; }
.dark, .dark .eqroot { --bg:#0d0f13; --panel:#161a21; --stage:#11141a; --grid:rgba(255,255,255,.045); --setupBg:#171310;
  --setupBd:#3a2a1d; --optBg:#0f1115; --border:#2a303b; --border2:#4a5263; --text:#eceef2; --muted:#a3aab7;
  --accent:#fb923c; --onAccent:#1a0f05; --accentInk:#fdba74; --accentSoft:#2a1a0e; --line:#8b93a3; --track:#262b34;
  --danger:#f87171; --infoInk:#bcd6ff; --infoSoft:#13233a; --shadow:0 1px 2px rgba(0,0,0,.4);
  --glow:0 0 0 4px rgba(251,146,60,.15),0 8px 24px rgba(251,146,60,.18); --badBg:#2c1214; --badBd:#f87171; --badTag:#fca5a5; }
body, gradio-app, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width: 1344px !important; margin: 0 auto !important; padding: 28px 16px 56px !important;
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important; color: var(--text); }
footer { display: none !important; }
.mono { font-family: 'IBM Plex Mono', monospace; }

/* header */
.hdr { display:flex; flex-direction:column; gap:8px; max-width:900px; margin-bottom:6px; }
.hdr .kick { font-family:'IBM Plex Mono',monospace; font-size:13px; letter-spacing:.08em; color:var(--accentInk); }
.hdr h1 { margin:0; font-family:'Space Grotesk',sans-serif; font-size:40px; line-height:1.1; font-weight:700;
  letter-spacing:-.02em; color:var(--text); }
.hdr p { margin:0; font-size:16px; line-height:1.5; color:var(--muted); }
.hdr a { color:var(--accentInk); }

/* setup strip */
.setup { border-radius:18px !important; padding:20px 8px !important; border:1px solid var(--setupBd) !important;
  background:var(--setupBg) !important; box-shadow:var(--shadow); gap:0 !important; }
.setup > div { padding:0 16px !important; border-right:1px solid var(--setupBd); }
.setup > div:last-child { border-right:none; }
@media (max-width: 980px) { .setup > div { border-right:none; } }
.sh { display:flex; gap:12px; align-items:flex-start; margin-bottom:10px; }
.sh .num { flex:none; width:30px; height:30px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:14px; font-weight:700; background:var(--accent); color:var(--onAccent); }
.sh .t { display:flex; flex-direction:column; gap:2px; }
.sh .t b { font-size:15px; letter-spacing:.04em; color:var(--text); }
.sh .t span { font-size:13px; color:var(--muted); }
.setup .block, .setup .form { background:transparent !important; border:none !important; box-shadow:none !important; }
.setup label.selected, .setup .wrap label:has(input:checked) { border-color:var(--accent) !important;
  background:var(--optBg) !important; box-shadow:0 0 0 1px var(--accent) inset !important; }
.setup .wrap label { min-height:52px; border-radius:12px !important; border:1px solid var(--setupBd) !important;
  background:var(--optBg) !important; width:100%; font-weight:700; }
#runbtn { min-height:64px; border-radius:14px; font-size:17px; font-weight:700; background:var(--accent) !important;
  color:var(--onAccent) !important; border:none !important; box-shadow:var(--glow); }
.hint { margin:12px 0 0; font-size:14px; line-height:1.45; border-radius:12px; padding:12px 14px;
  background:var(--optBg); border:1px solid var(--setupBd); color:var(--muted); }

/* results */
.eqd { display:flex; flex-direction:column; gap:18px; color:var(--text); font-family:'IBM Plex Sans',system-ui,sans-serif; }
.eqd .story { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; }
.eqd .step { display:flex; flex-direction:column; gap:8px; border-radius:12px; padding:12px 14px; background:var(--panel);
  border:1px solid var(--border); border-left:4px solid var(--accent); opacity:0; animation:eqin .5s ease forwards; }
.eqd .chip { align-self:flex-start; font-size:14px; padding:4px 12px; border-radius:999px; background:var(--accent);
  color:var(--onAccent); font-weight:600; }
.eqd .say { font-size:15px; line-height:1.45; }
@keyframes eqin { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
.eqd .stage { display:grid; grid-template-columns:1fr 1.15fr 1fr; gap:28px; padding:20px; border-radius:16px;
  border:1px solid var(--border); background-color:var(--stage); box-shadow:var(--shadow);
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:24px 24px; }
@media (max-width: 980px) { .eqd .stage { grid-template-columns:1fr; } }
.eqd .col { display:flex; flex-direction:column; gap:12px; justify-content:center; min-width:0; }
.eqd .col:nth-child(1) { opacity:0; animation:eqin .5s ease .15s forwards; }
.eqd .col:nth-child(2) { opacity:0; animation:eqin .5s ease .9s forwards; }
.eqd .col:nth-child(3) { opacity:0; animation:eqin .5s ease 1.5s forwards; }
.eqd .reveal { opacity:0; animation:eqin .5s ease 2.2s forwards; }
.eqd .card { margin:0; box-sizing:border-box; padding:10px; border-radius:12px; display:flex; flex-direction:column;
  gap:6px; background:var(--panel); border:1px solid var(--border); box-shadow:var(--shadow); }
.eqd .card.danger { border-color:var(--danger); }
.eqd .card.danger figcaption { color:var(--danger); }
.eqd .card.hl { border:2px solid var(--accent); box-shadow:var(--glow); }
.eqd .card.hl figcaption { color:var(--accentInk); font-weight:700; }
.eqd .card.hlb { border-color:var(--accent); }
.eqd figcaption, .eqd .fc { min-height:24px; display:flex; align-items:center; justify-content:space-between; gap:8px;
  font-size:14px; font-weight:600; }
.eqd .sub { font-size:12px; font-weight:400; color:var(--muted); }
.eqd .small { font-size:12px; font-weight:500; }
.eqd .card img { width:100%; aspect-ratio:2/1; object-fit:cover; border-radius:6px; display:block; }
.eqd .pert { display:flex; align-items:center; gap:10px; }
.eqd .pert img { width:38%; }
.eqd .pert span { font-size:12px; color:var(--muted); line-height:1.35; }
.eqd .arrow { align-self:center; display:flex; flex-direction:column; align-items:center; font-family:'IBM Plex Mono',monospace;
  font-size:11px; color:var(--line); }
.eqd .arrow::after { content:''; width:2px; height:22px; background:currentColor; margin-top:4px; }
.eqd .arrow.danger { color:var(--danger); }
.eqd .fan { position:relative; }
.eqd .metric { display:flex; align-items:center; gap:10px; min-height:36px; }
.eqd .mval { display:flex; flex-direction:column; line-height:1.05; min-width:70px; }
.eqd .mlbl { font-size:11px; color:var(--muted); }
.eqd .mnum { font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:500; }
.eqd .mnum.accent, .eqd .mtime.accent { color:var(--accentInk); }
.eqd .mnum.danger { color:var(--danger); }
.eqd .bar { flex:1 1 auto; height:6px; border-radius:3px; overflow:hidden; background:var(--track); }
.eqd .fill { height:100%; background:var(--line); }
.eqd .fill.accent { background:var(--accent); }
.eqd .mtime { font-family:'IBM Plex Mono',monospace; font-size:13px; color:var(--muted); white-space:nowrap; }
.eqd .badge { font-size:11px; font-weight:700; border-radius:6px; padding:2px 7px; white-space:nowrap; }
.eqd .badge.on { background:var(--accent); color:var(--onAccent); }
.eqd .badge.off { background:var(--infoSoft); color:var(--infoInk); }
.eqd .badge.alarm { background:var(--danger); color:#fff; }
.eqd .cap { margin:0; font-size:12px; line-height:1.4; color:var(--muted); }
.eqd .verdict { display:flex; flex-direction:column; gap:14px; }
.eqd .vbox { display:flex; flex-wrap:wrap; align-items:center; gap:14px; border-radius:14px; padding:16px 18px; }
.eqd .vbox.good { background:var(--accentSoft); border:1px solid var(--accent); }
.eqd .vbox.good .vlab { color:var(--accentInk); }
.eqd .vbox.bad { background:var(--badBg); border:1px solid var(--badBd); }
.eqd .vbox.bad .vlab { color:var(--badTag); }
.eqd .vlab { font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:500; letter-spacing:.06em; }
.eqd .vbox p { flex:1 1 520px; margin:0; font-size:18px; line-height:1.45; }
.eqd .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.eqd .stat { border-radius:14px; padding:16px; display:flex; flex-direction:column; gap:4px; background:var(--panel);
  border:1px solid var(--border); box-shadow:var(--shadow); }
.eqd .sv { font-family:'Space Grotesk',sans-serif; font-size:40px; font-weight:700; line-height:1; }
.eqd .sl { font-size:14px; line-height:1.4; color:var(--muted); }
.eqd .note { margin:0; font-size:13px; color:var(--muted); }
.eqd .empty { padding:40px; text-align:center; color:var(--muted); border:1px dashed var(--border2); border-radius:16px; }
"""

HEADER = """
<div class="hdr">
  <span class="kick">ROBUST PERCEPTION THROUGH EQUIVARIANCE · FYP DEMO</span>
  <h1>An invisible change breaks the model.<br>Our defense notices, and repairs it.</h1>
  <p>Semantic segmentation with DRN-D-22 on Cityscapes. Pick an image, a model and an attack, then press Run and
  follow the image down the paths. <a href="?__theme=light">Light theme</a> · <a href="?__theme=dark">Dark theme</a></p>
</div>"""


def step_head(n, title, sub):
    return (f'<div class="sh"><span class="num" aria-hidden="true">{n}</span><div class="t"><b>{title}</b>'
            f'<span>{sub}</span></div></div>')


def build_ui(engine):
    import gradio as gr

    city = lambda i: IMAGE_PATHS[i].split("/")[2].capitalize()   # noqa: E731
    choices = [(f"val #{i} · {city(i)}" + (" · BPDA cached" if i in EXAMPLES[:50] else ""), i) for i in EXAMPLES]

    with gr.Blocks(title="Equivariance defense demo") as ui:
        gr.HTML(HEADER)
        idx_state = gr.State(DEFAULT_EXAMPLE)
        with gr.Row(elem_classes="setup", equal_height=False):
            with gr.Column(min_width=220):
                gr.HTML(step_head(1, "INPUT", "Pick an example, or upload your own image"))
                image = gr.Image(value=load_example(DEFAULT_EXAMPLE), type="numpy", sources=["upload"],
                                 show_label=False, height=170)
                example = gr.Dropdown(choices=choices, value=DEFAULT_EXAMPLE, show_label=False,
                                      info="Cityscapes validation (test images)")
            with gr.Column(min_width=220):
                gr.HTML(step_head(2, "MODEL", "Select the segmentation model"))
                model = gr.Radio(list(MODEL_LABELS), value="Clean-trained", show_label=False,
                                 info="Clean-trained: standard training · Adv-trained: trained against attacks")
            with gr.Column(min_width=220):
                gr.HTML(step_head(3, "ATTACK", "Choose an adversarial attack (ε = 4/255)"))
                atk = gr.Radio(list(ATTACK_LABELS), value="PGD-50", show_label=False,
                               info="SegPGD: made for segmentation · BPDA: knows the defense")
            with gr.Column(min_width=220):
                gr.HTML(step_head(4, "RUN", "Start the demo and see the results"))
                btn = gr.Button("▶  Run the story", elem_id="runbtn")
                gr.HTML('<p class="hint">Runs the attack and every defense live. The paper\'s defense takes ≈7 s; '
                        'BPDA-50 ≈15 s unless cached.</p>')
        out = gr.HTML('<div class="eqd"><div class="empty">Press Run to play the story.</div></div>')

        example.change(lambda i: (load_example(i), i), example, [image, idx_state])
        image.upload(lambda: None, None, idx_state)

        def go(img, idx, m, a, progress=gr.Progress()):
            return run(engine, img, idx, m, a, progress)

        btn.click(go, [image, idx_state, model, atk], out)
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="compute the self-check threshold (tune100) and exit")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public Gradio link (off by default)")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
        return
    engine = Engine()
    if not engine.calib:
        print("note: no demo/selfcheck_calib.json; the self-check shows a score but no limit "
              "(python -m demo.app --calibrate)")
    import gradio as gr
    build_ui(engine).queue(default_concurrency_limit=1).launch(
        server_port=args.port, share=args.share, css=CSS, head=FONTS, js=FORCE_DARK,
        theme=gr.themes.Base(primary_hue="orange"))


if __name__ == "__main__":
    main()
