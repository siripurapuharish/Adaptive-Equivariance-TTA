# Reproduction Reference: *Robust Perception through Equivariance* (Mao et al., ICML 2023)

Validated on 2026-10-26 against three sources:
- **Paper:** arXiv 2212.06079, Sections 3.2 and 4.1–4.7, Tables 1, 2, 3 and 7.
- **Upstream code:** `Equi4Rob/` (mainly the `equi4robust.py` main block).
- **Our scripts and results:** `evaluate_*.py`, `results/*.json`.

This file replaces an earlier AI-generated gap checklist (now deleted). Several of its claims were wrong; Section 2 gives the verdict on each.

Labels used: **CONFIRMED** = shown by the paper, code or a measurement. **REFUTED** = contradicted by evidence. **OPEN** = plausible but untested.

---

## 1. Correct configuration (use exactly this)

| Setting | Adv-trained DRN-22-d (Table 1) | Clean DRN-22-d (Table 2) | Source |
|:--|:--|:--|:--|
| Input | 680×340, val split (500 images) | same | Paper §4.1, §4.5 |
| Attack ε / step / steps | 4/255, 1/255, 50 | same | code `run_test(...)`; paper captions |
| Defense bound ε_v | **2.5ε = 10/255** | **1.5ε = 6/255** | Paper §4.5 / §4.6; code lines 419, 491 |
| Defense steps T | 20 | 20 | Paper §4.4, §4.6 |
| Defense step size | code: 1.0 (=255/255) | same | ≡ paper's η = 2ε_v (see note A) |
| `num_view` (incl. identity) | **9**: identity, 4 resizes, 2 rotations (±15°), ColorJitter+resize, flip+resize | **7**: identity, 4 resizes, 2 rotations. No flip, no jitter | code `Equi_Set_Defense_good_inloop`; paper §4.4 describes only the 9-view set |
| Feature layer | `layer_ind=1` (the pre-upsampling 19-channel logits) | same | code |
| BPDA | 10 outer steps × 20 inner defense steps (paper §4.6). Upstream code uses **50** outer steps (comment "29.71") | 10 × 20 | paper vs code disagree |
| Adaptive attack (AA) λ_e | code default 1 | paper: **λ_e = 1000** is strongest | Paper §4.6 |
| Random-noise baseline | uniform noise, bound 2ε | — | code line 431 |

**Note A (step size).** Each defense step is `delta = clamp(delta − step·sign(g), −ε_v, ε_v)`. Any step ≥ 2ε_v always lands on ±ε_v, so the code's step of 1.0 and the paper's η = 2ε_v give the same result. CONFIRMED (algebra).

**Note B (clean-image runs in upstream).** Upstream runs the "clean + defense" rows with `step_num=0`. `PGD_attack_new` with 0 steps returns its random start, so the defended input is **x + U(−4/255, 4/255)**, not x. Our Task 2 and Task 8 scripts defend plain x. The effect is OPEN.

**Note C (Task 2 ε_v).** Paper Table 7 lists clean = 48.74 at ε_v = 10/255, which is our Task 2 setting. The upstream code comment puts "clean rev 48.74" at multiplier 2 (8/255), where Table 7 gives 48.84. Either value is fine. This is **not** the cause of the Task 2 gap.

**Note D (Task 8 config).** Upstream has no clean-model, clean-image defense call. Using 1.5ε with 7 views is our inference from paper §4.6 and the clean-model PGD call. It is well supported, but not stated explicitly anywhere.

---

## 2. Verdict on each finding in the old checklist

### Gap #1 Resolution mismatch: **mostly REFUTED**
- The old claim that the authors evaluated at full 1024×2048 is wrong. The paper says (§4.1, §4.5): *"we downsample the image from 2048×1024 to 680×340"*.
- The 340 → 344 output height is CONFIRMED (stride 8: ceil(340/8)·8 = 344). The old file's formula `ceil(h/8)*8/8` is wrong; the correct one is `out_h = ceil(h/8)*8`.
- Alignment is required. Upstream code cannot run on 680×340 without it: `NLLLoss` raises `RuntimeError: size mismatch` (verified in PyTorch 2.11), and `fast_hist` would fail on unequal lengths. How the authors handled this is unknown.
- It is **not** a systematic 1–2% deficit on all tasks. Same pipeline, clean model: 58.24% vs paper 58.29% (−0.05).
- On the adv model, a 680×336 crop gave 52.01% vs 51.34% (+0.67). That run also drops 4 rows of content, so it bounds the alignment effect but doesn't isolate it.
- **Action:** keep bilinear alignment before every loss and every mIoU computation.

### Gap #2 Single-GPU vs multi-GPU BatchNorm: **REFUTED**
- In `eval()` mode, BatchNorm uses the stored running stats, so the number of GPUs has no effect. Upstream also loads the model and calls `model.eval()`.
- Our loader reports 0 missing and 0 unexpected keys. Defense and attack never put the model in train mode.
- **OPEN alternative for the adv-model −1.89 clean gap:** the released checkpoint may not be the exact one used in the paper.
  - The authors' path was `.../city_adv_eps4_200e_s2_n3/last_200.pth.tar`; the released file's metadata says `best_prec1=33.91`.
  - Supporting evidence: the clean checkpoint matches the paper to 0.05 under the identical pipeline.
  - This can't be verified without the authors.

### Gap #3 "BPDA bug": **REFUTED**
- Using `X_pfy.grad` as the gradient for `delta` *is* BPDA: the identity backward approximation of Athalye et al. (2018).
- The paper describes exactly this (§3.2, §4.2): *"straight-through the gradient from our adapted image to the input image."* Both upstream `BPDA.py` and our `evaluate_bpda_defense.py` implement it correctly.
- The old "fix" (backpropagating through the 20-step purifier to `delta`) is a different attack: full-gradient / EOT. It is not BPDA, and it is not what the paper reports.
- The old evidence ("36.42% proves the bug") doesn't hold:
  - 36.42% is `model(x + δ_BPDA)` **without** purification.
  - δ was optimised against the purified pipeline, so it transfers imperfectly to the raw model.
  - The comparable no-defense number is Task 5 (25.25%), which equals PGD (25.26%) as expected. Paper Table 1 also lists 26.61 for both.
- **What remains OPEN:** Task 6 = 22.55% vs paper 29.83%, and our defended BPDA result is *below* undefended PGD.
  - Our config matches the paper text (10×20 steps, 2.5ε, 9 views).
  - Using 10 outer steps instead of the code's 50 should make the attack *weaker*, so it doesn't explain the gap.
  - Plausible reason: the defense is effectively one random-start gradient step (Section 3, N1), which an adaptive attacker can anticipate.
  - Paper Table 2 shows BPDA devastating the clean model (30.76 → 11.64), so a strong BPDA effect is not implausible.
- **Action:** report 22.55% as a valid measurement of a correct implementation and discuss it as a reproduction finding. Don't call it a bug.

### Gap #4 PGD loss misalignment: **REFUTED for our results; CONFIRMED for upstream functions**
- Our `evaluate_attack.py`, `evaluate_equi_defense.py` and BPDA scripts use their own PGD with bilinear alignment, so Tasks 3, 4, 5, 9 and 10 are **not** affected.
- The claim that it is "silently handled" is wrong: the unaligned loss raises an error.
- **Action:** before using any upstream attack for the remaining tasks, add alignment at these lines:
  - `attack_new_single.py:76, 153, 242, 1410` (PGD, MIM, AA, AA-inv)
  - `targeted_attack_new_single.py:74, 152, 239`
  - Houdini's loss in `houdini_attack.py`

### Gap #5 Unseeded defense: **PARTLY REFUTED**
- Our scripts already seed `random`, `numpy` and `torch` (seed 42). Only upstream is unseeded.
- The randomness also comes from the torch RNG (random start, gradient noise, ColorJitter), not only Python `random`.
- Seeding still doesn't give bit-exact runs: `grid_sample` (rotations) and bilinear-upsample backward are non-deterministic on CUDA, and `cudnn.deterministic` doesn't cover them.
- **Measured spread** (Task 10 setting: clean model, PGD-50 + equi, 6/255, 7 views, first 50 images). CONFIRMED:

  | Seed | 0 | 1 | 2 | 42 | **Mean ± std** |
  |:--|:--:|:--:|:--:|:--:|:--:|
  | Defended mIoU | 29.87 | 30.68 | 30.75 | 31.07 | **30.59 ± 0.51** (range 1.20) |
  | PGD, no defense | 1.03 | 1.01 | 1.02 | 1.05 | 1.03 ± 0.02 |

  - The defense is responsible for nearly all the variance; the attack is stable.
  - The old "±1–3%" estimate was too high. The real spread is about ±0.5 (1.2 range) on 50 images. It should be smaller on all 500 images, but that is untested.
- **Action:**
  - Fix the seed, and report improvements as mean ± std over ≥ 3 seeds.
  - On a 50-image subset, treat any single-seed difference below ~1.2 mIoU as noise.
  - Compare against these seed-matched baselines, not the single 500-image number.

### Gap #6 Different hyperparameters per model: **CONFIRMED, but documented**
- The old claim that the paper doesn't mention them is wrong: paper §4.5 gives 2.5ε, and §4.6 gives 1.5ε, T = 20 and η = 2ε_v. The upstream main block lists them too.
- 7 vs 9 views is code-only; the paper describes only the 8-transform (9-view) set.
- The 7-view set excludes **both** flip and ColorJitter, not only jitter.
- The old claim that "ColorJitter causes degradation" is unverified.

### Old "Implementation Checklist": corrections
- Item 7 is wrong for BPDA: the gradient w.r.t. the purified image is correct.
- Item 9 is wrong: `BPDA.py`'s gradient flow is correct. Its real problem is that importing it crashes, because of a missing rotation checkpoint (N4).
- Item 5 doesn't apply (BatchNorm is not an issue in eval mode).
- Items 1, 2, 3, 6, 8 and 10 remain valid.

---

## 3. New findings (not in the old checklist)

**N1. The 20-step defense is effectively one gradient step. CONFIRMED by measurement.**
- Each step adds Gaussian noise of std (T−1−i)/T to a gradient normalised over the *whole batch tensor*, then takes its sign.
- Normalised gradient entries are ~1e-4 (median) and ≤ 0.07 (max), while noise std is ≥ 0.05 until the last step.
- Measured agreement between the update's sign and the gradient's sign (adv model and clean model, 2 images): steps 0–18 are **0.500–0.503** (chance); step 19 is **1.000**.
- So the output is −ε_v·sign(∇L_equi) evaluated at a random ±ε_v corner. Steps 0–18 only randomise the starting point.
- Consequences:
  - (a) For faithful reproduction, keep it exactly as is.
  - (b) For your improvement, this is the biggest lever: real multi-step optimisation, a proper noise scale, step decay, or early stopping. Expect a large speed-up too.
  - (c) Batch size doesn't change the defense result (per-sample losses are summed, and the sign is unaffected).
- Probe script: `scratch/probe_defense.py` (run from the repo root). It patches a recorder in before `gradient.sign_()`.

**N2. `equi4robust.py` and `BPDA.py` can't be imported as shipped.** At import time, `BPDA.py` builds `RotReversal('data/ckpts/advpretrained_ssl_rot_19.pth')`, and that checkpoint isn't released. This is why our stand-alone scripts exist. For the same reason, the **Rotation baseline can't be reproduced**.

**N3. Upstream evaluates 498 or 496 images, not 500.** The val loader uses `drop_last=True` with batch 3, 8 or 16. The effect is negligible; mention it only for completeness.

**N4. The authors' own code comment says the clean-model PGD + defense run "reproduce[s] 32.9 mIoU"**, while the paper reports 30.76 (ours: 29.38). Their own runs varied by about 2 points, so treat ±1–2 as within reproduction noise.

**N5. Summary-file labels are wrong.** The Task 2, 4 and 6 summaries say "8 spatial views", but those runs used `num_view=9` including ColorJitter. Use "9 views (identity + 8 transforms incl. ColorJitter)" in the thesis.

**N6. Task 6's "BPDA (No Defense) 36.42%"** is a transfer number (δ optimised against the defended model, evaluated without the defense). Don't present it as the BPDA no-defense row; that row is Task 5 (25.25%).

---

## 4. Current results with validated status

| Task | Setting | Paper | Ours | Status |
|:--|:--|:--:|:--:|:--|
| T1 | Adv, clean | 53.23 | 51.34 | Config correct. Gap OPEN (checkpoint / alignment) |
| T2 | Adv, clean + equi (10/255, 9 views) | 48.74 | 44.06 | Config matches paper. Missing the Note B noise. Gap OPEN |
| T3 | Adv, PGD-50 | 26.61 | 25.26 | Valid |
| T4 | Adv, PGD-50 + equi | 29.65 | 28.48 | Valid (+3.22 recovery vs paper +3.04) |
| T5 | Adv, BPDA, no defense (≡ PGD) | 26.61 | 25.25 | Valid |
| T6 | Adv, BPDA-10 + equi | 29.83 | 22.55 | Implementation correct. Gap OPEN, report honestly |
| T7 | Clean, clean | 58.29 | 58.24 | Valid |
| T8 | Clean, clean + equi (6/255, 7 views) | 37.95 | 35.66 | Config inferred (Note D). Missing the Note B noise |
| T9 | Clean, PGD-50 | 1.31 | 1.32 | Valid |
| T10 | Clean, PGD-50 + equi (6/255, 7 views) | 30.76 | 29.38 | Valid. Authors' code got 32.9 (N4) |

---

## 5. Remaining Cityscapes rows and what you need for them

| Paper row | Upstream function | Must do first |
|:--|:--|:--|
| MIM (T1, T2) | `MIM_attack_new` | Align loss (line 153) |
| AA (T1, T2) | `PGD_attack_new_adaptive` | Align loss (line 242). Use λ_e = 1000 for the clean model (paper §4.6) |
| Houdini (T1, T2) | `Houdini_attack_new` | Align before the Houdini loss |
| BPDA, clean model (T2: 11.64) | reuse `evaluate_bpda_defense.py` | Use 1.5ε, 7 views, 10 BPDA steps |
| Invariance baseline | `Inv_Set_Defense_good_inloop` | Same ε_v / views as equi |
| Random baseline | code: uniform noise ±2ε | — |
| Rotation baseline | `RotReversal` | **Not possible**: checkpoint not released (N2) |
| Targeted (Table 3) | `targeted_attack_new_single.py` | Align losses (lines 74, 152, 239) |
| Equivariance score (Table 5) | `measure_equi.Measure_Equi_Set_Defense_good_inloop` | — |
| ε_v trade-off (Table 7) | BPDA + equi, ε_v ∈ {0, 1, 2, 4, 6, 8, 10}/255 | Expensive. Use a subset |
| IFGSM (T1) | none found in repo | Would need to be written |

Runtime guide (RTX 2000 Ada, 500 images):
- clean inference: ~15 s
- PGD-50: ~13 min
- PGD + equi: ~80 min
- BPDA-10 + equi: ~13 h (~1.5 min per image)

---

## 6. Research gaps for the FYP improvement (novelty-checked 2026-10-26)

Novelty was checked by web search only. Before committing, check Google Scholar's "Cited by" for arXiv 2212.06079.

| # | Gap | Novelty evidence | Builds on | Cost |
|:--|:--|:--|:--|:--|
| G1 | **Independent adaptive evaluation on segmentation:** SegPGD / CosPGD, BPDA+EOT, and full gradient through the effectively-1-step defense | No independent re-evaluation of this paper found. Croce et al. (ICML 2022) broke an earlier Mao et al. test-time defense and recommend BPDA+EOT, which the paper doesn't use | T6 (22.55 < undefended 25.26 contradicts the paper's BPDA claim); N1 | 50–100 images per attack |
| G2 | **Spatially adaptive defense budget:** a per-pixel ε_v map driven by the dense equivariance error, so clean regions are left almost untouched | Found only whole-input detect-then-purify (TTP 2025, audio 2026). Nothing per-pixel for segmentation | T8 clean drop 58.24 → 35.66; paper Table 5 shows the score separates clean from attacked | Moderate. Must also be tested under G1 |
| G3 | **Fix the degenerate optimiser:** remove the noise and use a real multi-step schedule, or use 1–3 steps | Partly anticipated: TPAP (CVPR 2024) shows one FGSM purification step works for classification. New part: this published defense secretly *is* one step | N1, T4, T10 | Easy, ~15 min per setting on 100 images |

**Suggested framing:** reproduce → show the defense is weaker and less efficient than claimed (G1 + G3) → propose the pixel-adaptive version (G2). Evaluate every result with ≥ 3 seeds (see Gap #5 numbers).

---

## 7. Pre-experiment checklist (corrected)

```
[ ] Input 680x340; logits bilinearly aligned 344->340 before EVERY loss and before mIoU
[ ] Any upstream attack function used? -> alignment patched in (see Gap #4 lines)
[ ] Config matches Section 1 for the model type (eps_v, num_view, T=20, step 1.0)
[ ] Clean+defense rows: decide whether to add U(-eps,eps) noise like upstream (Note B), and state it
[ ] Seed fixed; improvement claims use >=3 seeds (mean ± std)
[ ] model.eval(); params requires_grad=False; strict load with 'module.' stripped
[ ] BPDA = identity backward through the purifier (grad w.r.t. purified image) - this is correct
[ ] Compare against OUR baselines (Section 4) on the same images and seeds
[ ] Summary text states the real view count (9 incl. ColorJitter / 7 spatial-only)
```
