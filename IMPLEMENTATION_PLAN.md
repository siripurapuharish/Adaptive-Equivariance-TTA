# Implementation Plan: Equivariance-Guided Adaptive Defense

Created 2026-10-26. Deadline: about 2026-11-24. **Compressed to a 25-day schedule on 2026-10-26** (day 1 = 2026-10-26, day 25 = 2026-11-19, leaving 5 days of slack before the deadline).
Background facts, configs and baselines are in `REPRODUCTION_REFERENCE.md`. The literature research behind the 25-day changes is in `reports/Equivariance defense improvement ideas.md`.

### Schedule at a glance (25 days)

| Phase | Days | Change from the 30-day plan |
|:--|:--:|:--|
| P0 Shared setup | 1–2 | Day 1 also profiles GPU memory and time (RTX 2000 Ada, 16 GB) |
| P1 Defense module | 3 | none |
| P2 Go/no-go | 4 | Pixel-level table over view types and error measures; entropy baseline |
| P3 Faster optimiser | 5–7 | Adds a check that the unrolled gradient ≈ the BPDA gradient |
| P4 Main idea | 8–12 | 6 → 5 days; adds the signal ablation and random-crop views |
| P5 View averaging | 13–14 | 1 day earlier |
| P6 Attack testing | 15–20 | Starts 1 day earlier; adds CosPGD, BPDA over intermediate steps, fixed-seed and transfer attacks, ε(p)-aware attack |
| P7 Wrap-up | 21–25 | 9 → 5 days, 1 buffer day |

**Cut order if we slip:** P5 view averaging first, then the reduced SEA attack, then extra attack variants. The core is the pixel-level test, per-pixel budget, optimiser, BPDA and the gate-aware attacks.

## Goal

**Novel contribution:** the first test-time defense for semantic segmentation that uses dense equivariance error to adapt the defense budget per pixel.

**Supporting parts:**
- a ~20× faster defense (`one_step`, P3). This comes from our own measurement that the published 20-step loop is effectively one sign step (`REPRODUCTION_REFERENCE.md` N1), not from a cited optimiser. Momentum and step decay (MI-FGSM / APGD) were tested in P3 and did not help (finding N7), so present them as a tested-and-rejected ablation, not as an integrated technique.
- prediction averaging over the equivariance views (standard segmentation test-time augmentation; cited)

**Evaluation:** PGD, SegPGD, BPDA (+EOT), a full-gradient attack, and a gate-aware adaptive attack. Every number is mean ± std over 3 seeds.

**Success criteria (measured on `test100`, mean ± std over 3 seeds; baselines from `REPRODUCTION_REFERENCE.md` §4b):**

| Metric | Baseline (test100, original defense) | Target |
|:--|:--:|:--:|
| Clean-model, clean images + defense (T8 analog) | 34.16 ± 0.87 | clearly higher, ideally > 45 |
| Clean-model, PGD + defense (T10 analog) | 27.34 ± 0.87 | no drop beyond ~1.5 mIoU |
| Adv-model, clean images + defense (T2 analog) | 43.47 ± 0.30 | higher |
| Adv-model, PGD + defense (T4 analog) | 27.51 ± 0.71 | no drop beyond ~1.5 mIoU |
| Defense time per image | ~7.1 s (clean) / ~8.3 s (adv), 20 steps | ≥ 3× faster (re-measure on an idle GPU before P3) |

**Significance:** with 3 seeds, the standard error of a difference between two means is ~0.6 mIoU, so changes smaller than **~1.5 mIoU (approximate)** are treated as noise. The 3-seed std estimates are themselves rough; use 5 seeds for the final headline numbers if time allows.

---

## Order of work (why this order)

```
P0 Shared infrastructure ──► P1 Clean defense module ──► P2 Go/no-go separation check
   (everything uses it)        (exposes error map)          (decides if the idea works)
                                     │                               │
                                     ▼                               ▼
                               P3 Faster optimiser ─────────► P4 MAIN IDEA (gate + per-pixel)
                               (makes P4–P6 3–10× cheaper)           │
                                                                     ▼
                                              P5 View averaging ──► P6 Adaptive evaluation ──► P7 Final runs, demo, write-up
```

- **P0 first:** every later experiment needs the same subset split, seeds, metrics and cached attacks. Doing this once saves hours later.
- **P1 before P2:** the separation check needs the per-pixel error map, which the new module produces.
- **P2 early:** it is the go/no-go test for the main idea. Running it before investing a week avoids wasted work.
- **P3 before P4–P6:** a 3-step defense instead of 20 makes every later experiment (especially the attacks in P6, which run the defense inside every attack step) much cheaper.

---

## P0. Shared infrastructure (Days 1–2)

**Why:** the six `evaluate_*.py` scripts each duplicate loading, alignment and metrics. New experiments need one consistent harness.

**Create `equi_lib/`** (reuse code from the existing scripts, don't rewrite logic):

| File | Contents |
|:--|:--|
| `data.py` | `split_indices(name)` / `load_batch(idx)` for `test100` (val indices i % 5 == 0), `tune100` (i % 5 == 2), or `full500`. Interleaved because the val list is sorted by city |
| `model.py` | `load_model(ckpt)` (copied from `evaluate_equi_defense.py:78`), `normalize()` with the info.json mean/std |
| `metrics.py` | `align_output()` (bilinear 344→340), confusion matrix, mIoU, per-class IoU |
| `seed.py` | `set_seed(s)` for random / numpy / torch / cuda |
| `attacks.py` | `pgd()` (copied from `evaluate_equi_defense.py:96`, already aligned). SegPGD, BPDA etc. are added in P6 |
| `cache.py` | Save and load PGD-50 adversarial deltas per (checkpoint, split, seed) as float16 `.pt` files under `results/improvements/cache/` |
| `runner.py` | `run(config) -> json`: loops the split, applies attack and defense, records mIoU and seconds per image, writes `results/improvements/<name>_seed<k>.json` |

**Attack cache:** standard (non-adaptive) PGD doesn't depend on the defense, so it's generated once per seed and reused by every defense variant. That cuts ~2.5 min per 100 images from each run. Adaptive attacks (P6) are never cached.

**Steps:**
1. Commit the current work (`REPRODUCTION_REFERENCE.md`, `.claude/`, this plan).
2. Build `equi_lib/`.
3. **Sanity test:** clean mIoU and PGD mIoU on `full500` with seed 42 must equal the existing results: 58.24 / 1.32 (clean model) and 51.34 / 25.26 (adv model), within 0.05.

**Done when:** the sanity test passes and one command reproduces a baseline number.

---

## P1. Clean defense module with error map (Day 3)

**Why:** the upstream function (`Equi_Set_Defense_good_inloop`) can't expose the per-pixel error or take a per-pixel budget, and it hard-codes the noisy optimiser.

**Create `equi_lib/defense.py`** with `EquiDefense(model, num_views, eps_v, steps, optimizer=..., eps_map=None)`:
- `views(x)`: the same transforms as upstream (cnt 0–3 resize with random aspect; 4–5 rotation ±15°; 6 ColorJitter+resize; 7 flip+resize). Returns each view's features mapped back to the base grid, plus validity masks (rotations have empty corners).
- `loss(x)`: identical to upstream (global L2-normalised flattened features, negative pairwise dot products), so the original defense reproduces exactly.
- **`error_map(x)` (new):**

  e_p = 1 − mean over views t of cos( f₀[:, :, p], f_t[:, :, p] )

  This is the per-pixel cosine across the 19 channels, at feature resolution 43×85, ignoring invalid (masked) pixels. Bilinearly upsample it to 340×680.
- `optimizer="original"`: the exact upstream loop (gradient normalised over the batch, plus N(0,1)·(T−1−i)/T noise, then sign, step 1.0, clamp).

**Equivalence test (`tests/test_defense_equivalence.py`):** with the same seed, `EquiDefense(optimizer="original")` and the upstream function must give the same `delta` on 2 images, matching to about 1e-6. Small mismatches can come from non-deterministic CUDA ops; if so, compare the final mIoU on 20 images instead (must be within 0.3).

**Done when:** the equivalence test passes for both models (9 views at 10/255; 7 views at 6/255).

---

## P2. Go/no-go: does equivariance error separate clean from attacked? (Day 4, ~1 GPU-hour)

Run on `tune100` for both models (clean images vs PGD-50 images, seed 0).

| Check | Measure | Go threshold |
|:--|:--|:--|
| Image level (for the gate) | AUROC of mean(e) for clean vs attacked | ≥ 0.90 |
| Pixel level (for per-pixel budget) | AUROC of e_p for predicting "this pixel's prediction changed under attack" | ≥ 0.70 |

**Pixel-level table (new, forward passes only, no prior paper reports it).** Repeat the pixel-level check for each view type (flip, rescale, random-offset crop, ColorJitter) crossed with each error measure (L2 on softmax, symmetric KL, label-flip rate, penultimate-feature cosine), on PGD-50, SegPGD and a low-ε attack. Add an entropy-only baseline. "Attacked pixel" means a pixel whose prediction the attack flipped, because a global L∞ attack perturbs every pixel. Pick the best (view, measure) pair on `tune100` for P4.

Also save histograms and 5 example error maps (clean / attacked side by side). They go straight into the slides.

**Decision:**
- Both pass → build gate + per-pixel (P4 as planned).
- Only image level passes → build the gate plus a softer per-image budget (ε_v scaled by the image score). This is still novel at test time for segmentation.
- Neither passes → stop. Present the negative result, and switch the main idea to P3 + P5 with stronger evaluation. Decide with your guide.

**Same night:** run the 3-seed baselines on `test100` (original defense, both models, clean and PGD). That's about 3 GPU-hours, run overnight. These numbers are the bar every later variant must beat. *(Started 2026-10-26 17:50 via `scratch/run_queue.py`: clean and adv baselines, 3 seeds each, plus the two Note B checks. Results go to `results/improvements/`.)*

**Also overnight (optional but useful):** the global ε_v trade-off sweep on the clean model, ε_v ∈ {0, 2, 4, 6}/255, 1 seed. It gives the curve the main idea must beat, not just move along.

---

## P3. Faster optimiser (Days 5–7)

**Added check (day 5):** the defense steps with `sign()`, whose derivative is zero almost everywhere, so exact autograd through the unrolled defense should reduce to the BPDA gradient. Measure the cosine similarity between the two gradients on a few images. A value near 1 means the planned "full-gradient" attack in P6 is BPDA in disguise, so P6 must use a smoothed surrogate instead (tanh(g/τ) or g/‖g‖∞, 2–5 unrolled steps).

**Why first:** P6 runs the defense inside every attack step, so a 3-step defense makes adaptive attacks about 6× cheaper. It also fixes the degenerate loop we measured (19 of 20 steps are noise).

Add options to `EquiDefense`:

| Variant | Steps | Noise | Update |
|:--|:--:|:--:|:--|
| `original` | 20 | yes | sign, step 1.0 (saturates to ±ε_v) |
| `one_step` | 1 | no | δ = −ε_v·sign(g) from a random start |
| `sign_decay` | 3 / 5 | no | sign step with step size cosine-decayed from 2ε_v to ε_v/4 |
| `momentum` (MI-FGSM style) | 3 / 5 | no | g_m = μ·g_m + g/‖g‖₁, μ = 0.9, sign step with decay |

**Protocol:** tune on `tune100` with 1 seed, choosing by robust mIoU (PGD) and then speed. Confirm the top 2 on `test100` with 3 seeds.

**Done when:** one variant matches the original's robust mIoU (within noise) at ≥ 3× speed, or beats it. That variant becomes the base for P4. If none matches, keep `original` and continue; the speed claim is then dropped.

**Outcome (done 2026-10-27):**
- tune100, seed 0, PGD+def (clean / adv model): original 25.45 / 27.86, one_step 25.96 / 27.74, momentum3 22.58 / 28.03, momentum5 22.22 / 27.12, sign_decay3 20.98 / 28.22, sign_decay5 21.36 / 27.44.
- **Deviation from the protocol above:** the strict rule (top 2 by robust mIoU averaged over both models) picks one_step (26.85) and momentum3 (25.31). We confirmed one_step and **sign_decay3** (24.60) instead, because sign_decay3 was the extreme case (worst on the clean model, best on the adv model), so it tests whether the "more optimisation hurts the clean model" effect is real. momentum3 was **not** run on test100. Its tune100 result points the same way (−2.9 on the clean model), but it is a one-seed number.
- test100, 3 seeds: one_step matches the original (see the speed table below) and is the P4 base. sign_decay3 confirms N7 (clean model PGD+def −5.3, clean+def +6.0).
- Saved evidence that the multi-step variants really reach a lower equivariance loss: `results/improvements/p3/loss_check_*.json` (`python -m equi_lib.p3_loss_check`).

### P3b. Speed optimisation (same days; do the profiling first)

**Step 1: profile** on 10 images, both models. Time each part per image:
- the base forward pass
- each view's forward and backward
- the parameter update
- the PGD attack

This confirms where the time goes before changing anything.

**Step 2: changes that shouldn't affect accuracy.** Apply them, then verify mIoU on `tune100` stays within the seed noise.

| Change | How | Expected gain |
|:--|:--|:--:|
| Fewer steps | the P3 optimiser (3 steps) | ~5–6× |
| Mixed precision | `torch.autocast("cuda", dtype=torch.float16)` around the model calls; keep δ, the loss sum, and the gradient normalisation in float32 | ~1.5–2× |
| Larger batch | raise from 2 to the largest that fits in 16 GB. The defense result doesn't depend on batch size: per-image losses are summed, and only the sign of the gradient is used | ~1.2–1.5× |
| Reuse gate views | the error-map pass (P4 gate) also serves as the first defense step's forward | saves 1 pass |
| cuDNN autotune | `torch.backends.cudnn.benchmark = True` (input size is fixed at 680×340) | small |

**Step 3: changes that might affect accuracy.** Keep each only if robust mIoU on `tune100` stays within noise.

- **Cheaper upscale views:** views cnt 0 and cnt 2 upscale to 2× height and width (4× pixels each). They are ~60% of compute with 9 views and ~70% with 7 views.
  - Try a scale of 1.5× (2.25× pixels), or replace one of the two 2× views with a 1.5× view.
  - Risk: the paper's Table 6 says resize views help robustness the most.

**Output:** a speed table, ready to drop into the slides. test100, mean ± std over seeds 0–2, batch 2, RTX 2000 Ada. Each cell is clean model / adv model. Defense time only (attack excluded).

| Config | sec/img (attacked) | sec/img (clean) | Clean+def mIoU | PGD+def mIoU |
|:--|:--:|:--:|:--:|:--:|
| Original (20 steps) | 7.17 / 8.32 | 7.20* / 8.33 | 34.16 ± 0.87 / 43.47 ± 0.30 | 27.34 ± 0.87 / 27.51 ± 0.71 |
| `one_step` (P3) | 0.35 / 0.41 | 0.35 / 0.43 | 34.86 ± 0.80 / 44.03 ± 0.42 | 26.84 ± 0.62 / 27.51 ± 0.56 |
| + AMP + batch | not run | | | |
| + cheaper views | not run | | | |
| + gate (P4) | | | | |

\* Seeds 1–2 only; seed 0's clean+def pass ran on a busy GPU (44 s/img).

**Status of P3b (2026-10-27):**
- Profiling: done (`results/improvements/p3/profile_*.json`). The two 2× views take 76% (clean model) / 63% (adv) of the view cost.
- Batch 4: rejected. The adv model needs 15.3 GB and slows to 26.6 s/img, because Windows spills GPU memory into system RAM instead of raising out-of-memory.
- AMP and cheaper views: not run. `one_step` already gives ~20× (target ≥ 3×), and half precision could flip gradient signs, which is all the defense uses. Revisit only if P6 attacks are too slow.
- Reuse gate views: moves to P4. cuDNN autotune: not applied.

**BPDA and adaptive attacks get the same speed-up automatically,** since they run the defense inside each attack step. Do **not** reduce attack strength (BPDA steps, EOT samples) to save time, because that would make the defense look stronger than it is.

---

## P4. Main idea: gate + per-pixel budget (Days 8–12)

**Updated 2026-10-27 after P2 and P3:**
- The error signal is the P2 winner **`down|l2_softmax`**: the L2 distance between the softmax of the base output and of a 0.5×-downscaled view mapped back to the base grid. It is *not* the P1 `error_map()` (logit cosine over the defense views), which fails on the adv model (pixel AUROC 0.47).
- The defense objective stays upstream's. The base optimiser is `one_step`.
- Gate per model, as decided in P2: the clean model gets gate + per-pixel (image AUROC 1.00). The adv model gets per-pixel only, or at most a soft per-image scale; never a hard gate (test100 image AUROC 0.872 < 0.90).

**Step 0: controls first (day 8, ~30 min of GPU time with `one_step`):**
- **Global ε_v sweep** with `one_step` on `tune100`, both models: ε_v ∈ {0, 2, 4, 6, 8, 10}/255 (the adv model is at 10/255), PGD+def and clean+def. This gives the trade-off curve a per-pixel budget must beat; just moving along that curve doesn't count.
- **Matched-budget control (in the ablation):** a global ε_v equal to the *mean* of our ε(p) map on the same inputs. If ours doesn't beat this, the gain comes from a smaller average budget, not from placing it per pixel.
- Move `down|l2_softmax` from `equi_lib/p2_extended.py` into `equi_lib/defense.py` as a reusable `consistency_map(x)`, and check it reproduces the P2 tune100 AUROCs (0.987 / 0.921).
- Compute the cut-offs a, b on `tune100` (clean and PGD-50 pixels, seed 0) and save them. The P2 JSONs store only AUROCs, not percentiles.

**Pre-registered decision rule (written 2026-10-27 before any P4 mIoU was computed).** Decision runs: `tune100`, seeds 0–2, both models, `equi_lib.adaptive` with β = 0.2 and 9×9 smoothing; numbers are 3-seed means.
- **Placement matters (the novel claim)** if `perpixel` beats **both** `perimage` (same per-image budget, no placement) and `shuffled` (same budget histogram, wrong places) by > 1.5 mIoU on PGD+def or clean+def, while losing ≤ 1.5 on the other.
- **Beats the global curve** if the `perpixel` point (clean+def, PGD+def) lies above the global ε_v sweep curve: at the sweep's interpolated clean+def, PGD+def is > 1.5 higher, or the reverse.
- **Both hold:** proceed with per-pixel; tune β ∈ {0, 0.2, 0.4} on `tune100`; run the test100 ablation.
- **Only the per-image budget beats the curve:** the plan's fallback (gate + per-image budget); the per-pixel claim is dropped.
- **Neither:** report the negative result; the contribution becomes the P3 findings plus the evaluation (discuss with the guide).
- The clean-model gate is judged separately (it is not novel): it must keep PGD+def within 1.5 of `perpixel` while raising clean+def.

**Algorithm (for each input x):**
1. `e = consistency_map(x)` (`down|l2_softmax`), computed once from the input, **detached** (stop-gradient), so the optimiser can't game it. It uses one extra no-grad forward of a 0.5× image (cheap).
2. **Gate (clean model only):** s = mean(e). If s < τ, return x unchanged.
3. **Budget map:** smooth e with a 9×9 average pool at feature resolution (avoids speckle), then

   ε(p) = ε_v · clip( (e_p − a) / (b − a), β, 1 )

   - a, b: the 50th percentile of clean-pixel e and the 90th percentile of attacked-pixel e (computed in step 0 on `tune100`)
   - β: a floor (default 0.2) so no attacked pixel gets zero budget
4. Run `one_step` with the per-pixel budget: start at a random corner ±ε(p), take one sign step, and clamp element-wise to `[−ε(p), ε(p)]`. This gives δ = −ε(p)·sign(g).

**Hyperparameters, chosen on `tune100` only:**
- τ (clean model): the value at which 95% of clean images pass (skip the defense)
- β ∈ {0, 0.2, 0.4}
- a / b percentiles from step 0

**Ablation table (`test100`, 3 seeds, both models):**

| Variant | Clean+def | PGD+def | sec/img |
|:--|:--:|:--:|:--:|
| No defense | | | |
| Original defense (P2 baseline) | | | |
| P3 optimiser, global ε_v | | | |
| Global ε_v = mean of our ε(p) (matched-budget control) | | | |
| + gate only (clean model) | | | |
| + per-pixel only | | | |
| + gate + per-pixel (**ours**) | | | |

**Signal ablation (needed to defend the word "equivariance-guided").** Run the per-pixel budget with the signal swapped and everything else fixed: uniform ε, an entropy-only map, a saliency/attention map (the Pei 2025 style), and the equivariance map. Optionally also a two-feature fusion (equivariance + entropy, fitted on `tune100`). Include random-offset crop views (Xiao et al. 2018) in the error map if P2 shows they help.

**Novelty wording:** claim only "first test-time defense for *segmentation* that uses the dense equivariance error to set a per-pixel budget". The gate alone is not novel (Mao et al. Appendix A.3.1 propose detect-then-defend), so present it as a speed and clean-accuracy device. Cite Pei 2025, SuperPure, SSNI, UAD-RS and pixel-reweighted adversarial training. Re-run a Scholar search close to submission.

**Done when:** the table is filled. The success criteria are at the top of this file.

**Fallback:** if per-pixel doesn't help, ship gate + per-image budget (decided in P2).

---

## P5. Prediction averaging over views (Days 13–14)

After the defense, predict on the restored image and its views, map each prediction back (resize back; un-flip; inverse-rotate with mask), and average the softmax probabilities over valid pixels.

- Use the resize and flip views; include rotations only if they help on `tune100`.
- Report as an extra row in the P4 table, applied to both the original defense and ours, so its contribution is separated from the main idea.

**Done when:** we know whether it adds mIoU on clean and attacked inputs. If it adds nothing, report that and drop it.

---

## P6. Adaptive evaluation (Days 15–20)

Attack the **final system** (P4 + P5) and the **original defense**, on `test100` (or 50 images for the expensive attacks).

| Attack | Source | Implementation note | Images |
|:--|:--|:--|:--:|
| PGD-50 | Madry 2018 | cached (P0) | 100 |
| SegPGD-50 | Gu et al. 2022 | L = (1−λ_t)·CE(correct pixels) + λ_t·CE(wrong pixels), λ_t = (t−1)/(2T) | 100 |
| BPDA-10 | Athalye 2018 | identity backward through the defense (reuse `evaluate_bpda_defense.py` logic) | 50 |
| BPDA-10 + EOT-4 | Athalye 2018; Croce 2022 | average the BPDA gradient over 4 defense runs (random views) | 50 |
| Full-gradient (smoothed) | Lee & Kim 2023 (surrogate idea) | sign replaced by a smooth surrogate (see the P3 gradient check), 2–5 unrolled steps, gradient checkpointing and AMP; 16 GB will not fit the full 20-step unroll | 20–50 |
| **Gate-aware** | adapted from upstream `PGD_attack_new_adaptive` | L = L_task − λ·mean(e(x+δ)), λ ∈ {1, 10, 100, 1000} (paper's strongest λ_e is 1000), to slip past the gate. Gradients flow through the error map for the attacker | 50 |
| **ε(p)-aware** | this project | the gate fires but the attacker keeps the local error low where the attack sits, so ε(p) stays small there | 50 |
| CosPGD-50 | Agnihotri et al. 2023 | a few lines on top of PGD | 100 |
| BPDA over intermediate iterates | Croce et al. 2022 | average the BPDA gradient over the defense's intermediate steps (how SOAP was broken) | 50 |
| Fixed-seed and transfer | Croce et al. 2022 | freeze the RNG and attack the now-deterministic defense; transfer unsuccessful perturbations from the undefended model | 50 |

- **Report worst-case robust mIoU** over all attacks for each defense, and report the gate's TPR/FPR under the gate-aware attack, not only AUROC on plain PGD.
- Report results **with and without EOT**: the injected randomness can inflate robustness, so a cleaner optimiser may look worse under EOT.
- Sanity checks: attack-steps and ε curves should reach near 0 at large ε; white-box should beat transfer; add a random-noise control.
- Optional if time remains: a reduced SEA (APGD over masked CE and JS losses, 100 iterations, via BPDA). No test-time segmentation defense has been evaluated with SEA.
- The gate-aware attack is essential: the main idea relies on e, so lecturers will ask whether an attacker can fake a low error.
- If it breaks the gate, report that honestly and show how much the per-pixel budget (without the gate) still holds.

**Cost:** BPDA-10 with a 3-step defense is ~4× cheaper than Task 6's 1.5 min/image, about 20 min per 50 images.

---

## P7. Final runs, demo, write-up (Days 21–25)

- **Day 21:** final configurations on `full500` with seed 42 (overnight) for both models, so they line up with the reproduction table.
- **Day 22, demo** (`demo/app.py`, Gradio):
  - choose an image and a model
  - attack it (PGD or SegPGD)
  - show the clean / attacked / original-defense / our-defense segmentation maps, the equivariance error map, and the budget map
  - show mIoU and time
- **Days 23–24, thesis:**
  - reproduction (`REPRODUCTION_REFERENCE.md`)
  - the two findings (one-step collapse; weaker under BPDA on the adversarially trained model, since Mao already report the BPDA drop for the clean model)
  - method, ablations, adaptive evaluation, limitations
- **Day 25:** slides and buffer.
- **Cut from scope** (mention as future work): restart selection by lowest loss (SOAP did this and was broken), diffusion priors, certification, patch attacks, new datasets or backbones.

---

## Experiment rules (every run)

- Tune only on `tune100`; report only on `test100` / `full500`.
- 3 seeds (0, 1, 2) and mean ± std. On 100 images, differences under ~1 mIoU are noise.
- Config and git commit hash are saved in every result JSON.
- Commit after each phase with its results under `results/improvements/`.
- Never overwrite the reproduction results in `results/*.json`.

## GPU time budget (RTX 2000 Ada, 100 images)

| Run | Approx. time |
|:--|:--|
| PGD-50 generation (cached once per seed) | 2.5 min |
| Original 20-step defense | 13 min |
| 3-step defense | ~2–3 min |
| BPDA-10 with 3-step defense, 50 images | ~20 min |
| P2 baselines (3 seeds × 2 models × clean/PGD) | ~3 h (overnight) |
| Final full500 runs | ~1–2 h per model (overnight) |

## Risks

| Risk | Sign | Response |
|:--|:--|:--|
| Error doesn't separate clean from attacked | P2 AUROC below threshold | Fallback plan in P2 |
| Faster optimiser loses robustness | P3 table | Keep `original`; drop the speed claim |
| Gate-aware attack breaks the gate | P6 | Report it; show per-pixel-only results; discuss as a limitation |
| Schedule slips | Behind by > 2 days at end of P4 (day 12) | Skip P5, run P6 on 50 images only |
| Full-gradient attack doesn't fit in 16 GB | OOM, or the P3 gradient check shows cosine ≈ 1 (it does: 1.000000) | Smoothed 2–5 step surrogate with checkpointing and AMP, on 20–50 images |
| Windows spills GPU memory into system RAM | No out-of-memory error; the run just becomes several times slower (P3: the unrolled check peaked at 21.9 GB on the 16 GB card; adv batch 4 ran at 26.6 s/img) | For P6, log `torch.cuda.max_memory_allocated()` and sec/img per run; treat a peak > 15 GB as a failed configuration |
