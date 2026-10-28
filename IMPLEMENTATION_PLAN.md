# Implementation Plan: Equivariance-Guided Adaptive Defense

Created 2026-10-26. Deadline: about 2026-11-24. **Compressed to a 25-day schedule on 2026-10-26** (day 1 = 2026-10-26, day 25 = 2026-11-19, leaving 5 days of slack before the deadline).
Background facts, configs and baselines are in `REPRODUCTION_REFERENCE.md`. The literature research behind the 25-day changes is in `reports/Equivariance defense improvement ideas.md`.

### Schedule at a glance (re-planned 2026-10-28, after P6b)

P0–P6b finished by day 3 (2026-10-28). Under a strong adaptive attack the defense family (ours and the original) adds about 0 mIoU on the adv-trained model and about 2.7 on the clean model (P6b "Findings"). The remaining days go to the open question this raises (P5b), the optional P5, the final runs and the write-up. The leads from the Consensus literature summary (`reports/plan_additions_consensus_2026-10-28.md`, citations unverified) are folded in as A1 (P6b), P5b and the P7 future-work list.

| Day | Date | Phase | Work |
|:--:|:--|:--|:--|
| 1–3 | Oct 26–28 | P0–P6b | **Done** |
| 3 | Oct 28 | E1 | View-set code (flip swap, subsets) and per-image confusions; **overnight:** E1 runs on `tune100` |
| 4 | Oct 29 | E1, E2 | E1 rules (fixed set, per-image oracle). E2 = P5 view averaging: code and `tune100` runs |
| 5 | Oct 30 | E3 | P5b B1 self-check (held-out views) as a reliability output |
| 6 | Oct 31 | E4 / E1b | M5 if E3 passes; a label-free per-image view rule if the E1 oracle passes; D1 / A1 optional |
| 7 | Nov 1 | — | Exploration buffer; show the guide the results so far |
| 8–9 | Nov 2–3 | lock v2 | Lock final config v2 from the adopted items; `test100` 3 seeds and the BPDA-50 re-check on them |
| 10–11 | Nov 4–5 | P7 | `full500` runs of the final configs and the original (overnight) |
| 12 | Nov 6 | P7 | Gradio demo |
| 13–19 | Nov 7–13 | P7 | Thesis |
| 20–21 | Nov 14–15 | P7 | Slides |
| 22–25 | Nov 16–19 | — | Buffer (deadline Nov 24) |

**Cut order if we slip (2026-10-28):** E4 (M5), then E1b, then A1, then D1. **Never cut:** E1 (it is the guide's G1), E3 / B1, the v2 confirmation, the `full500` runs, the negative results. The `full500` runs are done last, on the final config (user's decision, 2026-10-28).

### Earlier schedule (re-planned 2026-10-27, after P4; kept for the record)

P0–P4 finished on day 2 (2026-10-27), well ahead of the 25-day plan. The per-pixel placement claim failed its pre-registered test (P4 outcome below), so the remaining days were re-planned around what worked: **image-level adaptation driven by the equivariance signal**. The re-plan also folds in the four improvements proposed in the mid-review slides (`Harish_midreview.pdf`); see the section "Mid-review improvements (M1–M4)" after P4.

| Day | Date | Phase | Work |
|:--:|:--|:--|:--|
| 1–2 | Oct 26–27 | P0–P4 | **Done** |
| 3 | Oct 28 | M1, M3, M2b | Code the reduced view sets (M1), the weighted defense loss (M3) and the entropy gate baseline (M2b); smoke test. **Overnight:** all three on `tune100`, 3 seeds |
| 4–5 | Oct 29–30 | P6a | **Gate-aware adaptive attack** on the clean model (`tune100`, 50 images). Day 4 morning: apply the pre-registered rules to M1 / M3 / M2b |
| 6 | Oct 31 | P4c | Lock the final config (budget floor if P6a breaks the gate; M1 / M3 if they passed). Show the reframing to the guide. **Overnight:** `test100` confirmation, 3 seeds |
| 7 | Nov 1 | P4c | Fill the final ablation table; retuned ε_v for the adv model |
| 8–12 | Nov 2–6 | P6b | The remaining attacks on the final system and on the original defense |
| 13–14 | Nov 7–8 | P5 | View averaging (optional) or buffer |
| 15–16 | Nov 9–10 | P7 | `full500` runs (overnight) |
| 17 | Nov 11 | P7 | Gradio demo |
| 18–22 | Nov 12–16 | P7 | Thesis |
| 23–25 | Nov 17–19 | P7 | Slides and buffer (deadline Nov 24) |

**Cut order if we slip:** P5, then M3, then the extra attack variants (CosPGD, transfer, reduced SEA), then the smoothed full-gradient attack. **Never cut:** P6a (gate-aware attack), BPDA (+EOT), the `test100` 3-seed confirmation, or the negative results.

## Goal

**Contribution (reframed 2026-10-27 after P4):** an **equivariance-guided adaptive test-time defense** for semantic segmentation. A cheap dense equivariance-consistency measure (`down|l2_softmax`, picked from 32 candidates in P2) decides **whether** to defend an image (gate) and **how strongly** (per-image budget). On the clean model it raises clean accuracy from 32 to 52 mIoU (the undefended level) at the same PGD robustness, at about 20× lower cost than the published defense (tune100; to be confirmed on test100 and under adaptive attack).

The old claim ("first defense with a per-pixel budget") is **dropped**: its pre-registered placement test failed on both models. The claim is now about choosing and evaluating the signal and the per-image budget, not about the gate itself (Mao et al. A.3.1 suggest detect-then-defend but never build or evaluate it). The strongest support is the adaptive-attack evaluation (P6).

**Supporting parts:**
- a ~20× faster defense (`one_step`, P3). This comes from our own measurement that the published 20-step loop is effectively one sign step (`REPRODUCTION_REFERENCE.md` N1), not from a cited optimiser. Momentum and step decay (MI-FGSM / APGD) were tested in P3 and did not help (finding N7), so present them as a tested-and-rejected ablation, not as an integrated technique.
- the mid-review improvements M1–M4, each with a result, including the negative ones (see the section after P4)
- prediction averaging over the equivariance views (P5, optional; standard segmentation test-time augmentation; cited)

**Findings for the thesis, positive and negative:**
1. The 20-step defense collapses to one effective step (N1). One step matches it at ~20× speed (P3).
2. More optimisation hurts the clean model (N7).
3. The defense is weaker under BPDA than reported on the adv-trained model.
4. The exact unrolled gradient equals BPDA (cosine 1.000).
5. Image-level adaptation gives a large gain on the clean model. Per-pixel placement doesn't help on either model.
6. Confidence is the wrong signal: attacked images are *more* confident.
7. Deep features detect attacks worse than softmax outputs.
8. The paper's ε_v = 2.5ε is not optimal with `one_step` on the adv model.

**Evaluation:** PGD, SegPGD, BPDA (+EOT), a smoothed full-gradient attack, and a gate-aware adaptive attack. Every number is mean ± std over 3 seeds.

**Success criteria (measured on `test100`, mean ± std over 3 seeds; baselines from `REPRODUCTION_REFERENCE.md` §4b):**

| Metric | Baseline (test100, original defense) | Target |
|:--|:--:|:--:|
| Clean-model, clean images + defense (T8 analog) | 34.16 ± 0.87 | clearly higher, ideally > 45 |
| Clean-model, PGD + defense (T10 analog) | 27.34 ± 0.87 | no drop beyond ~1.5 mIoU |
| Adv-model, clean images + defense (T2 analog) | 43.47 ± 0.30 | higher |
| Adv-model, PGD + defense (T4 analog) | 27.51 ± 0.71 | no drop beyond ~1.5 mIoU |
| Defense time per image | ~7.1 s (clean) / ~8.3 s (adv), 20 steps | ≥ 3× faster (re-measure on an idle GPU before P3) |
| **Worst case over the P6 attacks** (added 2026-10-27) | original defense's worst case, measured in P6 | no worse than it by more than ~1.5 mIoU, on both models |

The last row decides the thesis claim. A defense that wins on plain PGD but loses to an attacker who knows about the gate is not an improvement.

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

## P4. Main idea: gate + per-pixel budget (done on day 2)

*Kept as the record of what was planned and pre-registered. The outcome at the end of this section changed the plan: see "Mid-review improvements", P4c and P6a below.*

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

**Outcome of the decision runs (2026-10-27; tune100, seeds 0-2, beta 0.2, 9x9 smoothing; `python -m equi_lib.p4_analyze`, `results/improvements/p4/decision_tune100.json`):**

| Model | Variant | clean+def | PGD+def | mean eps frac (clean / PGD) |
|:--|:--|:--:|:--:|:--:|
| Adv | global 10/255 (preset) | 40.90 ± 0.18 | 27.80 ± 0.23 | 1 / 1 |
| Adv | global 6/255 (best of sweep on PGD) | 45.49 ± 0.41 | 28.19 ± 0.22 | 1 / 1 |
| Adv | perpixel | 47.53 ± 0.09 | 28.43 ± 0.21 | 0.22 / 0.31 |
| Adv | perimage | 47.78 ± 0.10 | 27.68 ± 0.14 | 0.22 / 0.31 |
| Adv | shuffled | 47.83 ± 0.13 | 27.14 ± 0.15 | 0.22 / 0.31 |
| Adv | perpixel, entropy signal | 46.29 ± 0.23 | 28.14 ± 0.18 | 0.31 / 0.34 |
| Clean | undefended | 52.14 | 1.15 | 0 / 0 |
| Clean | global 6/255 (preset, best of sweep on PGD) | 32.04 ± 0.17 | 26.31 ± 0.37 | 1 / 1 |
| Clean | perpixel | 49.41 ± 0.45 | 22.77 ± 0.35 | 0.22 / 0.78 |
| Clean | perimage | 49.97 ± 0.33 | 26.05 ± 0.52 | 0.22 / 0.78 |
| Clean | shuffled | 50.42 ± 0.11 | 19.06 ± 0.20 | 0.22 / 0.78 |
| Clean | gate | 49.37 ± 0.20 | 26.33 ± 0.55 | 0.05 / 1.00 |
| Clean | gate+perpixel | 51.90 ± 0.23 | 22.95 ± 0.21 | 0.01 / 0.78 |
| Clean | gate+perimage (fallback, run after the verdict) | 51.97 ± 0.22 | 26.12 ± 0.41 | 0.01 / 0.78 |
| Clean | perpixel, entropy signal | 34.75 ± 0.43 | 2.61 ± 0.09 | 0.57 / 0.28 |

- **Placement test: FAILS on both models.** Adv: perpixel − perimage = −0.25 clean+def / +0.75 PGD+def; perpixel − shuffled = −0.30 / +1.29 (paired over seeds; consistent sign, std ≤ 0.10, but below the 1.5 threshold). Clean: perpixel loses 3.28 PGD+def to perimage (it beats shuffled by 3.71). **The per-pixel claim is dropped**, as pre-registered.
- **Global-curve test.** Adv: nothing beats the curve by > 1.5 (perpixel +1.07, perimage +0.58). Clean: every image-adaptive variant beats it by a wide margin (perimage +23.1 PGD+def at equal clean+def; gate+perimage +24.8; gate +22.8). The gain comes from giving clean images a small budget and attacked images a large one, i.e. image-level adaptation.
- **Branch taken:** clean model → the plan's fallback (gate + per-image budget): gate+perimage keeps clean+def within 0.2 of the undefended model at the same PGD+def as the best global setting. Adv model → negative result for adaptive budgets (discuss with the guide). Side finding: on the adv model the preset 10/255 is not the best budget for `one_step`; 6/255 gives +4.6 clean+def and +0.4 PGD+def on tune100.
- **Caveats.** All numbers are against non-adaptive PGD-50. The gate and the per-image budget are driven by a signal the attacker can also lower, so the P6 gate-aware attack decides whether this survives. The gate is not novel (Mao et al. A.3.1 detect-then-defend). The result is tune100 only; test100 confirmation is pending.

---

## Mid-review improvements (M1–M4), added 2026-10-27

The mid-review slides (`Harish_midreview.pdf`) proposed four improvements. Their feasibility and novelty were checked on 2026-10-27, and each now has a place in the plan, with evidence from P2–P4. The thesis presents them as **"mid-review proposal → test → outcome"**, so the change of direction is visible and backed by data.

| Slide gap | Proposal | Status | Evidence / plan |
|:--|:--|:--|:--|
| G1 | Adaptive transformation selection | **Partly done; M1 below** | P2 compared 6 view types × 5 measures and picked `down` for the signal. M1 applies the same data-driven selection to the *defense* views |
| G2 | Adaptive defense strength from prediction confidence | **Done; the signal was changed** | Confidence fails. Attacked images are *more* confident (clean-model entropy image AUROC 0.018), and the entropy-guided budget collapses PGD+def to 2.61. The equivariance signal works (gate + per-image: 51.97 / 26.12). M2b adds a fair entropy-gate baseline |
| G3 | Confidence-weighted dense equivariance loss | **M3 below** (one cheap test) | The slide weights by σ(x+δ), which the optimiser can push up. Test the detached equivariance error as the weight, plus the slide's σ version, against uniform |
| G4 | Multi-scale feature equivariance | **Rejected as a signal (P2); defense loss is future work** | Feature-level measures separate attacks much worse than softmax L2 (adv-model pixel AUROC 0.39–0.61 vs 0.92) |
| Future work | Fewer optimisation steps | **Done (P3)** | `one_step`: same robustness, ~20× faster |
| Future work | ImageNet, VOC, COCO | **Out of scope** | No time to train or validate new checkpoints. Listed as future work |

**Pre-registered decision rules for M1–M3** (written 2026-10-27, before any run). Common setup:
- `tune100`, seeds 0–2, both models, PGD-50 cached.
- Each model runs in its current best pipeline: clean model = gate+perimage; adv model = global `one_step`.
- Numbers are 3-seed means. A change "passes" by the rule given for each item.

### M1. Data-driven defense view selection (G1)

**Why:** the two 2× upscale views (cnt 0 and 2) take 63–76% of the defense time (P3b profile) and are the weakest views in P2 (`up|l2_softmax` adv pixel AUROC 0.70 vs 0.92 for `down`). Dropping them is the cheapest speed-up left.

| Variant | Clean model (now 7 views) | Adv model (now 9 views) |
|:--|:--|:--|
| `full` | 2×, 0.3×, 2×, 0.5×, rot, rot | + ColorJitter, flip |
| `no_up` | 0.3×, 0.5×, rot, rot (5 views) | 0.3×, 0.5×, rot, rot, jitter, flip (7 views) |
| `up1.5` | both 2× views replaced by 1.5× | same |

- **Code:** add a `view_set` option to `EquiDefense`. The default must reproduce the current views exactly.
- **Rule:** keep the fastest variant whose PGD+def **and** clean+def are both within 1.5 of `full`, on both models. If none qualifies, keep `full` and report the speed–robustness trade-off.
- **Must also hold in P6:** fewer views could make BPDA easier. If a reduced set is adopted, run BPDA-10 on it as well.
- **Optional 4th variant:** `no_up` + random-offset crops (Xiao et al. 2018). Random views are harder for an attacker to predict than a fixed set.
- **Novelty:** modest; present it as an engineering and analysis result, not as a contribution.
- **Per-input learned selection, as on the slide:** rejected in the feasibility check. It needs training, and it is one more decision an attacker can steer. Learned per-input augmentation exists for corruptions (Kim et al., NeurIPS 2020), not for adversarial defense. Future work.

### M2b. Entropy-gate baseline (G2)

**Why:** the claim says *equivariance*-guided, so the image-level gate must be compared with the obvious confidence-based gate. Raw entropy points the wrong way, but its **inverse** (low entropy → defend) has clean-model image AUROC 0.982. That makes it a strong baseline, not a weak one.

- **Run:** gate+perimage with the image score = −mean entropy. τ is set so 95% of clean images pass, and the per-image budget is calibrated the same way as for `equi`.
- **Rule:**
  - If the equivariance gate beats the entropy gate by > 1.5 on either metric, the "equivariance-guided" wording stands on plain PGD.
  - If they tie, the wording has to rest on P6a. Both gates get attacked there, and the claim holds only if the equivariance gate is clearly harder to evade.
  - If entropy wins on both, rename the contribution "consistency-gated adaptive defense" and report both signals.
- Entropy is equally easy for an attacker to target (raise the entropy), so include the entropy gate in P6a.
- **Novelty (from the feasibility check):** low for "per-input defense strength" in general (SSNI, DGAP; Mao et al. A.3.1 detect-then-defend). Cite them. What is ours: the measured signal choice, the per-image budget for segmentation, and the adaptive evaluation.

### M3. Weighted dense equivariance loss (G3)

**Why:** this tests the slide's idea directly. The defense loss upweights pixels that look attacked, so the single sign step spends its effort there. After P4 the expected effect is small, since budget placement didn't matter. One overnight run settles it either way.

- **Weights:** per-pixel w_p ≥ 0 at feature resolution, normalised to mean 1 per image, **detached**. Each view's features (mapped back to the base grid) and f₀ are multiplied by √w_p before the upstream global L2 normalisation and dot product. With w ≡ 1 this reproduces the current loss exactly.

| Variant | w_p |
|:--|:--|
| `uniform` | 1 (current) |
| `equi` | smoothed `down|l2_softmax` map (the same map as the gate) |
| `conf` (slide version) | max softmax of f₀, detached |
| `inv_conf` | 1 − max softmax of f₀, detached. The feasibility check asked for both directions: upweighting confident pixels spends the effort where the image is already fine, while the attack sits in the uncertain ones |

- **Rule:** keep `equi` if it beats `uniform` by > 1.5 on PGD+def while losing ≤ 1.5 on clean+def, on at least one model. Otherwise report it as a negative result, next to the P4 placement test (same conclusion: *where* the defense acts matters less than *how much*).
- **Novelty:** there are pixel-weighted *training* losses (pixel-reweighted adversarial training, 2024). A test-time defense loss weighted by the dense equivariance error is, as far as we found, new for segmentation. Claim it only if it passes.

### Outcome of M1 / M2b / M3 (2026-10-27; tune100, seeds 0–2; `python -m equi_lib.m_analyze`, `results/improvements/mid/m_decision_tune100.json`)

Δ = paired difference vs the same-code base run (clean+def / PGD+def). Base rerun reproduced P4 (clean model 51.97 / 26.11; adv global 6/255 45.49 / 28.19).

| Model | Variant | clean+def | PGD+def | Δ | s/img (PGD+def) |
|:--|:--|:--:|:--:|:--:|:--:|
| Clean | base (gate+perimage, full views) | 51.97 ± 0.22 | 26.11 ± 0.40 | | 0.36 |
| Clean | M1 `no_up` | 51.39 ± 0.47 | 24.00 ± 0.39 | −0.58 / **−2.11** | 0.13 |
| Clean | M1 `up1.5` | 51.96 ± 0.06 | 25.38 ± 0.46 | −0.01 / −0.74 | 0.25 |
| Clean | M3 `equi` weight | 51.61 ± 0.19 | 25.29 ± 0.67 | −0.36 / −0.83 | 0.40 |
| Clean | M3 `conf` weight | 51.98 ± 0.19 | 26.11 ± 0.43 | +0.01 / −0.00 | 0.40 |
| Clean | M3 `inv_conf` weight | 51.53 ± 0.33 | 20.03 ± 0.08 | −0.44 / **−6.08** | 0.40 |
| Clean | M2b entropy gate + perimage | 52.11 ± 0.01 | 25.26 ± 0.08 | equi − entropy: −0.14 / +0.86 | 0.39 |
| Clean | M2b entropy gate only | 51.41 ± 0.08 | 25.27 ± 0.34 | equi − entropy: −2.04 / +1.06 | 0.39 |
| Clean | equi gate only (rerun) | 49.37 ± 0.19 | 26.33 ± 0.55 | | 0.38 |
| Adv | base (global one_step 6/255) | 45.49 ± 0.41 | 28.19 ± 0.22 | | 0.46 |
| Adv | M1 `no_up` | 45.18 ± 0.18 | 28.39 ± 0.32 | −0.31 / +0.20 | 0.20 |
| Adv | M1 `up1.5` | 45.53 ± 0.50 | 28.54 ± 0.26 | +0.04 / +0.35 | 0.34 |
| Adv | M3 `equi` / `conf` / `inv_conf` | 45.03 / 45.66 / 44.79 | 28.65 / 28.12 / 28.36 | all within ±0.7 | 0.46 |

- **M1: adopt `up1.5`.** It is within 1.5 of `full` on both models, at ~1.4× less defense time. `no_up` fails on the clean model (−2.11 PGD+def). That matches the paper's Table 6: resize views matter, but 2× isn't needed. Timings were measured on a mostly idle GPU; re-time on 10 images before quoting them.
- **M2b: tie → the wording rests on P6a.** The calibrated entropy gate defends 97% of PGD images at the 5% clean rate (equi: 100%) and gives the same mIoU within 1.5. So on plain PGD, the confidence gate (inverted) is as good as ours.
- **M3: fails on every variant (negative result).** `conf` weights are ~uniform (the max softmax is close to 1 almost everywhere), so they change nothing. `inv_conf` hurts the clean model by 6.1. `equi` is within noise. This is the same conclusion as the P4 placement test: where the defense acts matters less than how much.
- **Timing caveat:** M1 runs overlapped a few minutes of smoke tests at the start; the speed column is indicative only.

### M4. Multi-scale feature equivariance (G4): not run

P2 already shows that penultimate-layer and logit-cosine measures separate attacks worse than softmax L2 on the adv model. A multi-scale *defense loss* would mean a new loss with new hyperparameters and a new tuning round, which the schedule can't absorb. Report the P2 evidence and list it as future work.

---

## P4c. Final config and test100 confirmation (Days 6–7)

1. **Lock the config** from P6a and the M1–M3 rules:
   - the budget floor (if P6a broke the gate)
   - the view set (M1)
   - the loss weighting (M3)
   - the signal (M2b)

   Record it in this file before running test100.
2. **Adv model ε_v retune** (tune100 already done): `one_step` global ε_v at 4, 6 and 10/255. Tune100: 4/255 gives 47.06 clean+def / 27.86 PGD+def; 10/255 (preset) gives 40.90 / 27.80. Choose by PGD+def first, then clean+def. Confirm the choice on test100.
3. **test100, 3 seeds, both models.** Fill the table below. Seed std on test100 has been ≤ 0.9.

| Model | Variant | Clean+def | PGD+def | sec/img |
|:--|:--|:--:|:--:|:--:|
| both | No defense | 58.05 / 51.07 | 1.31 / 24.84 | 0 |
| both | Original defense (20 steps, paper ε_v) | 34.16 / 43.47 | 27.34 / 27.51 | 7.2 / 8.3 |
| both | `one_step`, paper ε_v | 34.86 / 44.03 | 26.84 / 27.51 | 0.35 / 0.41 |
| adv | `one_step`, retuned ε_v | | | |
| adv | perpixel (small effect, below threshold; reported for completeness) | | | |
| clean | gate only | | | |
| clean | perimage only | | | |
| clean | gate+perimage, equivariance signal (**ours**) | | | |
| clean | gate+perimage, entropy signal (M2b) | | | |
| clean | ours + budget floor (only if P6a needs it) | | | |
| both | ours + M1 view set / + M3 weighting (only if they passed) | | | |

**Locked config (2026-10-28, before any test100 run):**
- **Clean model (ours):** `one_step`, ε_v 6/255, view set `up1.5` (M1), uniform loss (M3 failed), equivariance gate `down|l2_softmax` (M2b + P6a), per-image budget, **no floor** (P6a: the floor fails). Def-opts `{"mode": "gate+perimage", "view_set": "up1.5"}`.
- **Adv model (ours):** `one_step`, global (no gate: the adv gate failed P2), view set `up1.5`, **ε_v 4/255**. Tune100, up1.5, 3 seeds (`results/improvements/p4c_adv_up15_g{2,4}_tune100_*`, `m_adv_up15_*`): 2/255 47.82 / 27.15, 4/255 47.03 / 28.14, 6/255 45.53 / 28.54 (clean+def / PGD+def). Choice procedure: (1) all three tie on PGD+def (within 1.5 of the best); (2) best clean+def is 2/255, and 6/255 falls out (−2.29), leaving 2 and 4 tied; (3) between those, the higher PGD+def → 4/255.
- Every test100 row below uses the up1.5 view set unless marked "full".

**Done when:** the table is filled and committed. The headline is a clean-model row: clean+def close to the undefended model, at PGD+def within 1.5 of the original defense.

**Outcome (2026-10-28; test100, 3 seeds, mean ± std; `results/improvements/p4c_*_test100_s{0,1,2}.json`; up1.5 views unless marked; non-adaptive PGD-50 only, see P6a for adaptive attacks):**

| Model | Variant | Clean+def | PGD+def | s/img | gate defends (clean / PGD) |
|:--|:--|:--:|:--:|:--:|:--:|
| clean | No defense | 58.05 | 1.31 ± 0.05 | 0 | — |
| clean | Original defense (20 steps, ε_v 6) | 34.16 ± 0.87 | 27.34 ± 0.87 | 7.2 | — |
| clean | `one_step` global ε_v 6 | 33.51 ± 0.42 | 26.83 ± 0.54 | 0.25 | — |
| clean | gate only | 55.17 ± 0.17 | 26.13 ± 0.51 | 0.27 | 7% / 100% |
| clean | perimage only | 54.87 ± 0.10 | 26.23 ± 0.22 | 0.27 | — |
| clean | **ours: equi gate + per-image** | **57.45 ± 0.09** | **25.77 ± 0.49** | 0.25 | 7% / 100% |
| clean | entropy gate + per-image (M2b) | 58.04 ± 0.00 | 23.95 ± 1.16 | 0.28 | 2% / 89% |
| clean | ours, full views (M1 off) | 57.43 ± 0.06 | 26.39 ± 0.46 | 0.38 | 7% / 100% |
| adv | No defense | 51.07 | 24.84 ± 0.02 | 0 | — |
| adv | Original defense (20 steps, ε_v 10) | 43.47 ± 0.30 | 27.51 ± 0.71 | 8.3 | — |
| adv | `one_step` global ε_v 10 | 45.25 ± 0.81 | 28.57 ± 0.38 | 0.35 | — |
| adv | **ours: `one_step` global ε_v 4 (retuned)** | **50.20 ± 0.09** | **26.73 ± 0.07** | 0.31 | — |
| adv | ours, full views (M1 off) | 49.87 ± 0.11 | 26.42 ± 0.03 | 0.46 | — |
| adv | perpixel ε_v 4 | 51.29 ± 0.06 | 25.87 ± 0.07 | 0.35 | — |

Paired differences (same seeds):
- **Clean model, ours vs original: clean+def +23.29, PGD+def −1.57.** Clean accuracy recovers to 0.6 below the undefended model, at 29× less time. PGD+def misses the "within 1.5" headline criterion by 0.07, so it is reported as "−1.6 PGD+def", not as "equal".
- **Adv model, ours vs original: +6.73 / −0.78** (PGD+def tied). The gain comes from the retuned ε_v, not from adaptivity: the adv gate failed P2.
- **M2b on test100:** ours vs entropy gate is −0.58 / **+1.83**. The entropy gate misses 11% of PGD images, so on test100 the equivariance gate wins on PGD+def by more than 1.5. Together with P6a (worst case 2.95 vs 0.95), the "equivariance-guided" wording stands.
- **Ours vs gate-only / perimage-only:** +2.29 / +2.58 clean+def, −0.36 / −0.46 PGD+def. Combining the two is better than either alone on clean images.
- **M1 (up1.5 vs full views):** clean model +0.03 / −0.61, adv +0.33 / +0.31, at 0.25 vs 0.38 s/img (clean) and 0.31 vs 0.46 (adv). Confirmed: same accuracy, ~1.5× faster.
- **Adv perpixel vs ours:** +1.09 / −0.86, below 1.5. It is consistent with P4: no per-pixel effect.

---

## Exploration phase E1–E4 (added 2026-10-28, days 3–7)

The time freed by finishing P0–P6b early goes to every lead that could still change the results. Each one gets a pre-registered rule on `tune100` (3 seeds, ours = the P4c locked config: clean model gate+perimage ε_v 6 `up1.5`; adv model global ε_v 4 `up1.5`). Anything adopted is confirmed on `test100` and re-attacked with BPDA-50 before it enters the final config (v2).

### E1. Transformation selection: fixed subsets and the per-image oracle (G1)

**Why:** the mid-review promised adaptive transformation selection (G1). It is already done at the dataset level twice (P2 picked the signal view from 6 view types × 5 measures; M1 picked `up1.5` for the defense views). E1 tests the two remaining questions:
1. Is there a better *fixed* subset? The paper's ablation (Table 6) says flip and resize help more than rotation, yet the clean-model defense uses two rotations and no flip.
2. Could a *per-image* choice beat any fixed set?

**View subsets** (built on `up1.5`; view groups: up = the two 1.5× upscales, down = 0.3× and 0.5×, rot = the two ±15° rotations, jit = ColorJitter, flip = the flip view):

| Model | Subsets |
|:--|:--|
| clean (base = up, down, rot) | base, −rot, −rot+flip, +flip, −down, −up |
| adv (base = up, down, rot, jit, flip) | base, −rot, −jit, −flip, −down, −up |

**Runs:** each subset, `tune100`, seeds 0–2, clean+def and PGD+def, with per-image confusion matrices saved (`--per-image`). Base is re-run with per-image matrices, plus defense seeds 1–5 on the seed-0 attacks for the noise oracle. Job list: `results/improvements/e1/e1_jobs.json`; analysis: `python -m equi_lib.e1_analyze`.

**Rule 1, fixed subset (pre-registered):**
- Adopt subset S instead of base if:
  - it beats base by > 1.5 on PGD+def or clean+def while losing ≤ 1.5 on the other, **or**
  - it is within ±1.5 on both and takes ≤ 0.75× base's sec/img.
- If several qualify, take the best PGD+def.

**Rule 2, per-image oracle (pre-registered):**
- Per image, pick the subset with the highest per-image mIoU (seed 0 runs; mIoU over the classes present in the label or the prediction), and pool the chosen confusion matrices into the oracle mIoU.
- The **noise oracle** applies the same best-of-6 pick to 6 reruns of base on the **same seed-0 attacks** (defense seeds 0–5, `--attack-seed 0`; clarified 2026-10-28 before any E1 run, so that only the defense's randomness varies, as for the subsets). Picking the best of several noisy runs inflates the score even when the runs are identical, and this measures by how much.
- **Go to E1b** (a label-free per-image rule) only if oracle − noise oracle > 1.5 on PGD+def.
- Clean+def is not used for the clean model's oracle: the gate leaves 93% of clean images undefended.
- If no-go, report: "even with labels, choosing views per image cannot beat a fixed data-chosen set by more than seed noise."

**E1b (only if Rule 2 passes):** a label-free rule using each view's consistency error on the input. Tune on `tune100`, then confirm on `test100` and attack with BPDA (selection is input-dependent, so it can be gamed like the gate).

**Outcome (2026-10-28; tune100, seeds 0–2, ours config with each subset; `python -m equi_lib.e1_analyze` → `results/improvements/e1/e1_decision_tune100.json`):**

| Model | Subset | clean+def | PGD+def | Δ clean / Δ PGD vs base | s/img (× base) |
|:--|:--|:--:|:--:|:--:|:--:|
| clean | base (`up1.5`) | 51.96 ± 0.06 | 25.38 ± 0.46 | — | 0.250 |
| clean | −rot | 51.35 ± 0.41 | 25.93 ± 0.18 | −0.61 / +0.55 | 0.76× |
| clean | −rot +flip | 51.92 ± 0.07 | 26.36 ± 0.40 | −0.04 / +0.98 | 0.88× |
| clean | +flip | 51.84 ± 0.04 | 25.30 ± 0.38 | −0.12 / −0.08 | 1.12× |
| clean | −down | 51.91 ± 0.15 | 21.64 ± 0.28 | −0.05 / **−3.74** | 0.96× |
| clean | −up | 51.39 ± 0.47 | 24.00 ± 0.39 | −0.57 / −1.38 | **0.46×** |
| adv | base (`up1.5`) | 47.02 ± 0.32 | 28.14 ± 0.14 | — | 0.318 |
| adv | −rot / −jit / −flip / −down | 46.92–47.21 | 27.82–28.34 | all within ±0.35 | 0.82–0.97× |
| adv | −up | 46.77 ± 0.05 | 28.04 ± 0.26 | −0.25 / −0.10 | **0.56×** |

Per-image oracle (PGD+def, seed-0 attacks, best-of-6):

| Model | subsets oracle | noise oracle (6 reruns of base) | gain | verdict |
|:--|:--:|:--:|:--:|:--:|
| clean | 27.48 (mean single run 24.45) | 28.80 (mean single 25.43) | −1.32 | no-go |
| adv | 28.65 (28.00) | 28.71 (28.08) | −0.07 | no-go |

- **Rule 2: per-image selection is a no-go on both models.** Even with labels, picking the best view subset per image gains nothing over picking the best of 6 reruns of the same subset. The noise oracle shows why the control matters: best-of-6 *reruns* alone inflate the clean model's PGD+def by +3.4 (25.43 → 28.80). Without it, per-image selection would have looked like a +3 gain. E1b is not run.
- **Rule 1: −up qualifies on both models** (within ±1.5 on both metrics at 0.46× / 0.56× time). It is the same view set as M1's `no_up`, and the runs reproduce M1 exactly (51.39 / 24.00), as does base (51.96 / 25.38).
- **Caveat (drift):** M1 rejected `no_up` on the clean model against `full` (−2.11 PGD+def). E1 accepts it against `up1.5`, which was itself accepted as within 1.5 of `full`. Two equivalence steps can add up to more than the threshold. So −up is **not adopted directly**; it is carried into the v2 `test100` confirmation as a candidate with a stricter, pre-registered test (below).
- **Flip swap (−rot +flip, clean model): +0.98 PGD+def at 0.88× time, below the 1.5 threshold, so not adopted.** Its direction agrees with the paper's ablation (flip/resize > rotation). Rotations add nothing on the clean model (−rot: +0.55 PGD+def, 0.76× time, just missing the 0.75× speed rule).
- **What the views do:** on the clean model the downscale views carry the defense (−down: −3.74), the same scale direction as the P2 signal view (0.5×). On the adv model at ε_v 4, no view group matters (all within ±0.35); only the time differs.

**−up in v2 (pre-registered 2026-10-28, before any test100 run of it):** adopt −up for a model only if, on `test100` with 3 seeds, it is within 1.5 of the P4c locked config **and** within 1.5 of the P4c full-view row (`ours, full views`) on both clean+def and PGD+def. Otherwise report it as a "fast mode" with its measured cost.

### E2. Prediction averaging over views (= P5 below)

Rules and variants are in the P5 section.

### E3. Self-check output (= P5b B1)

The held-out-view error becomes a reliability output per image and per pixel, if B1's pre-registered reading passes. This is where the BPDA collapse would turn into "the system flags it".

### E4. M5 held-out budget selection (P5b), D1 and A1

As written in P5b and P6b. Optional; first in the cut order.

---

## P5. Prediction averaging over views (E2, day 4)

After the defense, predict on the restored image and its views, map each prediction back (resize back; un-flip; inverse-rotate with mask), and average the softmax probabilities over valid pixels.

- Use the resize and flip views; include rotations only if they help on `tune100`.
- Report as an extra row in the P4 table, applied to both the original defense and ours, so its contribution is separated from the main idea.

**Variants (added 2026-10-28):**
- `avg_flip`: identity + horizontal flip (2 predictions)
- `avg_rs`: identity + flip + resize 0.75× + resize 1.25× (4 predictions)

Each is applied to every image after ours (gated-out images too, so the pipeline stays the same for all inputs). Also run on the **undefended** model, to separate the effect of plain test-time augmentation (a known input-transformation defense; cite Guo et al. 2018, Xie et al. 2018) from the effect on the defense.

**Rule (pre-registered):** on `tune100`, 3 seeds, adopt the best variant if it beats ours by > 1.5 on PGD+def or clean+def while losing ≤ 1.5 on the other. The averaging is differentiable, so the BPDA re-check (v2) attacks through it exactly; BPDA's identity backward applies only to the purifier.

**Done when:** we know whether it adds mIoU on clean and attacked inputs. If it adds nothing, report that and drop it.

**Outcome (2026-10-28; tune100, seeds 0–2, ours = P4c config with `up1.5`; `python -m equi_lib.e2_analyze` → `results/improvements/e2/e2_decision_tune100.json`; paired Δ vs `single`):**

| Model | Variant | clean (no def) | PGD (no def) | clean+def | PGD+def | predict s/img |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| clean | single | 52.14 | 1.15 | 51.96 ± 0.06 | 25.38 ± 0.46 | 0.040 |
| clean | avg_flip | 53.01 (+0.87) | 1.33 (+0.17) | 52.84 (+0.88) | 27.75 (**+2.38**) | 0.069 |
| clean | **avg_rs** | 53.25 (+1.11) | 1.99 (+0.84) | 53.09 (+1.13) | 29.07 (**+3.69**) | 0.134 |
| adv | single | 48.13 | 25.95 | 47.02 ± 0.32 | 28.14 ± 0.14 | 0.040 |
| adv | avg_flip | 48.45 (+0.32) | 27.84 (+1.89) | 47.43 (+0.41) | 29.64 (+1.50) | 0.073 |
| adv | **avg_rs** | 47.88 (−0.25) | 29.00 (+3.05) | 46.81 (−0.22) | 30.36 (**+2.22**) | 0.139 |

- **Rule: adopt `avg_rs` on both models** (best PGD+def among the qualifying variants), as a v2 candidate. It costs ~0.1 s/img.
- **Clean model: the gain needs the defense.** Averaging alone barely helps the undefended model under PGD (1.15 → 1.99), but after the defense it adds +3.69. So the defense and the averaging work together: the purified image is close enough to correct that the views vote the remaining errors away.
- **Adv model: the gain is mostly plain test-time augmentation.** Averaging alone adds +3.05 to the undefended model under PGD, and +2.22 after the defense. Report it as such and cite the input-transformation defenses (Guo et al. 2018; Xie et al. 2018).
- **Caveat (decisive):** the averaging is differentiable, and input-transformation defenses are known to fall to attacks that differentiate through them, with EOT for random ones (Athalye et al. 2018). The v2 BPDA-50 re-check attacks through the averaging exactly (identity backward only through the purifier) before any claim is made.

---

## P5b. Does restored equivariance mean restored correctness? (added 2026-10-28, days 4–5)

**Why:** Mao et al. assume that restoring equivariance restores the prediction, and leave open whether that holds under strong adaptive attacks (the Consensus summary names it as the main open question). P6b shows the defense fails under BPDA-50. The question is now: **is the failure visible?** If the restored image still shows high equivariance error on a view the defense never optimised, the system can at least flag its own failures. If not, the attack restores equivariance without restoring correctness, which is a direct counterexample to the paper's premise. Either outcome is a finding.

**B1. Held-out proxy analysis** (forward passes only):
- **Held-out views**, never optimised by the defense:
  - Clean model (7 views: resizes 2× (1.5× in `up1.5`), 0.3×, 0.5×, and two rotations): horizontal flip, and resize 0.75×.
  - Adv model (9 views, which include a flip): resize 0.75×.
  - The gate's 0.5× view is *not* held out (the defense has a 0.5× view, cnt 3).
- **Per image, after the defense:** the held-out consistency error (softmax L2 to the base output, as in `consistency_map`) and the image's own mIoU and pixel accuracy.
- **Inputs:** clean, PGD-50 (cached) and BPDA-50 on 50 test100 images, seed 0, ours (P4c config). Check whether `p6b_attacks.py` saves x_adv; if not, regenerate (~15–19 s/img).
- **Report:**
  - The Spearman ρ between the held-out error and per-image mIoU, per input type and pooled.
  - The pixel-level AUROC of the held-out error for "defended prediction wrong".
  - Mean held-out error per input type: does BPDA-50 leave it high, like PGD, or low, like clean?
- **Pre-registered reading:**
  - "Proxy holds" on an input type if ρ ≤ −0.5 and the pixel AUROC is ≥ 0.80.
  - "Failure visible under BPDA" if the image-level AUROC of the held-out error for BPDA-50 vs clean is ≥ 0.90.
  - Caveat to state: an attacker can add a held-out-error term (as λ did for the gate in P6a), so visibility is again a matter of evasion cost, not a guarantee.

**M5 (optional, cut first): held-out-view budget selection.** Only if B1 shows the proxy holds under PGD-50 on the clean model.
- For each image the gate lets through, run `one_step` at ε ∈ {ε_v/2, ε_v} and keep the result with the lower held-out error (B1 views). This costs about 2× the defense time on gated images.
- Compare against the locked `gate+perimage` on `tune100`, 3 seeds.
- **Pre-registered rule:** adopt only if it beats `gate+perimage` by > 1.5 on PGD+def or clean+def, losing ≤ 1.5 on the other, **and** it is not > 1.5 worse under the P6a gate-aware BPDA attack (λ = 1, 50 images).
- **Known risk:** choosing among candidates by a loss is the pattern that broke SOAP (Croce et al. 2022). A negative result is reported as a confirmation of Croce et al.

**D1 (optional): view redundancy.** Mao's theorem says added constraints help only if they carry new information about the label.
- On 10 tune100 images (both models, clean and PGD inputs), compute each view's equivariance-loss gradient w.r.t. δ at the `one_step` start point.
- Report the pairwise cosine and the sign agreement (`one_step` uses only the sign).
- **Use:** explain M1. Near-duplicate 2× views would explain why `up1.5` = `full`. `no_up` failing on the clean model would mean the upscale direction carries information the downscale views don't.
- Minutes of GPU time.

**Done when:** the B1 table is filled and its reading recorded. M5 and D1 are reported if they were run.

---

## P6a. Gate-aware adaptive attack (Days 4–5, moved forward on 2026-10-27)

**Why first:** the main result works by giving attacked images a large budget and clean images almost none. An attacker who keeps the consistency score s(x+δ) low gets past the gate, and the undefended clean model drops to about 1 mIoU. This decides the thesis claim, so it runs **before** the test100 confirmation and the slides.

**Attack** (clean model, `tune100`, 50 images, seed 0; then 3 seeds on the chosen settings):
- Maximise L_task(defended prediction) − λ·s(x+δ), where s = mean `down|l2_softmax`. Gradients go through s exactly (it is differentiable: one 0.5× forward) and through the defense by BPDA.
- λ ∈ {1, 10, 100, 1000}, 50 steps, ε = 4/255, plus one variant with 100 steps.
- A hard version: early stopping per image at the step where s < τ *and* the undefended prediction is still wrong.
- Run the same attack on the M2b entropy gate (target: lower the entropy score).

**Report:**
- Gate TPR under attack (the fraction of attacked images still defended).
- The mean per-image budget.
- Final robust mIoU of ours, of the original defense, and of the undefended model.
- **The evasion cost:** the undefended mIoU of the attacks that evade the gate. If evading the gate means the perturbation must be consistent across scales, the attack gets weaker. This is the paper's own argument, applied to the gate.

**Decision (pre-registered):**
- **Gate holds:** the worst-case robust mIoU over λ is within 1.5 of the original defense's PGD+def → keep the config.
- **Gate breaks:** add a **budget floor**, ε_img ≥ β_img·ε_v with β_img ∈ {0.3, 0.5}, so no image is fully undefended. Choose β_img on `tune100` by worst-case robust mIoU under this attack. Report the clean-accuracy cost of the floor.
- **Even the floor fails:** report the gate as broken by adaptive attack. The contribution then rests on the speed result, the per-image budget (if it survives) and the findings list. Discuss with the guide on day 6.

**Outcome, first run (2026-10-27; clean model, tune100 first 50 images, seed 0, 50 steps; gate+perimage with `up1.5`; `results/improvements/p6/p6a_{equigate,entgate}_tune100_n50_s0.json`):**

| λ | Equi gate: undefended / pipeline / always-on* | defended | Entropy gate: undefended / pipeline / always-on* | defended |
|:--:|:--:|:--:|:--:|:--:|
| 0 | 2.68 / 3.61 / 5.29 | 1.00 | 1.14 / 1.14 / 14.70 | 0.04 |
| 1 | 2.81 / **2.95** / 12.22 | 0.84 | 0.98 / 0.98 / 16.61 | 0.00 |
| 10 | 5.50 / 5.48 / 17.27 | 0.06 | 0.98 / 0.98 / 20.64 | 0.00 |
| 100 | 6.05 / 6.05 / 19.25 | 0.00 | 0.99 / 0.99 / 25.35 | 0.00 |
| 1000 | 6.32 / 6.32 / 19.10 | 0.00 | 0.95 / 0.95 / 25.19 | 0.00 |

\* always-on = `one_step` at global ε_v on the same x_adv. It is not attacked directly, so for λ > 0 it is a transfer number, not a robustness number.

- **Both gates are evaded.** The equivariance gate needs an explicit evasion term, and evading it costs the attacker damage (undefended 6.3 vs 2.7). The entropy gate is evaded even at λ = 0: the per-image best-iterate picks the early iterates, before the model turns confident. **M2b verdict: the equivariance gate is clearly harder to fool (worst case 2.95 vs 0.95), so the "equivariance-guided" wording stands, but only in relative terms.**
- **BPDA through the defense breaks the clean model even without evasion** (λ = 0: pipeline 3.61, always-on 5.29, against 26 under plain PGD). The paper reports the same weakness for its 20-step defense (Table 2: 11.64 under BPDA-10). A budget floor can gain at most the gap to always-on (~2.3 mIoU here).
- **Pre-registered rule:** "gate breaks", and in absolute terms the floor can't fix it. A control was added before deciding: BPDA-10 (the paper's setting) against the original 20-step defense, always-on `one_step` and the gated pipeline (`p6a_bpda10_*`). It shows whether `one_step` or the gate is weaker than the original under BPDA, which is the comparison that matters.

**BPDA-10 control (2026-10-27; same 50 images, seed 0, 10 steps; `results/improvements/p6/p6a_bpda10_*_tune100_n50_s0.json`):**

| Defense attacked | λ | undefended | **attacked defense** | always-on* | defended | s/img |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| original 20-step (upstream) | 0 | 8.45 | **9.98** | 9.86 | 1.00 | 87.2 |
| `one_step`, always on (up1.5) | 0 | 8.98 | **9.74** | 9.74 | 1.00 | 3.7 |
| ours: equi gate + per-image (up1.5) | 0 | 6.46 | **9.18** | 11.80 | 1.00 | 3.4 |
| ours | 1 | 8.50 | **8.75** | 14.65 | 0.88 | 3.5 |
| ours | 10 | 14.05 | **14.22** | 18.65 | 0.02 | 2.1 |

- **Under the same attack, the original 20-step defense is no better:** 9.98 vs 9.74 for `one_step` (Δ −0.24, noise) at 24× less time. The paper's 11.64 (Table 2, full val) is the same regime. So BPDA breaks the defense family, not our changes.
- **The gate's worst case costs 1.23 vs the original** (8.75 at λ = 1 vs 9.98), within the 1.5 threshold. Full evasion (λ = 10) *helps* the defender (14.22): the evasion term uses up the attacker's budget, the same evasion-cost effect as in the 50-step run.
- **50 steps are much stronger than 10:** `one_step` always-on drops from 9.74 (BPDA-10) to 5.29 (BPDA-50, λ = 0, above). The paper's BPDA-10 understates the attack; P6b reports BPDA-50 too. The original at 50 steps would take ~6 h for 50 images, so it is not run here.
- **Reading of the rule:** the strict pre-registered comparison (worst case vs the original's *plain-PGD* PGD+def, 27.34) says "gate breaks", so the budget floor is tested next as pre-registered (β_img ∈ {0.3, 0.5}, BPDA-50, λ ∈ {0, 1, 10}). The like-for-like comparison (same attack on both) says the gate costs ≤ 1.23. Both are reported.

**Budget floor (2026-10-28; BPDA-50 gate-aware attack as in the first run, same 50 images, seed 0; clean cost on tune100, 3 seeds; `results/improvements/p6/p6a_floor{03,05}_tune100_n50_s0.json`, `results/improvements/m_clean_floor{03,05}_tune100_s{0,1,2}.json`):**

| Config (up1.5, equi gate + per-image) | λ = 0 | λ = 1 | λ = 10 | **worst case** | clean+def (tune100, 3 seeds) | PGD+def (plain PGD) |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| no floor | 3.61 | 2.95 | 5.48 | **2.95** | 51.96 ± 0.06 | 25.38 ± 0.46 |
| floor β_img = 0.3 | 3.63 | 3.39 | 5.76 | **3.39** | 47.76 ± 0.28 (−4.20) | 25.50 ± 0.55 |
| floor β_img = 0.5 | 3.70 | 3.40 | 5.59 | **3.40** | 42.71 ± 0.23 (−9.25) | 25.50 ± 0.55 |
| ref: `one_step` always on, attacked directly | 5.29 | — | — | 5.29 | — | — |

- **Verdict (pre-registered): "even the floor fails".** The floor gains +0.44 in the worst case (< 1.5) and costs 4.2–9.3 clean+def. **Not adopted.**
- The ceiling for any floor is always-on `one_step` under the same attack (5.29), only 2.3 above the gate. Under BPDA-50, the defense family itself is broken. The gate adds at most ~2.3 of extra weakness (BPDA-50) or 1.23 (BPDA-10, vs the original).
- **Consequence for the claim** (for the guide, day 6): the gate is a *clean-accuracy* mechanism (clean+def 51.96 vs 34.16 for the original, with equal PGD+def under non-adaptive attacks). It is **not** an adaptive-robustness mechanism. The thesis states this limitation openly. The contribution rests on: (1) the 24× faster `one_step`, equal to the original under PGD and BPDA-10; (2) the gate's clean-accuracy recovery and its evasion cost (the equivariance gate is harder to evade than the entropy gate: 2.95 vs 0.95); (3) the findings list (the 20-step defense is one step; BPDA-50 ≫ BPDA-10; the per-pixel budget fails its placement test).

## P6b. Remaining attacks (Days 8–12)

Attack the **final system** (P4c config, + P5 if adopted) and the **original defense**, on `test100` (or 50 images for the expensive attacks).

| Attack | Source | Implementation note | Images |
|:--|:--|:--|:--:|
| PGD-50 | Madry 2018 | cached (P0) | 100 |
| SegPGD-50 | Gu et al. 2022 | already in `equi_lib/attacks.py` (`segpgd`) | 100 |
| BPDA-10 | Athalye 2018 | identity backward through the defense (reuse `evaluate_bpda_defense.py` logic) | 50 |
| BPDA-10 + EOT-4 | Athalye 2018; Croce 2022 | average the BPDA gradient over 4 defense runs (random views) | 50 |
| Full-gradient (smoothed) | Lee & Kim 2023 (surrogate idea) | sign replaced by a smooth surrogate (see the P3 gradient check), 2–5 unrolled steps, gradient checkpointing and AMP; 16 GB will not fit the full 20-step unroll | 20–50 |
| **Gate-aware** | P6a | re-run the strongest P6a setting on `test100`, 3 seeds, against the final config | 50 |
| ε(p)-aware | this project | **only if** a per-pixel variant is in the final config (after P4 it is not); otherwise dropped | — |
| CosPGD-50 | Agnihotri et al. 2023 | a few lines on top of PGD | 100 |
| BPDA over intermediate iterates | Croce et al. 2022 | average the BPDA gradient over the defense's intermediate steps (how SOAP was broken) | 50 |
| Fixed-seed and transfer | Croce et al. 2022 | freeze the RNG and attack the now-deterministic defense; transfer unsuccessful perturbations from the undefended model | 50 |

- **Report worst-case robust mIoU** over all attacks for each defense, and report the gate's TPR/FPR under the gate-aware attack, not only AUROC on plain PGD.
- Report results **with and without EOT**: the injected randomness can inflate robustness, so a cleaner optimiser may look worse under EOT.
- Sanity checks: attack-steps and ε curves should reach near 0 at large ε; white-box should beat transfer; add a random-noise control.
- Optional if time remains: a reduced SEA (APGD over masked CE and JS losses, 100 iterations, via BPDA). No test-time segmentation defense has been evaluated with SEA.
- **A1 (added 2026-10-28, optional; Consensus summary → Lee & Kim 2023, to verify):** purification robustness depends on the attack's settings. BPDA-50 against ours, clean model, 50 test100 images, seed 0: step sizes 0.5, 1 and 2/255, plus 3 random restarts (keep the most damaging per image). Report the minimum as the BPDA-50 number. If it is > 1.5 below the default, re-run the 3-seed row with that setting. It can only lower our numbers; the P6b conclusion already stands without it. APGD-CE through BPDA (Croce & Hein 2020) was considered and dropped for the same reason: a stronger attack can't change "≈ 0 / ≈ 2.7".
- The gate-aware attack is essential: the main idea relies on e, so lecturers will ask whether an attacker can fake a low error. It runs first (P6a); this phase repeats it on test100 against the locked config.
- If it breaks the gate, report that honestly and show how much the budget floor (P6a) still holds.
- If M1 adopted a reduced view set, also run BPDA-10 on it (fewer views could be easier to attack).

**Cost:** BPDA-10 with a 3-step defense is ~4× cheaper than Task 6's 1.5 min/image, about 20 min per 50 images.

**Outcome (2026-10-28; test100; `python -m equi_lib.p6b_summary` → `results/improvements/p6/p6b_summary.json`; jobs in `p6b_jobs.json`; ours = P4c locked config):**

| Model | Defense | Attack | seeds | undefended on x_adv | **defended** | gate defends |
|:--|:--|:--|:--:|:--:|:--:|:--:|
| clean | ours | PGD-50 (transfer, cached, 100 img) | 3 | 1.31 | **25.77 ± 0.49** | 1.00 |
| clean | ours | SegPGD-50 (transfer, 100 img) | 3 | 0.99 | **26.86 ± 0.54** | 1.00 |
| clean | ours | CosPGD-50 (transfer, 100 img) | 3 | 0.00 | **29.16 ± 0.60** | 1.00 |
| clean | ours | random-noise control (100 img) | 1 | 52.75 | **52.02** | 0.07 |
| clean | ours | BPDA-10 (50 img) | 1 | 7.21 | **9.77** | 1.00 |
| clean | ours | BPDA-10 + EOT-4 (50 img) | 1 | 4.80 | **6.60** | 1.00 |
| clean | ours | BPDA-50 (50 img) | 3 | 3.27 | **4.34 ± 0.30** | 1.00 |
| clean | ours | BPDA-50, fixed defense seed (50 img) | 1 | 2.61 | **2.73** | 1.00 |
| clean | ours | SegPGD-50 via BPDA (50 img) | 1 | 1.81 | **2.65** | 1.00 |
| clean | ours | gate-aware BPDA-50, λ = 1 (50 img) | 3 | 3.28 | **3.67 ± 0.17** | 0.91 |
| clean | ours | gate-aware BPDA-50, λ = 10 (50 img) | 3 | 5.61 | **5.66 ± 0.21** | 0.16 |
| clean | original | SegPGD-50 / CosPGD-50 (transfer, 100 img) | 1 | 0.98 / 0.00 | **27.45 / 28.13** | — |
| clean | original | random-noise control (100 img) | 1 | 51.13 | **33.86** | — |
| clean | original | BPDA-10 (50 img) | 1 | 10.14 | **11.18** | — |
| adv | ours | SegPGD-50 / CosPGD-50 (transfer, 100 img) | 3 | 23.33 / 22.17 | **26.42 ± 0.14 / 26.82 ± 0.09** | — |
| adv | ours | random-noise control (100 img) | 1 | 51.04 | **50.02** | — |
| adv | ours | BPDA-10 (50 img) | 1 | 27.90 | **26.35** | — |
| adv | ours | BPDA-10 + EOT-4 (50 img) | 1 | 27.29 | **25.83** | — |
| adv | ours | BPDA-50 (50 img) | 3 | 26.85 | **24.05 ± 0.12** | — |
| adv | ours | BPDA-50, fixed defense seed (50 img) | 1 | 27.68 | **22.17** | — |
| adv | ours | SegPGD-50 via BPDA (50 img) | 1 | 26.37 | **22.80** | — |
| adv | original | SegPGD-50 / CosPGD-50 (transfer, 100 img) | 1 | 23.34 / 22.18 | **26.84 / 27.31** | — |
| adv | original | random-noise control (100 img) | 1 | 51.14 | **43.80** | — |
| adv | original | BPDA-10 (50 img) | 1 | 33.20 | **23.71** | — |

Undefended model under its own strongest attack, same 50 images (`*_test100_n50_s0`): clean model 0.00 (CosPGD); adv model 22.40 (CosPGD; PGD 25.05, SegPGD 23.66).

**Findings:**
1. **Non-adaptive attacks (transfer): ours = original.** SegPGD 26.86 vs 27.45, CosPGD 29.16 vs 28.13 (clean model); 26.42 vs 26.84 and 26.82 vs 27.31 (adv). All within 1.5.
2. **Random-noise control: the gate is quiet** (defends 7% of noisy images = its clean false-positive rate), so ours keeps 52.02 vs the original's 33.86 on the clean model. On the adv model (no gate) ours keeps 50.02 vs 43.80, from the smaller ε_v.
3. **Like-for-like adaptive (BPDA-10): ours vs original −1.41 (clean model), +2.64 (adv).** Both within or above the 1.5 threshold. The original's 11.18 reproduces the paper's Table 2 (11.64).
4. **Worst case over the adaptive attacks run (ours):** clean model **2.65** (SegPGD via BPDA) vs undefended 0.00. Adv model **22.17** (BPDA-50, fixed seed) vs the undefended adv-trained model's own worst case 22.40. **Under a strong adaptive attack, the test-time defense adds ~0 robustness on top of adversarial training, and ~2.7 mIoU on the clean model.** This is the main negative finding of P6.
5. **Sanity checks pass:** white-box ≪ transfer (BPDA 4.34 vs PGD+def 25.77); more steps are stronger (BPDA-10 9.77 → BPDA-50 4.34); EOT is stronger (9.77 → 6.60); a fixed defense seed is stronger (4.34 → 2.73, adv 24.05 → 22.17), so the defense's randomness inflates robustness (Croce et al. 2022). On the adv model, "undefended on x_adv" > "defended" under BPDA: the perturbation is tuned to the purified input.
6. **Gap closed: BPDA-50 on the original (2026-10-28; first 20 test100 images, seed 0; `p6b_bpda50_{ours,original}_{clean,adv}_test100_n20_s0.json`; 378–445 s/img for the original vs 15–19 for ours):**

   | Model | ours | original | Δ (ours − original) |
   |:--|:--:|:--:|:--:|
   | clean | 4.01 | 5.78 | −1.77 |
   | adv | 25.79 | 21.88 | +3.91 |

   **The original collapses too** (clean 11.18 under BPDA-10 → 5.78 under BPDA-50). The defense family is broken by a stronger BPDA, not our faster version. The clean-model gap (−1.77, 20 images, 1 seed) is just past the 1.5 threshold and on few images. Report it as "slightly weaker, within the noise of a 20-image run". On the adv model ours is better by 3.9, from the smaller ε_v (less to exploit). **Thesis line: the paper's BPDA-10 evaluation overstates the defense; at 50 steps the upstream defense drops by about half (clean model) and falls below the undefended adv-trained model.**

---

## P7. Final runs, demo, write-up (Days 15–25)

- **Days 15–16:** final configurations on `full500` with seed 42 (overnight) for both models, so they line up with the reproduction table.
- **Day 17, demo** (`demo/app.py`, Gradio):
  - choose an image and a model
  - attack it (PGD, SegPGD or the gate-aware attack)
  - show the clean / attacked / original-defense / our-defense segmentation maps, the equivariance consistency map, the gate decision and the per-image budget
  - show mIoU and time
- **Days 18–22, thesis:**
  - reproduction (`REPRODUCTION_REFERENCE.md`)
  - the findings list under "Goal" (one-step collapse; weaker under BPDA on the adv-trained model, since Mao already report the BPDA drop for the clean model; unrolled gradient = BPDA; placement doesn't matter; confidence inverted; features weaker; ε_v not optimal)
  - method (signal choice from P2, gate, per-image budget), ablations (P4c table), adaptive evaluation (P6), limitations
  - **a chapter or section "From the mid-review to the final design"** built on the M1–M4 table, so the reader sees each proposal, its test and its outcome. Negative results count as results.
- **Days 23–25:** slides (reuse the P2 figures and the P4c / P6 tables) and buffer.
- **Future work** (list in the thesis):
  - multi-scale feature equivariance as a defense loss (G4)
  - per-input view selection (G1)
  - other datasets (ImageNet, VOC, COCO) and backbones
  - restart selection by lowest loss (SOAP did this and was broken)
  - diffusion priors, certification, patch attacks
  - added 2026-10-28 from the Consensus summary (verify each source first):
    - a learned one-step initialiser for the repair (amortisation; Lee 2023, Bi 2024)
    - feature-space or dual-domain repair (Wang 2025, Su 2026)
    - equivariance-preserving randomized smoothing for certified robustness (Schuchardt et al. 2023)
    - per-input transform policies (Kim et al. 2020)
- **Citation rule (2026-10-28):** every source taken from the Consensus summary (`reports/consensus_summary_2026-10-28.md`) is checked against the paper itself (title, authors, venue, the claim attributed to it) before it goes into the thesis. Several (Mittal 2026, Su 2026, Krumtunger 2026, Wesego 2026) could not be checked when the summary was received.

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
| Gate-aware attack breaks the gate | P6a (day 5) | Budget floor β_img; if that fails too, report it as a limitation and fall back to the findings + speed contribution (guide meeting day 6) |
| Entropy gate does as well as the equivariance gate | M2b (day 4) | Settle it with P6a; otherwise rename the contribution as in M2b |
| test100 doesn't confirm the tune100 gain | P4c (day 7) | Report the test100 numbers as they are; never re-tune on test100 |
| Schedule slips | Behind by > 2 days at day 12 | Skip P5 and M3, run P6b on 50 images only |
| Full-gradient attack doesn't fit in 16 GB | OOM, or the P3 gradient check shows cosine ≈ 1 (it does: 1.000000) | Smoothed 2–5 step surrogate with checkpointing and AMP, on 20–50 images |
| Windows spills GPU memory into system RAM | No out-of-memory error; the run just becomes several times slower (P3: the unrolled check peaked at 21.9 GB on the 16 GB card; adv batch 4 ran at 26.6 s/img) | For P6, log `torch.cuda.max_memory_allocated()` and sec/img per run; treat a peak > 15 GB as a failed configuration |
