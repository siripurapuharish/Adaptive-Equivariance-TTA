# Implementation Plan: Equivariance-Guided Adaptive Defense

Created 2026-10-26. Deadline: about 2026-11-24 (30 days).
Background facts, configs and baselines are in `REPRODUCTION_REFERENCE.md`.

## Goal

**Novel contribution:** the first test-time defense for semantic segmentation that uses dense equivariance error to adapt the defense budget per pixel.

**Supporting parts (cited, adapted):**
- a faster optimiser for the defense (momentum and step decay, from MI-FGSM / APGD)
- prediction averaging over the equivariance views (standard segmentation test-time augmentation)

**Evaluation:** PGD, SegPGD, BPDA (+EOT), a full-gradient attack, and a gate-aware adaptive attack. Every number is mean ± std over 3 seeds.

**Success criteria (measured on the 100-image test subset):**

| Metric | Current baseline | Target |
|:--|:--:|:--:|
| Clean-model, clean images + defense (T8 analog) | 35.66 (500 img) | clearly higher, ideally > 45 |
| Clean-model, PGD + defense (T10 analog) | 29.38 (500 img) | within ±0.5 of baseline (the seed noise) |
| Adv-model, PGD + defense (T4 analog) | 28.48 (500 img) | within ±0.5 of baseline |
| Defense time per image | ~8 s (20 steps) | ≥ 3× faster |

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
| `data.py` | `load_split(name)` returns tensors for `test100` (val images 0–99), `tune100` (val images 400–499), or `full500` |
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

Also save histograms and 5 example error maps (clean / attacked side by side). They go straight into the slides.

**Decision:**
- Both pass → build gate + per-pixel (P4 as planned).
- Only image level passes → build the gate plus a softer per-image budget (ε_v scaled by the image score). This is still novel at test time for segmentation.
- Neither passes → stop. Present the negative result, and switch the main idea to P3 + P5 with stronger evaluation. Decide with your guide.

**Same night:** run the 3-seed baselines on `test100` (original defense, both models, clean and PGD). That's about 3 GPU-hours, run overnight. These numbers are the bar every later variant must beat.

**Also overnight (optional but useful):** the global ε_v trade-off sweep on the clean model, ε_v ∈ {0, 2, 4, 6}/255, 1 seed. It gives the curve the main idea must beat, not just move along.

---

## P3. Faster optimiser (Days 5–7)

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

**Output:** a speed table, ready to drop into the slides.

| Config | sec/img (attacked) | sec/img (clean) | Clean+def mIoU | PGD+def mIoU |
|:--|:--:|:--:|:--:|:--:|
| Original (20 steps) | | | | |
| + 3 steps | | | | |
| + AMP + batch | | | | |
| + cheaper views (if kept) | | | | |
| + gate (P4) | | | | |

**Combined estimate:** ~8 s → ~0.5–1 s per attacked image (~8–15×), and less for gated clean images. These are estimates to confirm by measurement.

**BPDA and adaptive attacks get the same speed-up automatically,** since they run the defense inside each attack step. Do **not** reduce attack strength (BPDA steps, EOT samples) to save time, because that would make the defense look stronger than it is.

---

## P4. Main idea: gate + per-pixel budget (Days 8–13)

**Algorithm (for each input x):**
1. `e = error_map(x)`, computed once from the input, **detached** (stop-gradient), so the optimiser can't game it.
2. **Gate:** s = mean(e). If s < τ, return x unchanged.
3. **Budget map:** smooth e with a 9×9 average pool at feature resolution (avoids speckle), then

   ε(p) = ε_v · clip( (e_p − a) / (b − a), β, 1 )

   - a, b: the 50th percentile of clean-pixel e and the 90th percentile of attacked-pixel e (from `tune100`)
   - β: a floor (default 0.2) so no attacked pixel gets zero budget
4. Run the P3 optimiser with element-wise clamp `δ = clamp(δ, −ε_map, ε_map)` and step size scaled by ε(p)/ε_v.

**Hyperparameters, chosen on `tune100` only:**
- τ: the value at which 95% of clean images pass (skip the defense)
- β ∈ {0, 0.2, 0.4}
- a / b percentiles from the P2 distributions

**Ablation table (`test100`, 3 seeds, both models):**

| Variant | Clean+def | PGD+def | sec/img |
|:--|:--:|:--:|:--:|
| No defense | | | |
| Original defense (P2 baseline) | | | |
| P3 optimiser, global ε_v | | | |
| + gate only | | | |
| + per-pixel only | | | |
| + gate + per-pixel (**ours**) | | | |

**Done when:** the table is filled. The success criteria are at the top of this file.

**Fallback:** if per-pixel doesn't help, ship gate + per-image budget (decided in P2).

---

## P5. Prediction averaging over views (Days 14–15)

After the defense, predict on the restored image and its views, map each prediction back (resize back; un-flip; inverse-rotate with mask), and average the softmax probabilities over valid pixels.

- Use the resize and flip views; include rotations only if they help on `tune100`.
- Report as an extra row in the P4 table, applied to both the original defense and ours, so its contribution is separated from the main idea.

**Done when:** we know whether it adds mIoU on clean and attacked inputs. If it adds nothing, report that and drop it.

---

## P6. Adaptive evaluation (Days 16–21)

Attack the **final system** (P4 + P5) and the **original defense**, on `test100` (or 50 images for the expensive attacks).

| Attack | Source | Implementation note | Images |
|:--|:--|:--|:--:|
| PGD-50 | Madry 2018 | cached (P0) | 100 |
| SegPGD-50 | Gu et al. 2022 | L = (1−λ_t)·CE(correct pixels) + λ_t·CE(wrong pixels), λ_t = (t−1)/(2T) | 100 |
| BPDA-10 | Athalye 2018 | identity backward through the defense (reuse `evaluate_bpda_defense.py` logic) | 50 |
| BPDA-10 + EOT-4 | Athalye 2018; Croce 2022 | average the BPDA gradient over 4 defense runs (random views) | 50 |
| Full-gradient | — | backpropagate through the 3-step defense (feasible because of P3); detach nothing except the error map | 50 |
| **Gate-aware** | adapted from upstream `PGD_attack_new_adaptive` | L = L_task − λ·mean(e(x+δ)), λ ∈ {1, 10, 100, 1000}, to slip past the gate | 50 |

- **Report worst-case robust mIoU** over all attacks for each defense.
- The gate-aware attack is essential: the main idea relies on e, so lecturers will ask whether an attacker can fake a low error.
- If it breaks the gate, report that honestly and show how much the per-pixel budget (without the gate) still holds.

**Cost:** BPDA-10 with a 3-step defense is ~4× cheaper than Task 6's 1.5 min/image, about 20 min per 50 images.

---

## P7. Final runs, demo, write-up (Days 22–30)

- **Days 22–23:** final configurations on `full500` with seed 42 (overnight) for both models, so they line up with the reproduction table.
- **Day 24, demo** (`demo/app.py`, Gradio):
  - choose an image and a model
  - attack it (PGD or SegPGD)
  - show the clean / attacked / original-defense / our-defense segmentation maps, the equivariance error map, and the budget map
  - show mIoU and time
- **Days 25–28, thesis:**
  - reproduction (`REPRODUCTION_REFERENCE.md`)
  - the two findings
  - method, ablations, adaptive evaluation, limitations
- **Days 29–30:** slides and buffer.

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
| Schedule slips | Behind by > 2 days at end of P4 | Skip P5, run P6 on 50 images only |
