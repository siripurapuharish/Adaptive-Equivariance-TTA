# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A final-year-project reproduction of *"Robust Perception through Equivariance"* (Mao et al., ICML 2023): test-time equivariance defense for DRN-D-22 semantic segmentation on Cityscapes. The upstream authors' code lives in `Equi4Rob/` (vendored, with local patches). The repo root holds our own evaluation scripts, results, and reports. There is no build system, linter, or test suite; everything is run as plain Python scripts (single GPU, CUDA).

Key docs: `results/FINAL_BENCHMARK_REPORT.md` (the 10 reproduced tasks vs. paper targets) and `REPRODUCTION_REFERENCE.md` (validated configs, verdicts on each reproduction gap, seed variance, research gaps, and the pre-experiment checklist).

## Current work: FYP improvement (deadline ~2026-11-24)

**Follow `IMPLEMENTATION_PLAN.md`.** It is the source of truth for what to build, in what order, and how to evaluate it. Summary:

- **Contribution (reframed 2026-10-27 after P4):** an equivariance-guided adaptive defense. A dense consistency measure (`down|l2_softmax`) decides *whether* to defend each image (gate) and *how strongly* (per-image budget). The per-pixel budget claim was **dropped** (failed its pre-registered placement test). Don't claim the gate itself as novel (Mao et al. A.3.1), nor "first spatially adaptive defense". Mid-review slide ideas are folded in as M1–M4 in `IMPLEMENTATION_PLAN.md`.
- **Supporting parts:** a ~20x faster defense (`one_step`, from our own finding N1; momentum / step decay from MI-FGSM / APGD were tested in P3 and did not help, so they are a rejected ablation) and prediction averaging over the equivariance views (cite it).
- **Evaluation:** PGD, SegPGD, BPDA (+EOT), full-gradient, and a gate-aware adaptive attack. 3 seeds, mean ± std.
- **Phases (do in order):**
  - P0: shared `equi_lib/` + attack cache
  - P1: `EquiDefense` module with `error_map()`, checked equal to upstream
  - P2: go/no-go separation check (AUROC thresholds) + baselines
  - P3: faster optimiser + speed optimisation (profile first; AMP, larger batch, cheaper 2× views, reuse gate views)
  - P4: gate + per-pixel budget (done; per-pixel dropped, gate + per-image kept)
  - M1 / M3 / M2b: reduced view set, weighted loss, entropy-gate baseline (tune100, pre-registered rules)
  - P6a: gate-aware attack (moved before the confirmation)
  - P4c: lock config + test100 confirmation
  - P6b: remaining attacks
  - P5: view averaging (optional)
  - P7: full500 runs, Gradio demo, write-up
- **Rules:**
  - Implement from one Claude session only (two sessions editing at once overwrote notes on 2026-10-26).
  - Tune only on `tune100` (val indices i % 5 == 2); report on `test100` (i % 5 == 0) / `full500`. Interleaved because the val list is sorted by city.
  - Run experiments with `python -m equi_lib.runner --model {clean,adv} --split test100 --seeds 0 1 2 --modes clean pgd clean+def pgd+def` (see `equi_lib/runner.py`).
  - New results go in `results/improvements/`; never overwrite the reproduction results in `results/*.json`.
  - Error map is computed from the input and detached (stop-gradient).
- **Progress:** update the line below as phases finish.
  - Schedule: re-planned 2026-10-28 after P6b (P0–P6b done by day 3; day 25 = 2026-11-19). Exploration phase E1–E4 on days 3–7: E1 view subsets + per-image oracle (G1, includes the flip-for-rotation swap), E2 = P5 view averaging, E3 = P5b self-check, E4 = M5 / D1 / A1 (optional). Then lock v2 + test100 + BPDA-50 re-check (days 8–9), full500 last (days 10–11), demo, thesis, slides. Leads from the Consensus summary are folded in (reports/plan_additions_consensus_2026-10-28.md; citations unverified). See the table at the top of `IMPLEMENTATION_PLAN.md`.
  - E1 done 2026-10-28 (tune100, 3 seeds; results/improvements/e1/; IMPLEMENTATION_PLAN.md E1 'Outcome'): per-image view selection NO-GO on both models (oracle gain over noise oracle −1.32 clean / −0.07 adv; best-of-6 reruns alone inflate +3.4). Fixed subsets: −up qualifies on speed (0.46x / 0.56x time, within 1.5) but is carried to the v2 test100 check because of drift vs M1; flip swap +0.98 PGD+def on clean model (below 1.5); −down −3.74 on clean model (downscale views carry the defense).
  - E2 (= P5 view averaging) done 2026-10-28 (tune100, 3 seeds; results/improvements/e2/; plan P5 'Outcome'): `--predict avg_rs` (identity + flip + 0.75x + 1.25x, mean log-softmax) PASSES on both models: PGD+def +3.69 clean model (needs the defense: undefended PGD only +0.84) / +2.22 adv (mostly plain TTA: undefended +3.05); clean+def +1.13 / −0.22. v2 candidate; must survive BPDA-50 attacking through the averaging.
  - E3 (= P5b B1 self-check) done 2026-10-28 (test100 n50 s0; results/improvements/e3/; plan P5b 'Outcome of B1'): held-out-view error (flip + 0.75x clean model; 0.75x adv) DETECTS the BPDA-50 failure: image AUROC bpda vs clean 0.999 (clean model) / 0.954 (adv); BPDA leaves held-out error 5x the clean level, so the attack breaks correctness without restoring equivariance. Pixel AUROC for wrong pixels 0.84-0.88. Within-input image ranking weak (rho only -0.56 for clean-model PGD).
  - E3b done 2026-10-28 (clean model, test100 n50 s0): self-check-aware BPDA-50 lam 1 / 10 / 100 -> defended 4.49 / 16.84 / 49.60, detection AUROC 0.970 / 0.782 / 0.023: the self-check SURVIVES (detected, or evasion costs >= 5 mIoU). The defense can be broken but not silently. Rerun shows B1's PGD proxy (rho) is unstable -> M5 dropped. lam 3 added post hoc as a check.
  - v2 done 2026-10-28 (test100, 3 seeds; results/improvements/v2/; plan 'v2 confirmation' Outcome): avg_rs ADOPTED on both models (PGD+def +4.62 clean model / +2.43 adv; BPDA-50 through it within 1.5 of plain). -up adopted for adv only. Final v2: clean model = up1.5 + equi gate + per-image + avg_rs: clean+def 58.69 / PGD+def 30.39 (original 34.16 / 27.34); adv = up1.5-up + eps 4 + avg_rs: 49.82 / 29.51 (original 43.47 / 27.51). Pending: BPDA-50 on the adv v2 config (-up + avg_rs). Then full500 last.
  - P1 done 2026-10-26: `equi_lib/defense.py` EquiDefense passes the equivalence test (results/improvements/p1/). P2 core done 2026-10-26 (tune100, seed 0, results/improvements/p2/): clean model PASS (image AUROC 1.00, pixel 0.90); adv model FAIL (image 0.52, pixel 0.47; entropy does better: 0.70 / 0.83). Extended P2 table done (results/improvements/p2/p2ext_*): measure `down|l2_softmax` (softmax L2 between base output and a 0.5x-downscaled view, mapped back) passes on BOTH models for PGD (adv pixel 0.921 / image 0.901; clean 0.987 / 1.000) and SegPGD; adv image-level fails for low-eps PGD (0.635). Chosen as best of 32 on tune100; confirmed on test100 on 2026-10-27 with the choice locked in advance (results/improvements/p2/p2ext_*_test100_s0.json): clean model pixel 0.989 / image 1.000 (PASS both); adv model pixel 0.918 (PASS) / image 0.872 (FAIL, < 0.90; it was 0.901 on tune100). So for the adv model the per-pixel budget is supported but a hard image-level gate is not; the P4 design for the adv model must not rely on the gate. P4 should use it for the budget map; the defense objective stays upstream's.
  - P3 (optimiser) done 2026-10-27 (results/improvements/p3/, p3_*, p3confirm_*): profile (2x views = 63-76% of step cost; batch 4 rejected); exact unrolled gradient == BPDA (so P6 needs a smoothed surrogate); `one_step` chosen as the P4 base: matches the original on test100 x 3 seeds (PGD+def clean 26.84 vs 27.34, adv 27.51 vs 27.51) at ~20x speed (0.35 / 0.41 s/img). Finding N7: `sign_decay3` lowers clean-model robustness by 5.3 while raising clean+def by 6.0. P3b AMP / cheaper-view tests not run (decided 2026-10-27: not needed at ~20x; revisit only if P6 is too slow). Deviation: the 2nd confirmed variant was sign_decay3, not momentum3 (the strict top-2 rule); reason in IMPLEMENTATION_PLAN.md P3 'Outcome'. Equivariance-loss evidence for N7: results/improvements/p3/loss_check_*.json. Slide figures for the chosen P2 measure: results/improvements/p2/fig_*.png (`python -m equi_lib.p2_figures`).
  - P4 decision runs done 2026-10-27 (tune100, 3 seeds; IMPLEMENTATION_PLAN.md P4 "Outcome"; results/improvements/p4/decision_tune100.json): pre-registered placement test FAILS on both models (adv perpixel vs perimage/shuffled +0.75/+1.29 PGD+def, < 1.5; clean perpixel loses 3.3 to perimage), so the per-pixel claim is dropped. Clean model: image-level adaptation (gate / per-image budget) beats the global eps_v curve by > 17 mIoU; fallback gate+perimage = 51.97 clean+def / 26.12 PGD+def (undefended clean 52.14, best global 32.04 / 26.31). Adv model: nothing beats the curve by > 1.5 (negative). Only vs non-adaptive PGD so far; the P6 gate-aware attack is decisive. Next (re-plan 2026-10-27): day 3 = code M1 / M3 / M2b and run them overnight on tune100; then P6a gate-aware attack before any test100 confirmation.
  - P6a done 2026-10-28 (clean model, tune100 n50 s0; results/improvements/p6/; IMPLEMENTATION_PLAN.md P6a "Outcome"): the gate-aware BPDA-50 attack evades both gates (ours worst case 2.95, entropy gate 0.95). BPDA-10 control: the original 20-step defense 9.98 vs one_step 9.74 vs gated worst case 8.75, so BPDA breaks the defense family; the gate adds ≤ 1.23. Budget floor fails (+0.44 worst case, −4.2/−9.3 clean+def), not adopted. Claim: the gate is a clean-accuracy mechanism, not adaptive robustness (limitation stated; discuss with the guide). P4c config locked 2026-10-28 (clean: gate+perimage up1.5 eps6; adv: global up1.5 eps4); test100 3-seed queue: results/improvements/p4c/p4c_test100_jobs.json. P4c done 2026-10-28 (IMPLEMENTATION_PLAN.md P4c "Outcome"): clean ours 57.45 / 25.77 vs original 34.16 / 27.34 (paired +23.29 / −1.57; misses "within 1.5" by 0.07); adv ours (eps4) 50.20 / 26.73 vs 43.47 / 27.51 (+6.73 / −0.78); equi gate beats entropy gate +1.83 PGD+def on test100. P6b done 2026-10-28 (IMPLEMENTATION_PLAN.md P6b "Outcome"; `python -m equi_lib.p6b_summary`): transfer attacks (SegPGD, CosPGD) tie the original; gate quiet on random noise (7%); BPDA-10 like-for-like −1.41 (clean) / +2.64 (adv) vs original; worst adaptive case ours 2.65 (clean) / 22.17 (adv, vs undefended adv-trained 22.40), so the defense adds ~0 adaptive robustness. Gap closed: original under BPDA-50 (20 img) collapses too: clean 5.78 (ours 4.01), adv 21.88 (ours 25.79). Next: P5 (optional) / P7.
  - Status: P0 done 2026-10-26 (`equi_lib/` reproduces full500 seed-42 baselines within 0.05; PGD cache and original-defense wrapper verified). 3-seed test100 baselines done 2026-10-26 (4 h 17 min; files `results/improvements/baseline_{clean,adv}_test100_s{0,1,2}.json`; also in REPRODUCTION_REFERENCE.md section 4b). Improvements below ~1.5 mIoU are not significant. Note B: no effect larger than ~2 mIoU found (1 seed only). GPU profiling was done in P3 (results/improvements/p3/profile_*.json).
  - Own baselines, test100, mean ± std over 3 seeds (original defense):

    | Model | clean | PGD-50 | clean+def | PGD+def |
    |:--|:--:|:--:|:--:|:--:|
    | Clean | 58.05 | 1.31 ± 0.05 | 34.16 ± 0.87 | 27.34 ± 0.87 |
    | Adv-trained | 51.07 | 24.84 ± 0.02 | 43.47 ± 0.30 | 27.51 ± 0.71 |

    Note B (upstream-style noise, seed 0): clean model 34.51 vs 35.08 without noise; adv model 44.16 vs 43.71. Differences are about 0.5, the size of the seed noise (1 seed only), so defending plain x looks fine. The 20-step defense costs ~7.1–8.3 s/image (clean / adv model). The ~20 s average printed for the clean model comes from one slow seed-0 clean+def pass (44 s/img, likely GPU contention), so time runs only on an idle GPU.

## Layout

- `Equi4Rob/` – upstream code. Core pieces: `models/DRNSeg.py` + `models/drn.py` (model), `learning/attack_new_single.py` (PGD attack, `Equi_Set_Defense_good_inloop` defense), `learning/BPDA.py` (BPDA attack), `learning/utils_learn.py` (`fast_hist`, `per_class_iu` for mIoU). `equi4robust.py` is the original multi-GPU entry point; our scripts bypass it.
- Root `evaluate_*.py` – one script per experiment type. They add `Equi4Rob/` to `sys.path` and import from it:
  - `evaluate_baseline.py` – clean/no-attack (Tasks 1, 7)
  - `evaluate_attack.py` – PGD-50 with no defense (Tasks 3, 9)
  - `evaluate_clean_equi_defense.py` – clean images + equivariance defense (Tasks 2, 8)
  - `evaluate_equi_defense.py` – PGD-50 + equivariance defense (Tasks 4, 10)
  - `evaluate_bpda_nodefense.py` / `evaluate_bpda_defense.py` – BPDA without / with defense (Tasks 5, 6)
- `run_overnight_queue.py` – runs a fixed queue of long evaluations sequentially, writes `results/overnight_queue_status.txt`, and **auto-commits and pushes results to `origin main`** after each task. `sync_queue_status.py` produces `results/LIVE_QUEUE_STATUS.*`.
- `preprocess_cityscapes.py` – builds `cityscapes_dataset_subsampled/` (680×340). 
- `results/` – each run writes `<prefix>_metrics.json`, `_summary.txt`, `_progress.txt`.
- Gitignored and large: `checkpoints/`, `cityscapes_dataset*/`, `scratch/`, `experiments/`.

## Common commands

Run from the repo root. Checkpoints: `checkpoints/clean_drn_d_22_cityscapes.pth` and `checkpoints/advtrain_drn_d_22_cityscapes.pth.tar` (validate with `python check_checkpoints.py`). Environment spec: `Equi4Rob/environment_equi.yml`.

```bash
# Quick sanity check before a long run
python evaluate_baseline.py --smoke-test
python evaluate_equi_defense.py --smoke-test
# Limit to N images
python evaluate_equi_defense.py --num-samples 10

# Table 2 (clean model) defense runs use these hyperparameters
python evaluate_equi_defense.py --checkpoint checkpoints/clean_drn_d_22_cityscapes.pth \
  --output-prefix baseline_drn_d_22_pgd_equidefense --reverse-multiply 1.5 --num-views 7 --batch-size 2
python evaluate_clean_equi_defense.py --checkpoint checkpoints/clean_drn_d_22_cityscapes.pth \
  --output-prefix baseline_drn_d_22_clean_equidefense --epsilon-defense 0.0235294 --num-views 7 --batch-size 2

# Full queue (hours; commits and pushes on its own)
python run_overnight_queue.py
```

## Things that are easy to get wrong

- **Protocol differs per model.** Clean model (paper Table 2): defense eps_v = 1.5·eps (6/255), 7 spatial-only views. Adversarially trained model (Table 1): eps_v = 2.5·eps (10/255), 9 views including non-invertible `ColorJitter`. Using the wrong set costs ~12 mIoU points.
- **Resolution mismatch.** DRN-D-22 has 8× stride, so a 340-high input produces 344-high logits. Always bilinearly interpolate logits to label size before computing loss or mIoU (see `align_output()` in the evaluate scripts and `BPDA.py`). `PGD_attack_new` in `attack_new_single.py` does not do this itself.
- **Defense is stochastic** (Python `random` + torch RNG: random start, gradient noise, ColorJitter). Our scripts seed 42, but CUDA `grid_sample`/upsample backward is still non-deterministic; report improvements over several seeds.
- **The 20-step defense is effectively one gradient step**: the injected noise drowns the batch-normalised gradient until the last step (measured). Keep it as-is when reproducing; it's the main lever for improvements.
- **BPDA (`BPDA.py`, `evaluate_bpda_defense.py`) is implemented correctly** (identity backward through the purifier). `BPDA.py`/`equi4robust.py` can't be imported because of a missing rotation checkpoint, so use the stand-alone scripts.
- `REPRODUCTION_REFERENCE.md` holds the validated configs, per-finding verdicts, seed variance, and research gaps. Read it before any new experiment.
- When evaluating a new method, compare against our own baselines in the report, not only the paper's numbers.
- Platform is Windows; the scripts are cross-platform but the shell here may be PowerShell or Git Bash.
