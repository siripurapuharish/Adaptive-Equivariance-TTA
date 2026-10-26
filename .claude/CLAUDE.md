# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A final-year-project reproduction of *"Robust Perception through Equivariance"* (Mao et al., ICML 2023): test-time equivariance defense for DRN-D-22 semantic segmentation on Cityscapes. The upstream authors' code lives in `Equi4Rob/` (vendored, with local patches). The repo root holds our own evaluation scripts, results, and reports. There is no build system, linter, or test suite; everything is run as plain Python scripts (single GPU, CUDA).

Key docs: `results/FINAL_BENCHMARK_REPORT.md` (the 10 reproduced tasks vs. paper targets) and `REPRODUCTION_REFERENCE.md` (validated configs, verdicts on each reproduction gap, seed variance, research gaps, and the pre-experiment checklist).

## Current work: FYP improvement (deadline ~2026-11-24)

**Follow `IMPLEMENTATION_PLAN.md`.** It is the source of truth for what to build, in what order, and how to evaluate it. Summary:

- **Novel contribution:** an equivariance-guided adaptive defense. The dense per-pixel equivariance error decides *whether* to defend (image-level gate) and *how much* per pixel (budget map ε(p)). Novelty claim (checked on Scholar / Semantic Scholar): "first test-time defense for semantic segmentation that uses dense equivariance error to adapt the defense budget per pixel." Don't claim "first spatially adaptive defense"; pixel-reweighted adversarial training (2024), SuperPure and FaceGuard exist.
- **Supporting parts (cite them):** a faster optimiser for the defense (momentum + step decay, from MI-FGSM / APGD) and prediction averaging over the equivariance views.
- **Evaluation:** PGD, SegPGD, BPDA (+EOT), full-gradient, and a gate-aware adaptive attack. 3 seeds, mean ± std.
- **Phases (do in order):**
  - P0: shared `equi_lib/` + attack cache
  - P1: `EquiDefense` module with `error_map()`, checked equal to upstream
  - P2: go/no-go separation check (AUROC thresholds) + baselines
  - P3: faster optimiser + speed optimisation (profile first; AMP, larger batch, cheaper 2× views, reuse gate views)
  - P4: gate + per-pixel budget
  - P5: view averaging
  - P6: adaptive attacks
  - P7: full500 runs, Gradio demo, write-up
- **Rules:**
  - Implement from one Claude session only (two sessions editing at once overwrote notes on 2026-10-26).
  - Tune only on `tune100` (val indices i % 5 == 2); report on `test100` (i % 5 == 0) / `full500`. Interleaved because the val list is sorted by city.
  - Run experiments with `python -m equi_lib.runner --model {clean,adv} --split test100 --seeds 0 1 2 --modes clean pgd clean+def pgd+def` (see `equi_lib/runner.py`).
  - New results go in `results/improvements/`; never overwrite the reproduction results in `results/*.json`.
  - Error map is computed from the input and detached (stop-gradient).
- **Progress:** update the line below as phases finish.
  - Schedule: compressed to 25 days on 2026-10-26 (day 25 = 2026-11-19). P0 days 1–2, P1 day 3, P2 day 4 (go/no-go), P3 days 5–7, P4 days 8–12, P5 days 13–14, P6 days 15–20, P7 days 21–25. See the table at the top of `IMPLEMENTATION_PLAN.md`.
  - P1 done 2026-10-26: `equi_lib/defense.py` EquiDefense passes the equivalence test (results/improvements/p1/). P2 core done 2026-10-26 (tune100, seed 0, results/improvements/p2/): clean model PASS (image AUROC 1.00, pixel 0.90); adv model FAIL (image 0.52, pixel 0.47; entropy does better: 0.70 / 0.83). Extended P2 table done (results/improvements/p2/p2ext_*): measure `down|l2_softmax` (softmax L2 between base output and a 0.5x-downscaled view, mapped back) passes on BOTH models for PGD (adv pixel 0.921 / image 0.901; clean 0.987 / 1.000) and SegPGD; adv image-level fails for low-eps PGD (0.635). Chosen as best of 32 on tune100, so it must be confirmed on test100 (fixed choice, no re-picking) before claims. P4 should use it for the budget map; the defense objective stays upstream's.
  - Status: P0 done 2026-10-26 (`equi_lib/` reproduces full500 seed-42 baselines within 0.05; PGD cache and original-defense wrapper verified). 3-seed test100 baselines done 2026-10-26 (4 h 17 min; files `results/improvements/baseline_{clean,adv}_test100_s{0,1,2}.json`; also in REPRODUCTION_REFERENCE.md section 4b). Improvements below ~1.5 mIoU are not significant. Note B: no effect larger than ~2 mIoU found (1 seed only). Next: P1 (`EquiDefense` + `error_map()`), then the P2 go/no-go. GPU profiling (per-stage time, peak memory, idle GPU) is deferred until after P1 but must be done before P3.
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
