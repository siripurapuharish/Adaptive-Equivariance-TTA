# Adversarial Attacks and Defenses for Semantic Segmentation (2017–2026): Evaluation Upgrades and Feasible Extensions for a Test-Time Equivariance Defense

Context: reproduction of Mao et al., ICML 2023 ("Robust Perception through Equivariance"), DRN-D-22, Cityscapes at 680x340, eps = 4/255 L-inf. Reproduced numbers from this repo (`results/FINAL_BENCHMARK_REPORT.md`): adversarially trained model clean 51.34 / PGD-50 25.26 / PGD-50+defense 28.48 / BPDA+defense 22.55. Clean model clean 58.24 / PGD-50 1.32 / PGD-50+defense 29.38 / clean+defense 35.66.

## Q1. Which attacks are now the standard for segmentation robustness evaluation, and is code available?

### Takeaway
By 2026 a reviewer will expect more than PGD-CE. The expected minimum is SegPGD and CosPGD, and ideally the SEA ensemble (APGD with 3 segmentation losses, 300 iterations each, radius reduction). SEA showed that CosPGD and SegPGD overestimate robust mIoU by 6–13 points. A test-time defense must also face adaptive attacks (BPDA+EOT, full gradient through the purifier) per Croce et al. 2022. Code exists for all of the main attacks.

### Cited Findings
**SEA (Croce, Singh, Hein, ECCV 2024, arXiv 2306.12941).** Current gold standard.
- SEA = worst case over 3 losses: Jensen-Shannon divergence, masked cross-entropy (MCE), and balanced masked CE (MCE-Bal). Each runs 300 iterations of APGD with progressive radius reduction: the budget is split 3:3:4 over radii 2ε, 1.5ε, ε. The worst case for mIoU is a greedy approximation, because mIoU isn't defined per image — [arXiv HTML](https://arxiv.org/html/2306.12941v2)
- Main claim: "existing attacks severely overestimate the robustness of semantic segmentation models." Code and models at https://github.com/nmndeep/robust-segmentation — [arXiv abs](https://arxiv.org/abs/2306.12941)
- Concrete overestimation: PIR-AT UPerNet ConvNeXt-T on PASCAL-VOC at 8/255 scores 77.8 acc / 47.3 mIoU under CosPGD and 74.2 / 41.4 under SegPGD, but SEA gives 71.7 / 34.6 — [arXiv HTML](https://arxiv.org/html/2306.12941v2)
- DDC-AT, which SegPGD showed non-robust at 8/255, is non-robust under SEA even at 2/255, where SegPGD still reports robustness — [arXiv HTML](https://arxiv.org/html/2306.12941v2)

**CosPGD (Agnihotri, Jung, Keuper, ICML 2024, arXiv 2302.02213).**
- Weights each pixel's loss by the cosine similarity between softmax(prediction) and one-hot target. This gives smooth pixel weighting instead of SegPGD's discrete correct/incorrect split. It also works for regression tasks (optical flow, disparity, restoration) — [arXiv HTML](https://arxiv.org/html/2302.02213v3)
- Untargeted update: X_{t+1} = X_t + α·sign(∇ Σ_i cos(softmax(f(X_t)_i), Y'_i) · L(f(X_t)_i, Y_i)) — [arXiv HTML](https://arxiv.org/html/2302.02213v3)
- PASCAL VOC, DeepLabV3, ε≈8/255, α=0.01, ~100 iterations: PGD gives 6.79% mIoU, SegPGD 2.69%, CosPGD ≈0% — [arXiv HTML](https://arxiv.org/html/2302.02213v3)
- Code: https://github.com/shashankskagnihotri/cospgd — [project page](https://www.uni-mannheim.de/dws/research/projects/machine-learning/cospgd/); [PMLR](https://proceedings.mlr.press/v235/agnihotri24b.html)
- Has Cityscapes experiments (UNet with ConvNeXt backbone, appendix B.5); the exact numbers were not retrieved — [arXiv HTML](https://arxiv.org/html/2302.02213v3)

**SegPGD (Gu, Zhao, Tresp, Torr, ECCV 2022, arXiv 2207.12391).**
- Loss: correctly classified pixels are weighted (1−λ) and misclassified pixels λ, both averaged over H×W. Linear schedule λ(t) = (t−1)/(2T); log and exp schedules perform similarly — [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.12391)
- No official repo is linked on the arXiv page — [arXiv](https://arxiv.org/abs/2207.12391). It is easy to re-implement (a few lines on top of PGD).

**ALMA prox (Rony, Pesquet, Ben Ayed, CVPR 2023).**
- Minimum-norm L-inf attack for segmentation using proximal splitting and an augmented Lagrangian, with per-pixel constraints. Finds much smaller-norm perturbations than prior attacks. Code: github.com/jeromerony/alma_prox_segmentation — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Rony_Proximal_Splitting_Adversarial_Attack_for_Semantic_Segmentation_CVPR_2023_paper.html)

**Universal perturbations (Metzen et al., ICCV 2017).**
- A universal perturbation reduced PSPNet's IoU on Cityscapes from 75.8% to 8.8%. It transferred across nets as an untargeted attack but not as a targeted one — [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Metzen_Universal_Adversarial_Perturbations_ICCV_2017_paper.pdf)

**Patch attacks.**
- Nesti et al., WACV 2022: EOT extended to segmentation, digital and physical patches on Cityscapes and CARLA. Real-world patches were "notably less effective." Code: github.com/retis-ai/semsegadvpatch — [CVF PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Nesti_Evaluating_the_Robustness_of_Semantic_Segmentation_for_Autonomous_Driving_Against_WACV_2022_paper.pdf); [GitHub](https://github.com/retis-ai/semsegadvpatch)
- Journal extension: the effect is often spatially confined around the patch — [ar5iv 2201.01850](https://ar5iv.arxiv.org/html/2201.01850)
- Indirect local attacks (universal local patches, Cityscapes, PSANet) — [arXiv 1911.13038](https://arxiv.org/pdf/1911.13038)
- OmniPatch (2026): universal patch with ViT–CNN cross-architecture transfer, evaluated on Cityscapes (details not verified) — [arXiv 2603.20777](https://arxiv.org/pdf/2603.20777)

**Other attacks.**
- SegRMT (2025): a genetic-algorithm attack over spatial and spectral transforms. Reports DeepLabV3 on Cityscapes reduced to 6.4% mIoU, against 21.7–8.5% for baseline attacks. Not a norm-bounded L-inf attack — [arXiv 2504.02335](https://arxiv.org/html/2504.02335v1)
- Houdini and DAG: Houdini already appears in Mao et al.'s Cityscapes tables (see Q3) — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)

**Test-time-defense evaluation standard (Croce, Gowal, Brunner, Shelhamer, Hein, Cemgil, ICML 2022).**
- Of the adaptive test-time defenses they evaluated (classification), "none significantly improve upon static defenses" under careful evaluation. Some weaken the underlying static model while increasing inference compute. The paper extends the Carlini et al. 2019 checklist with steps specific to adaptive defenses — [PMLR](https://proceedings.mlr.press/v162/croce22a.html); [arXiv](https://arxiv.org/pdf/2202.13711)
- The taxonomy separates input adaptation (purification, which is what the equivariance defense is) from model adaptation — [ICML slides](https://ICML.cc/media/icml-2022/Slides/18041.pdf)

**Benchmark tooling.**
- SemSegBench (Agnihotri et al., May 2025): 76 segmentation models on 4 datasets incl. Cityscapes, under adversarial attacks and corruptions. Tools and 6139 evaluations are open source — [arXiv 2505.18015](https://arxiv.org/html/2505.18015v1)

### Inferences
- **Minimum credible eval for a 2026 reviewer:** PGD-CE, SegPGD, CosPGD (all cheap and re-implementable) plus BPDA+EOT and a full-gradient / defense-aware attack. Adding SEA, or at least the APGD-MCE / APGD-JS losses with radius reduction, would be a strong differentiator. No prior equivariance-defense paper evaluated with SEA, so this is a clear evaluation upgrade.
- **Cost of SEA:** 3 losses × 300 iterations = 900 forward+backward passes per image on the undefended model. DRN-D-22 at 680x340 should be manageable on one GPU for 100–500 images. Through the defense it is far more expensive, because each attack step must pass through the 20-step purifier. Use SEA losses with BPDA (identity backward), which costs about the same as attacking the undefended model plus one defense run per EOT sample. A reduced budget (e.g., 100 iterations per loss) on test100 is feasible.
- **Mao et al. omissions:** the paper's adaptive attack (λe=1000 equivariance term) and BPDA are weaker than modern standards. It never used SegPGD, CosPGD or SEA. Our reproduction already shows BPDA+defense (22.55) falls below the undefended PGD number (25.25). This is exactly the Croce 2022 pattern ("weakens the static model") and should be reported honestly.
- **Gate-aware attack:** an attacker can add a term that minimises the image-level equivariance error (making the gate say "clean") while maximising the seg loss. This is analogous to Mao's λe term. It is the most important adaptive attack for the planned gated defense.
- **The pixel weighting in SegPGD/CosPGD is related to the planned per-pixel budget ε(p).** An attack that targets pixels with low ε(p), for example via a mask-weighted loss, should be part of the adaptive suite.

### Gaps
- No official SegPGD repo was found; third-party implementations exist (e.g., inside CosPGD repo — not verified).
- No CosPGD Cityscapes numbers were retrieved (they sit in appendix B.5).
- The SEA paper's visible text contains no Cityscapes results. SEA appears to report on PASCAL-VOC and ADE20K only, so no SEA-on-Cityscapes reference numbers exist in that paper.
- DAG (Xie et al., ICCV 2017) and PAIR were not checked directly in this pass.

## Q2. Which test-time / purification defenses exist for segmentation, has any been combined with equivariance, and what Cityscapes mIoU do defenses get at 4/255 and 8/255?

### Takeaway
Segmentation-specific test-time defenses are scarce. They are Mao et al. 2023 (equivariance), its baselines (random noise, rotation self-supervision, contrastive / Mao 2021, invariance), Naseer et al. 2020 (self-supervised input purification), and detection-only methods (spatial consistency, entropy). For patches there are Z-Mask, PatchZero and Demasked Smoothing. I found no diffusion-purification work evaluated on Cityscapes and no equivariance-based follow-up for segmentation, so the gap is open but it is a small niche. On Cityscapes the best adversarially trained CNNs reach about 13–15% mIoU under PGD-100 at 8/255. Mao et al. report about 30% at 4/255 for the clean model plus defense under PGD, but only 11.6% under BPDA.

### Cited Findings
**Adversarial training on Cityscapes (SegPGD paper, ε=0.03≈8/255).** mIoU values: Clean / PGD20 / PGD100
- PSPNet standard: 73.98 / 0.61 / 0.27
- PSPNet PGD3-AT: 71.28 / 17.3 / 3.95
- PSPNet DDC-AT: 71.86 / 20.90 / 12.97
- PSPNet SegPGD3-AT: 71.01 / 25.23 / 13.04
- DeepLabv3 standard: 73.82 / 0.84 / 0.44
- DeepLabv3 PGD3-AT: 71.45 / 20.23 / 6.74
- DeepLabv3 DDC-AT: 71.68 / 20.80 / 14.82
- DeepLabv3 SegPGD3-AT: 71.04 / 25.49 / 15.23
- Under SegPGD-100 (PSPNet, Cityscapes): PGD3-AT 1.91, SegPGD3-AT 8.86
- Source: [ar5iv 2207.12391](https://ar5iv.labs.arxiv.org/html/2207.12391)

**DDC-AT (Xu et al., 2020).** Dynamic divide-and-conquer AT. Reports Cityscapes white-box and black-box robustness, and +2.7 / +1.9 clean mIoU over SAT for PSPNet / DeepLabv3 — [arXiv 2003.06555](https://arxiv.org/pdf/2003.06555). SEA later broke it at 2/255 (VOC) — [arXiv HTML](https://arxiv.org/html/2306.12941v2)

**PIR-AT (Croce et al. ECCV 2024), SEA-evaluated, acc / mIoU.**
- PASCAL-VOC:
  - PSPNet-50: clean 90.6 / 68.9; 4/255 81.5 / 47.7; 8/255 50.6 / 11.2
  - UPerNet ConvNeXt-T: clean 92.7 / 75.2; 4/255 88.6 / 64.9; 8/255 71.7 / 34.6
  - UPerNet ConvNeXt-S: 4/255 66.0 mIoU; 8/255 36.4 mIoU
- ADE20K:
  - UPerNet CN-T: clean 31.7; 4/255 17.2; 8/255 4.9 mIoU
- Clean-initialised AT of the same models (DDC-AT, AT on PSPNet-50 VOC, 3 steps) gives 0.0 mIoU at 4/255 under SEA.
- Source: [arXiv HTML](https://arxiv.org/html/2306.12941v2)

**Mao et al. ICML 2023 (Cityscapes, DRN-22-d, 4/255).**
- Adversarially trained model, ε_v = 2.5ε, mIoU:

  | | Vanilla | Random | Rotation | Contrastive | Invariance | Equivariance |
  |---|---|---|---|---|---|---|
  | Clean | 53.23 | 52.96 | 51.72 | 53.00 | 49.04 | 48.74 |
  | PGD | 26.61 | 27.04 | 27.68 | 28.14 | 27.74 | 29.65 |
  | Houdini | 23.47 | — | — | — | — | 29.80 |
  | AA | 26.61 | — | — | — | — | 29.63 |
  | BPDA | 26.61 | 27.00 | 26.37 | 28.50 | 23.31 | 29.83 |

- Clean model, ε_v = 1.5ε, mIoU:

  | | Vanilla | Random | Rotation | Contrastive | Invariance | Equivariance |
  |---|---|---|---|---|---|---|
  | Clean | 58.29 | 52.38 | 34.30 | 37.22 | 33.84 | 37.95 |
  | PGD | 1.31 | 1.47 | 13.20 | 8.44 | 14.49 | 30.76 |
  | BPDA | 1.31 | 1.47 | 6.20 | 4.78 | 6.82 | 11.64 |

- The adaptive attack (AA) adds an equivariance term with λe=1000.
- Stated costs: on COCO, 1.7 s vs 0.046 s per image and 10.36 GB vs 0.39 GB of GPU memory. The authors suggest detecting attacks first to avoid the clean-accuracy drop. That suggestion is an explicit pointer toward the gate idea.
- No full-gradient, transfer/black-box, or certified evaluations.
- Source: [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)

**Naseer et al. CVPR 2020.** Self-supervised input-space purifier, shown on classification, segmentation and detection — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.html)

**Detection rather than defense.**
- Spatial consistency (Xiao et al., ECCV 2018): random overlapping-patch consistency differs between adversarial and benign images. The detector's randomness is claimed to give robustness to adaptive attacks — [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/CHAOWEI_XIAO_Characterize_Adversarial_Examples_ECCV_2018_paper.html); [arXiv](https://arxiv.org/pdf/1810.05162)
- Entropy / uncertainty-based detection (Maag & Fischer, VISAPP 2024): a lightweight post-processing detector — [arXiv 2305.12825](https://arxiv.org/abs/2305.12825v1); [SciTePress](https://scitepress.org/PublishedPapers/2024/123035)

**Patch defenses.**
- Z-Mask (Rossolini et al., AAAI 2023): Z-score analysis of internal over-activation masks adversarial patch pixels. Covers segmentation and detection and claims robustness to defense-aware attacks — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26758); [arXiv](https://arxiv.org/pdf/2203.07341.pdf)
- PatchZero: detects the patch and repaints it with mean pixel values — [arXiv 2207.01795](https://arxiv.org/pdf/2207.01795v1)
- Demasked Smoothing (ICLR 2023): certified patch defense for any segmentation model without retraining — [ICLR](https://iclr.cc/virtual/2023/poster/11767)

**Certified (L2) segmentation.** Fischer, Baader, Vechev, ICML 2021, "Scalable Certified Segmentation via Randomized Smoothing": per-pixel abstention with multiple-testing correction. Evaluated on Cityscapes, Pascal Context and ShapeNet. Code: github.com/eth-sri/segmentation-smoothing — [PMLR](https://proceedings.mlr.press/v139/fischer21a.html); [arXiv](https://arxiv.com/abs/2107.00228)

**Diffusion purification.** Searches found only classification work (CIFAR / ImageNet):
- Heterogeneous purification (stronger noise on model-attended pixels): spatially varying purification strength, close in spirit to ε(p) — [arXiv 2503.01407](https://arxiv.org/abs/2503.01407v2)
- TARO 2026 — [arXiv 2605.08440](https://arxiv.org/pdf/2605.08440)
- REAL 2025 — [arXiv 2507.03427](https://arxiv.org/pdf/2507.03427)
- AGDM — [arXiv 2403.16067](https://arxiv.org/pdf/2403.16067)

### Inferences
- **Novelty status:** I found no published follow-up combining equivariance with segmentation defense after Mao 2023. There is no work using a dense equivariance-error map to gate or to set per-pixel budgets. The gate itself is close to prior detection work (spatial consistency, entropy) and to Mao's own "detect first" suggestion, so frame the gate as an application of a detector, not as novel by itself.
- **Novelty risk — per-pixel adaptive budget:** the closest prior art is in classification purification. "Heterogeneous purification" (2503.01407) varies noise strength per pixel based on model attention. Together with the pixel-reweighted AT / SuperPure / FaceGuard already noted in CLAUDE.md, it means the claim must stay narrow: "dense *equivariance-error*-driven per-pixel budget for *segmentation* test-time defense." Cite 2503.01407 explicitly.
- **Comparability:** Cityscapes numbers are not comparable across papers.
  - SegPGD reports full-res PSPNet / DeepLabv3 at 8/255. Mao reports DRN-22 at 680x340 and 4/255.
  - Our 4/255 adversarially trained numbers (~25–29) sit in the same range as SegPGD-AT's 8/255 PGD20 numbers (~25), but the attack strength, ε and resolution differ.
  - In the report, compare only against our own baselines plus the Mao tables, and cite SegPGD-AT / PIR-AT numbers only as context.
- **Croce 2022 risk already visible:** BPDA+defense (22.55) < undefended PGD (25.25) on the adversarially trained model. If this persists for the improved method, the honest headline is "the defense helps the clean model a lot (1.3 → ~29), but on the adversarially trained model it is not robust to BPDA." The gate could mitigate the clean-accuracy loss, but it cannot fix the BPDA loss.

### Gaps
- No Cityscapes robust-mIoU at 4/255 under SEA was found for any defense.
- No certified Cityscapes mIoU numbers were extracted from Fischer et al.; their table values were not retrieved.
- No segmentation-specific diffusion purification paper on Cityscapes was found. This may reflect search limits, not proof of absence.
- The SemSegBench numbers for Cityscapes adversarially trained models were not retrieved.

## Q3. Which cheap, novel, feasible extensions would most strengthen the project (4 weeks, 1 GPU, DRN-D-22 at 680x340)?

### Takeaway
Ranked by value per GPU-hour:
- (1) SegPGD + CosPGD + SEA-lite (APGD with MCE/JS losses) against the undefended model and via BPDA+EOT against the defense.
- (2) A gate-aware adaptive attack and an ε(p)-aware attack.
- (3) Per-class IoU and error-map vs. error AUROC analysis.
- (4) Compute-vs-robustness curves (defense steps, views, AMP).
- (5) An ε sweep (2/255, 4/255, 8/255).
- (6) Natural corruptions on a Cityscapes subset (Kamann & Rother style), to show the gate does not wrongly fire or hurt under benign shift.

Certification, patches, other datasets and other backbones are lower priority or risky in 4 weeks.

### Cited Findings
- **Corruption robustness benchmark:** Kamann & Rother, CVPR 2020 / IJCV 2021. About 400k corrupted images from Cityscapes, VOC and ADE20K (no official "Cityscapes-C" name). Segmentation robustness generally rises with clean accuracy — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Kamann_Benchmarking_the_Robustness_of_Semantic_Segmentation_Models_CVPR_2020_paper.html); [arXiv 1908.05005](https://arxiv.org/pdf/1908.05005)
- **Realistic shift benchmark:** Cityscape-Adverse (2024) uses diffusion editing for 8 adverse conditions (weather, lighting, season) with labels preserved — [arXiv 2411.00425](https://arxiv.org/html/2411.00425v1)
- **Compute cost is a known weakness:** Mao et al.'s defense costs ~37× time and ~26× memory (COCO) — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079). Croce 2022 flags extra inference compute without real gains as a core failure mode — [PMLR](https://proceedings.mlr.press/v162/croce22a.html)
- **Certification code:** randomized smoothing for segmentation (L2, per-pixel abstain) at github.com/eth-sri/segmentation-smoothing — [PMLR](https://proceedings.mlr.press/v139/fischer21a.html)
- **Patch defenses and code:** Z-Mask, PatchZero and Demasked Smoothing exist as patch baselines; patch attack code at retis-ai/semsegadvpatch — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26758); [GitHub](https://github.com/retis-ai/semsegadvpatch); [ICLR](https://iclr.cc/virtual/2023/poster/11767)
- **Entropy detection baseline:** a natural comparison for the equivariance-error gate — [arXiv 2305.12825](https://arxiv.org/abs/2305.12825v1)
- **Universal perturbations:** cheap to apply once computed (Cityscapes PSPNet 75.8 → 8.8 IoU) — [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Metzen_Universal_Adversarial_Perturbations_ICCV_2017_paper.pdf)

### Inferences (extension menu; cost estimates are my own and unverified)

| # | Extension | Value | Novelty vs. existing test-time seg defenses | 1-GPU cost (test100) | Risk |
|---|---|---|---|---|---|
| E1 | SegPGD + CosPGD on undefended model, then defense on the cached adversarial images (non-adaptive transfer to defense) | Reviewer-expected; Mao never ran them | First equivariance-defense eval with these attacks | ~same as current PGD-50 runs (hours) | Low |
| E2 | SEA-lite: APGD (momentum, step-size halving) with MCE, MCE-Bal, JS losses + radius reduction; 100–300 iter; worst case per image; via BPDA(+EOT) for defended | Strongest credible attack; "existing attacks overestimate" narrative | No test-time seg defense evaluated with SEA (none found) | 3× the current PGD cost; reuse nmndeep/robust-segmentation loss code | Med (porting to DRN/680x340; label 255 masking) |
| E3 | Gate-aware attack: max seg loss + λ·(−equivariance error), sweep λ; ε(p)-aware attack: weight loss by low-ε(p) pixels | Mandatory for gated/adaptive defense (Croce 2022) | Required to validate the novel claim | Similar to BPDA runs | Low; may reveal gate is bypassable, so report honestly |
| E4 | Gate-vs-detector comparison: AUROC of equivariance error vs. output entropy (Maag & Fischer) and spatial consistency (Xiao) | Positions the gate against prior detectors | Equivariance error as seg attack detector not found in the literature | Minutes–1 h (forward passes only) | Low |
| E5 | Per-class IoU under attack ± defense (19 classes); error-map vs. pixel-error correlation / pixel AUROC | Cheap, insightful, figure-rich; supports per-pixel ε(p) rationale | Not done by Mao | Free (from existing confusion matrices) | None |
| E6 | Compute-robustness Pareto: defense steps {1, 5, 10, 20}, views {3, 5, 7, 9}, AMP; plot mIoU vs. s/img and GB | Addresses known biggest weakness (37× time); supports P3 optimiser | Mao reported only one point | Several short runs on tune100 | Low |
| E7 | ε sweep {2, 4, 8}/255 for attack, defense ε_v scaled | Enables loose comparison with SegPGD-AT/PIR-AT 8/255 numbers | Mao only 4/255 | 2 extra attack runs per config | Low |
| E8 | Natural corruptions (Kamann-style: noise, blur, fog, brightness, JPEG; ~5 types × 2 severities on test100; generate with `imagecorruptions` pkg) — does the gate stay closed? Does defense hurt/help? | Shows gate does not destroy clean/benign-shift performance; practical for driving | Equivariance defense on corruptions not reported | Forward + defense per corruption (~hours) | Low |
| E9 | Universal perturbation / transfer black-box (craft on clean model, test on adv model + defense) | Covers black-box threat model Mao omitted | Minor | ~1–2 h | Low |
| E10 | Targeted / class-specific attack (e.g., make "person"/"car" vanish → road) | Safety-relevant story; tests whether ε(p) protects small objects | Minor | Same as PGD | Low–Med |
| E11 | Patch attack (EOT, retis-ai code) against gated defense | Different threat model; equivariance error likely high at patch → interesting map figure | Partially overlaps Z-Mask/PatchZero | Patch optimisation ~hours–1 day | Med (out-of-scope for L-inf defense; frame as qualitative) |
| E12 | Randomized-smoothing certification of the gated defense | Strong theoretical add | Stochastic test-time optimisation + smoothing is very expensive (n≈100–1000 samples × 20-step defense per image) | Days for even ~20 images | High: likely infeasible; at most certify the base model with Fischer code as context |
| E13 | Other backbone/dataset (VOC, ADE20K, robust PIR-AT models from Croce) | Generality | Generalisation claim | Need new checkpoints + data pipeline; 1+ week | High given deadline; maybe 1 small VOC experiment with a public PIR-AT checkpoint if time remains |

- **Recommended core package within 4 weeks:** E1, E2 (lite), E3, E4, E5, E6, E7. Then E8 if time allows. This covers the reviewer checklist (strong attacks + adaptive attacks + compute) cheaply, and E4/E5 back up the novelty claim directly.
- **On reporting:** follow Croce 2022 and SEA. Report the worst case over attacks per image, and compare with the static (no-defense) model under the same strongest attack. Robust mIoU from the paper's own AA/BPDA should not be the headline.

### Gaps
- GPU-hour estimates are my own extrapolation, not measured. Profile on tune100 first (P3).
- I could not confirm whether a packaged "Cityscapes-C" dataset with official splits exists. The `imagecorruptions` package used by Kamann & Rother is believed to be the standard tool but was not verified in this pass.
- No public PIR-AT Cityscapes checkpoint was confirmed. The SEA repo's models appear to target VOC and ADE20K.
