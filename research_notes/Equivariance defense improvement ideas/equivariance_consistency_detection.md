# Equivariance / Consistency-Based Adversarial Detection and Dense Signals (2018–2026)

Scope: transformation consistency, equivariance error and TTA disagreement used as adversarial/OOD detectors, especially per pixel in segmentation; how the maps become weights; prior art for "dense equivariance error → per-pixel defense budget"; adaptive attacks on consistency detectors. Research budget: ~17 tool calls. Numbers come from the fetched papers unless marked otherwise.

## Q1. Which papers detect adversarial examples via transformation consistency / equivariance / uncertainty, and what AUROCs do they report?

### Takeaway
Segmentation has strong precedent for image-level consistency detection. Xiao et al. (ECCV 2018) used DRN on Cityscapes, the same model family as this project, and got AUC ≈100%. Their spatial-consistency detector stayed near 100% under their adaptive attack, but their blur/scale-consistency detector fell to 0% AUC. Mao et al. (the reproduced paper) already report that attacks lower the equivariance score on Cityscapes (0.694 clean → 0.448 attacked). They also propose a detect-then-defend threshold in their appendix. So an image-level gate is not novel on its own and should be framed as a component. No paper I found reports pixel-level AUROC for separating attacked from clean pixels using equivariance error.

### Cited Findings
**Segmentation-specific**
- Xiao, Deng, Li, Yu, Liu, Song, "Characterizing Adversarial Examples Based on Spatial Consistency Information for Semantic Segmentation", ECCV 2018. Method: draw two random 512×512 patches with a corner offset of 32–64 px, segment each, and compute mIoU between the predictions on the overlap. Average over K pairs and threshold at the lower bound of benign scores. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1810.05162), [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/CHAOWEI_XIAO_Characterize_Adversarial_Examples_ECCV_2018_paper.html)
- Xiao et al. Table 1 (Cityscapes; DRN and DLA; DAG and Houdini attacks with "Pure"/"Kitty" targets): spatial consistency with K=1 gives AUC 91–98%, and K≥5 gives 100%, both non-adaptive and adaptive. The scale-consistency baseline (Gaussian blur, std 0.5/3/5) gives 95–100% non-adaptive. Under an adaptive attack it falls to 0% at std 3 and 5 on the Kitty target for DRN. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1810.05162)
- Xiao et al.'s adaptive attack on the spatial detector: the attacker picks K random patches, the same K the defender uses, and attacks the full image plus those patches. AUC stays near 100%; the authors attribute this to the large patch/overlap search space. The attack on scale consistency added a conv layer in front of the model (Carlini–Wagner style) with an L2 bound of 0.06. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1810.05162)
- Xiao et al. also found that segmentation adversarial examples barely transfer between models. — [arXiv](https://arxiv.org/pdf/1810.05162)
- Maag & Fischer, "Uncertainty-Based Detection of Adversarial Attacks in Semantic Segmentation", VISAPP 2024. Extended version: Maag, Resner, Fischer, arXiv 2408.10021 (2024). Pixel-level entropy, variation ratio (1 − max prob) and probability margin are averaged into |C|+3 image-level features. A "Heatmap" variant feeds the full entropy map to a two-layer CNN. — [SciTePress](https://scitepress.org/PublishedPapers/2024/123035), [arXiv html](https://arxiv.org/html/2408.10021v1)
- Maag et al. results: targeted class-deletion attacks get AUROC 80–100. Untargeted I-FGSM is detected poorly. PGD-targeted and SSMM are detected worse than ALMA/DAG. Average optimal detection accuracy is 89.36%. There is **no adaptive-attack evaluation and no per-pixel localization metric**; entropy heatmaps only show qualitatively that attacked regions are more uncertain. — [arXiv html](https://arxiv.org/html/2408.10021v1)
- Chen, Monteuuis, Petit, "Multi-Task Consistency-based Detection of Adversarial Attacks", arXiv 2608.07750 (Aug 2026). Inconsistency between the outputs of different tasks (object detection and instance segmentation) gives ROC-AUC 99.9% against PGD on BDD100k "within the considered attacker model". The abstract mentions no adaptive attack and no pixel-level use. — [arXiv](https://arxiv.org/abs/2608.07750)
- Lis et al., "Detecting the Unexpected via Image Resynthesis" (ICCV 2019). Resynthesis-discrepancy maps for anomaly segmentation, also evaluated against DAG/Houdini attacks. I did not retrieve its numbers. — [arXiv](https://arxiv.org/pdf/1904.07595)

**From the reproduced paper (Mao et al., ICML 2023)**
- Equivariance score (higher means more equivariant) for clean / attacked / restored images. Cityscapes: 0.694 / 0.448 / 0.713. PASCAL VOC: 0.900 / 0.642 / 0.921. COCO: 0.901 / 0.774 / 0.914. ImageNet: 0.539 / 0.538 / 0.581, i.e. **almost no separation on classification**. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Appendix A.2: output-level equivariance variance is used for anomaly/corruption detection, with AUROC over 15 corruptions and dataset shift (Tables 9–10). Appendix A.3.1 proposes a **detect-then-defend threshold on the equivariance score to preserve clean accuracy**. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)

**Classification (transferable ideas)**
- Tian, Yang, Cai, "Detecting Adversarial Examples Through Image Transformation", AAAI 2018. Rotation/shift prediction change detects ~99% of oblivious attacks; a randomised white-box version detects ~70%. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11828)
- Feature squeezing (Xu et al., NDSS 2018) compares softmax outputs after color-depth reduction and median smoothing. As summarised by He et al., it reaches 99.74% accuracy on a balanced set for static FGSM/JSMA. — [He et al. arXiv 1706.04701](https://arxiv.org/pdf/1706.04701)
- ATAC (arXiv 2511.17362, 2025, CLIP) reports that augmentation-induced latent drift separates clean from adversarial inputs at AUC ≈1. I have not verified its setup. — [arXiv](https://arxiv.org/pdf/2511.17362)
- Wang et al., aleatoric uncertainty from test-time augmentation for medical segmentation (2018). TTA disagreement tracks segmentation error, differently from test-time dropout. This is the clean/OOD setting, not adversarial. — [arXiv 1807.07356](https://arxiv.org/pdf/1807.07356)

### Inferences
- Detection on Cityscapes is easy for non-adaptive attacks: several signals give AUROC ≈1 at image level. An image-level gate AUROC near 1 on PGD is therefore expected and will not impress a reviewer by itself. The more informative go/no-go numbers are **pixel-level AUROC** and **gate AUROC under the gate-aware adaptive attack**.
- Mao's ImageNet result (no score drop) is a warning: separation can depend on the task. Dense prediction gives many constraints, one per pixel; classification gives few. That favours segmentation.
- Untargeted PGD may be harder to detect than targeted attacks (Maag: I-FGSM untargeted is detected poorly). The P2 go/no-go should cover PGD, SegPGD and a low-ε attack, not only PGD-50 at 4/255.

### Gaps
- I found no paper reporting **pixel-level AUROC** of an equivariance/consistency map for attacked versus clean pixels in segmentation. Under global L∞ attacks every pixel is perturbed, so the "attacked pixel" label probably has to be defined as **pixels whose prediction was flipped by the attack**. This is a methodological choice the project must state.
- I did not retrieve per-attack AUROC from Maag et al. (it is only in figures) or detection numbers from Lis et al.

## Q2. Has anyone used a dense consistency/equivariance error map to modulate a defense spatially? (Novelty check)

### Takeaway
I found no prior work that uses a dense **equivariance/consistency error map** to set a **per-pixel purification budget or step size in a test-time defense for semantic segmentation**. The narrow novelty claim survives this search. The closest works are spatially varying defenses driven by other signals (saliency/attention, reconstruction error, pixel importance during training), and image-level gating (Mao's own appendix, DGAP, IntensPure). Novelty risk: **moderate for "per-pixel budget from equivariance error in segmentation"**, **high for "equivariance-based gate / detect-then-defend"** (Mao A.3.1 already proposes it), and **high for "view averaging"** (TTA ensembling, Perez et al. 2021).

### Cited Findings
- Pei et al., "Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification", arXiv 2503.01407 (2025). Diffusion purification applies **higher-intensity noise to pixels the target model focuses on** (interpretability-based) and lower intensity elsewhere. It appears to be classification. It is the closest "per-pixel purification strength" prior art, but the signal is attribution, not equivariance, and the task is not segmentation. — [arXiv](https://arxiv.org/abs/2503.01407)
- SuperPure (arXiv 2505.16318, 2025) masks pixels with high super-resolution reconstruction error, iterating until few new pixels get masked. It targets patches in classification and uses a per-pixel reconstruction error signal. — [arXiv](https://arxiv.org/html/2505.16318)
- DGAP (arXiv 2610.10752, 2026, audio deepfake) gates purification strength per input using a detector. IntensPure (IJCAI 2024, person re-ID) scales diffusion strength with estimated attack intensity. Both adapt per input, not per pixel. — [DGAP](https://arxiv.org/html/2610.10752v1), [IntensPure](https://www.ijcai.org/proceedings/2024/0106.pdf)
- "Interpretability-Guided Test-Time Adversarial Defense" (arXiv 2409.15190, 2024) is guided at the neuron level, not per pixel. It notes that test-time defenses cost roughly 8–500× a forward pass and often break under adaptive attacks. — [arXiv](https://arxiv.org/pdf/2409.15190)
- LiDAR range-view segmentation purification (arXiv 2508.19290, 2025) is segmentation-specific purification, but not per-pixel adaptive. Seen only through a secondary site. — [pith.science](https://pith.science/paper/2508.19290)
- Rethinking Uncertainty in Segmentation (arXiv 2604.13262, 2026) combines uncertainty with confidence to rank pixels and **defer/abstain** on the most uncertain ones, with large error reduction. It covers clean retinal segmentation, not adversarial inputs. It is relevant to a per-pixel abstention extension. — [arXiv](https://arxiv.org/html/2604.13262v1)
- Perez et al., "Enhancing Adversarial Robustness via Test-Time Transformation Ensembling", ICCVW 2021. Ensembling over flips/crops improves robustness (classification). This is prior art for the "view averaging" component. — [CVF](https://openaccess.thecvf.com/content/ICCV2021W/AROW/papers/Perez_Enhancing_Adversarial_Robustness_via_Test-Time_Transformation_Ensembling_ICCVW_2021_paper.pdf)
- PATCHOUT (Neural Processing Letters, 2025) uses semantic consistency against a coarse segmentation to detect and localize adversarial **patches**. It is a localization map, but not used to modulate a defense. — [Springer](https://link.springer.com/article/10.1007/s11063-025-11775-5)
- Pixel-reweighted adversarial training (Zhang et al., ICML 2024, "Improving Accuracy-robustness Trade-off via Pixel Reweighted Adversarial Training") already exists per the project's CLAUDE.md. It is training-time, classification, with an attribution-based per-pixel ε. I did not re-fetch it in this session.

### Inferences
- Safe wording: "To our knowledge, the first test-time defense for semantic segmentation that uses the dense equivariance error map to set a per-pixel purification budget." Cite Pei et al. 2025 (per-pixel noise intensity from attribution, diffusion, classification), SuperPure, pixel-reweighted AT, Xiao 2018 and Mao A.3.1 as the closest works, and state the differences explicitly.
- Do not claim novelty for the gate itself. Present the gate as an efficiency and clean-accuracy device, with Mao A.3.1 as precedent, and claim novelty only for the per-pixel ε(p) map and its gate-aware evaluation.
- Re-run a Scholar check for "pixel-wise" + "test-time" + "segmentation" + "purification" close to submission, since 2026 arXiv activity is high.

### Gaps
- My search coverage of 2025–2026 arXiv was limited (about 17 calls). I could not confirm that Pei et al. 2025 is classification only.

## Q3. Which transformations give the most discriminative equivariance signal?

### Takeaway
For defense strength on Cityscapes, Mao's ablation ranks resize > flip > small rotation ≫ large rotation. I found no study that ranks transformations by **detection** AUROC in segmentation. Xiao's results suggest that **spatial shift/crop consistency** (translation equivariance with random offsets) is both highly discriminative and resistant to adaptive attacks. Low-pass/blur consistency is discriminative but easily bypassed adaptively.

### Cited Findings
- Mao et al. Table 6 (Cityscapes, standard model; robust mIoU with equivariance vs invariance loss): Flip 20.50 vs 9.56; Resize 26.00 vs 9.90; Rotation ≤15° 17.03 vs 9.75; Rotation ≥90° 8.61 vs 9.60. Large rotations hurt because segmentation models are not equivariant to them even on clean data. Robustness grows with the number of constraints (Fig. 3). ColorJitter is not ablated alone. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Xiao et al.: random-offset patch (translation/crop) consistency gives 100% AUC (K≥5) that holds under their adaptive attack. Gaussian-blur consistency gives 100% non-adaptive but 0% under an adaptive attack at std 3–5. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1810.05162)
- Tian et al.: rotation/shift for classification; randomising the transformation was needed for white-box robustness (~70% detection). — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11828)

### Inferences
- A cheap, well-motivated addition: **random crop/translation views with random offsets** (Xiao-style) as extra equivariance views for the error map, and possibly for the defense. Randomness over a large view space is what defeated the adaptive attacker in Xiao's study. Feasible in about 2–3 days: crops are invertible on the overlap, so the error is computed only on overlapping pixels.
- Ablate the error map per transformation family (flip, rescale, crop-shift, ColorJitter) in P2, using pixel-level AUROC on tune100. This is cheap (forward passes only) and is a publishable mini-result, since no such table exists for detection.
- Non-invertible transforms (ColorJitter) provide only an invariance-type constraint at the output. Check whether including them dilutes the per-pixel map.

### Gaps
- No source compares transformations by adversarial detection AUROC in segmentation.

## Q4. Feature-level vs output-level equivariance; error metric (KL vs L2 vs feature-space)

### Takeaway
Mao's defense uses **dense feature-level** equivariance, but their detection appendix uses **output-level** equivariance variance, and there is no layer ablation. I found no comparison of KL vs L2 vs feature-space error for adversarial detection in segmentation. Uncertainty work (Maag) shows that output entropy, variation ratio and margin each carry signal and can be combined.

### Cited Findings
- Mao et al.: the defense measures equivariance on the dense feature representation; detection (A.2.1, Eq. 19) uses output-level equivariance; there is no layer ablation. Theorem 3.2: robustness bounds increase with the number of informative constraints. Equivariance gives one constraint per feature location, versus one per objective for invariance. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Zhang et al., "An Adaptive View of Adversarial Robustness from Test-time Smoothing Defense" (arXiv 1911.11881). Test-time smoothing of inputs and intermediate-layer outputs. It is precedent for intervening at intermediate layers. — [arXiv](https://arxiv.org/pdf/1911.11881)
- Singh et al., "Fast Adaptive Test-Time Defense with Robust Features" (2023). Projects onto a robust feature subspace. — [CISPA](https://cispa.de/en/research/publications/71788-fast-adaptive-test-time-defense-with-robust-features)
- Maag et al.: three pixel-level uncertainty measures (entropy, variation ratio, probability margin) combined into image-level features. A CNN on the entropy heatmap is another option. — [arXiv html](https://arxiv.org/html/2408.10021v1)

### Inferences
- A cheap P2 experiment: compute the per-pixel error map in four ways (L2 on softmax, symmetric KL on softmax, label-flip rate across views, cosine distance on the penultimate feature map upsampled to label size) and compare pixel AUROC on tune100. This takes about 1–2 days, forward passes only.
- Multi-scale feature-level maps (e.g., DRN stages 5–8) could be fused as a sum of normalised errors, but each extra layer adds hyperparameters. Do this only if the output-level map fails the go/no-go.
- Combine the equivariance error with entropy (Maag) into a 2-feature logistic regression per pixel, fit on tune100. This gives a calibrated probability that a pixel is attacked, usable as a weight. The entropy is free to compute.

### Gaps
- No direct evidence on which layer gives the most discriminative signal for adversarial pixels.

## Q5. Calibrating maps into per-pixel weights

### Takeaway
I found no established method for turning a dense consistency map into a per-pixel ε for adversarial defense. Existing works threshold image-level scores using benign statistics (Xiao: lower bound of benign scores; Maag: a threshold τ), or fit a small classifier (Maag's heatmap CNN, Maag's |C|+3 feature logistic model). Pei et al. use attribution to split pixels into high and low noise.

### Cited Findings
- Xiao: threshold set at the lower bound of benign normalised mIoU. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1810.05162)
- Maag: image-level threshold τ; feature-based classifier; heatmap CNN trained on one attack and tested on others. — [arXiv html](https://arxiv.org/html/2408.10021v1)
- Pei et al. 2025: a two-level split (high noise on attended pixels, low noise elsewhere). — [arXiv](https://arxiv.org/abs/2503.01407)

### Inferences (design suggestions, not sourced results)
- Simple, defensible mapping: compute the per-pixel percentile rank of the error relative to the **clean tune100 distribution**. Then set ε(p) = ε_min + (ε_max − ε_min)·σ((r(p) − τ)/T), smoothed with a Gaussian or average-pool kernel. The image is 8×-stride anyway, so a 9–17 px kernel avoids salt-and-pepper budgets. Tune only τ and T on tune100.
- Fallback if pixel AUROC is weak but image AUROC is strong: a two-level budget (Pei-style split) or gate only.

## Q6. Adaptive attacks against consistency-based detectors

### Takeaway
Consistency detectors with a **deterministic, low-dimensional transformation** (blur, squeezing) are reliably broken by adaptive attacks: Xiao's blur detector fell to 0% AUC, and He et al. broke feature squeezing. Detectors with **randomised, large transformation spaces** held up better: Xiao's spatial detector stayed near 100%, and Tian's randomised version kept ~70%. Mao's adaptive attack, which adds λ·equivariance loss, barely changed robust accuracy. The general lesson from Tramèr et al. is that bespoke adaptive attacks broke 13 published defenses, so the gate-aware attack must be designed carefully.

### Cited Findings
- Xiao et al.: an adaptive attack (conv layer in front of the model plus the blur stds) took scale-consistency AUC to 0% (DRN, Kitty target, std 3/5). Spatial consistency remained 92–100% under an adaptive attack using the same K patches. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1810.05162)
- He, Wei, Chen, Carlini, Song, "Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong", USENIX WOOT 2017. An adaptive adversary bypasses feature squeezing and its combinations with low distortion. The attack optimises misclassification + distortion + detection score jointly. — [arXiv](https://arxiv.org/abs/1706.04701)
- Tian et al.: a white-box attack against randomised transformation detection leaves ~70% detection. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11828)
- Mao et al.: an adaptive attack maximising task loss + λe·equivariance loss (λe=1000 strongest on Cityscapes) lowered robust accuracy by only ~0.2 points on the adversarially trained Cityscapes model. Under BPDA the robust mIoU was 29.83 on the adversarially trained Cityscapes model and 11.64 on the standard one. The authors argue that invariance (but not equivariance) relies on obfuscated gradients. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Tramèr, Carlini, Brendel, Mądry, "On Adaptive Attacks to Adversarial Example Defenses", NeurIPS 2020: 13 defenses circumvented. No single attack strategy suffices; attacks must be tailored. — [NeurIPS](https://papers.nips.cc/paper/2020/hash/11f38f8ecd71867b42433548d1078e38-Abstract.html)

### Inferences
- Gate-aware attack for P6: minimise equivariance error (EOT over the defender's random views), so the gate stays closed and ε(p) stays small, while maximising the task loss. Run it at several λ values and report the whole curve. Also report the attacker's success at **evading the gate** (gate AUROC under attack) separately from final mIoU.
- Expect a trade-off: Mao's equivariance score fell from 0.694 to 0.448 under plain attack, and making an attack equivariance-consistent should cost attack strength. That trade-off is the argument for the method's robustness, and it is exactly what reviewers will ask for.
- Randomise the gate views (random crop offsets and scales at test time) to enlarge the space the attacker must cover, following Xiao and Tian. Evaluate with EOT.

### Gaps
- No published adaptive attack specifically targets a **per-pixel** consistency-modulated defense; the project's P6 attack would be the first such evaluation.

## Q7. Concrete further improvements: ranked by feasibility (1 GPU, ~4 weeks)

Design suggestions synthesised from the findings above; feasibility estimates are my own.
1. **Pixel-level go/no-go protocol + per-transformation and per-metric ablation** (L2/KL/flip-rate/feature-cosine × flip/resize/crop/ColorJitter). Forward passes only, 1–3 days. Novel as an analysis; no such table found.
2. **Random-offset crop views** (Xiao 2018) added to the error map and the gate. Low cost, and their adaptive-robust track record is the strongest available. 2–3 days.
3. **Equivariance error + entropy fused per pixel** (logistic regression on tune100) → calibrated ε(p). 2 days. Cite Maag et al.
4. **Per-pixel abstention**: output "ignore" (label 255) for pixels whose post-defense error stays above a threshold. Report risk–coverage / mIoU at fixed coverage. Cheap, 2–3 days, and prior art exists only on clean segmentation (arXiv 2604.13262).
5. **Selective re-prediction / per-pixel fusion**: in high-error pixels, use the view-averaged prediction; elsewhere use the single-view prediction. Cheap; overlaps with P5. Novelty is low on its own (TTA ensembling, Perez 2021).
6. Multi-scale feature-level equivariance map. Medium cost (hooks on DRN stages); attempt only if the output map fails.
7. Learned transformation selection. Not recommended in 4 weeks: it requires a training loop and increases adaptive-attack surface.
