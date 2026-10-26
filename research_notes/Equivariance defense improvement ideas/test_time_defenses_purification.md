# Test-time / self-supervised adversarial defenses and input purification (2019–2026): transferable ideas for the equivariance defense

Scope note: research run on 2026-10-26. About 17 tool calls. Semantic Scholar's API returned HTTP 429 (rate-limited) and its web page returned 403, so the "cited by" list below comes from OpenAlex plus web search and is **incomplete**. Entries marked **[prior-knowledge link]** are well-known papers whose URL and claims come from background knowledge and were *not* re-fetched in this session. Verify them before citing them in the write-up.

## Q1. How do the main test-time defense families differ, and which ideas transfer?

### Takeaway
All test-time defenses in this family follow the same template: minimise a label-free self-supervised (SSL) loss, or a generative/energy prior, over the input before classifying. They differ in (a) the signal (contrastive, rotation, inpainting/MAE, equivariance, score/energy, a diffusion prior), (b) how strong the purification is (fixed vs. adaptive), and (c) how well they survive adaptive attacks. Under careful adaptive evaluation (APGD+BPDA+EoT, transfer from the static model), almost all of them collapse to roughly the static model's robustness or below. So the most transferable "improvements" are (i) evaluation rigour, (ii) an adaptive purification strength, and (iii) combining several objectives. A large generative prior is not one of them.

### Cited Findings
**Contrastive reverse attack (Mao et al., ICCV 2021, "Adversarial Attacks are Reversible with Natural Supervision")**
- Mechanism: a few steps of gradient descent on the input pixels to *minimise* a SimCLR-style contrastive loss. This finds a small additive vector that "repairs" the image's internal consistency. The motivation is that attacks aimed at the label "collaterally damage" self-supervised signals. — [arXiv 2103.14222](https://arxiv.org/pdf/2103.14222); [Columbia PDF](http://www.cs.columbia.edu/~junfeng/papers/attack-reverse-iccv21.pdf)
- Independent evaluation (Croce et al. 2022, CIFAR-10, ℓ∞ 8/255): reported 63.83% (worst-case CW; headline AutoAttack was 67.79%) → evaluated **58.4%**, *below* the static model's 59.4%. It was broken by APGD+BPDA+EoT (16 EoT samples). With a fixed seed and no EoT, the result was 56.4%. — [Croce et al. ICML 2022 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2202.13711)

**SOAP (Shi, Holtz, Mishne, ICLR 2021, "Online Adversarial Purification based on Self-Supervised Learning")**
- Mechanism: train the classifier jointly with an auxiliary SSL head (e.g., rotation / reconstruction / label-consistency). At test time, run gradient steps on the input to minimise that auxiliary loss. — [arXiv 2101.09387](https://arxiv.org/pdf/2101.09387); [ICLR 2021 poster](https://iclr.cc/virtual/2021/poster/2996)
- Reported 51.02% → evaluated **3.7%** (static model 0.0%). It was broken by APGD+BPDA, using gradients averaged over the *intermediate purification iterates*, with 1000 iterations. — [Croce et al. 2022](https://ar5iv.labs.arxiv.org/html/2202.13711)
- Difference from Mao 2023: SOAP *requires training* an SSL head. Equivariance needs no training, because it uses the model's own outputs under transforms.

**"Hedgehog" / HD = Wu et al. 2021, "Attacking Adversarial Attacks as a Defense"**
- Croce et al. call this defense **HD** (the "hedgehog" name used in the brief could not be confirmed). Mechanism: an "anti-attack" perturbation of the input before classification. Reported 65.70% → evaluated **61.0%** (static 63.0%). It was broken by transfer attacks plus APGD+BPDA+EoT. The default AutoAttack *returned unperturbed copies* (a false-robust artefact). — [Croce et al. 2022](https://ar5iv.labs.arxiv.org/html/2202.13711)
- No paper titled "Robustifying via Self-supervised Online Adversarial Purification" was found as a distinct work. It most likely refers to SOAP (above). Flagged as unresolved.

**Croce et al. 2022, "Evaluating the Adversarial Robustness of Adaptive Test-time Defenses" (ICML 2022)**
- Headline: "none significantly improve upon static defenses when subjected to our careful case study evaluation", and some defenses *degrade* the underlying static model while adding inference cost. — [arXiv 2202.13711](https://arxiv.org/abs/2202.13711)
- Defenses evaluated (reported → evaluated, CIFAR-10 ℓ∞ 8/255): SODEF/Kang (NeurIPS'21) 57.76→52.2; Close-loop control (Chen, ICLR'21) →5.6; HD (Wu) 65.70→61.0; Anti-adversaries (Alfarra, AAAI'22) 79.2→66.6 (RayS); SOAP 51.02→3.7; ATLD+IMF (Qian) 65.07→12.6; AID-purifier (Hwang, ICMLW'21) 52.65→43.8; Mao'21 reverse attack 63.83→58.4; ADP (Yoon, ICML'21, score-based) 69.71→**33.7** (APGD+EoT). — [Croce et al. 2022](https://ar5iv.labs.arxiv.org/html/2202.13711)
- Extended checklist: (1) transfer attacks from the static model, transferring *unsuccessful* adversarial perturbations rather than clean inputs; (2) black-box attacks (Square, RayS); (3) strong white-box attacks through the *full* defense (APGD with multiple losses and restarts); (4) BPDA with *different backward approximations, including intermediate iterates*; (5) EoT for randomness, or fix the seed; (6) always build defense-specific adaptive attacks. — [Croce et al. 2022](https://ar5iv.labs.arxiv.org/html/2202.13711)
- TTE and DENT were **not** in the content retrieved from Croce et al. (see Gaps).

**Test-time transformation ensembling (TTE; Pérez et al., ICCV-W 2021)** [prior-knowledge link]
- Mechanism: average predictions over flips/crops (no optimisation). It is the classification analogue of "prediction averaging over equivariance views". — [arXiv 2107.14110](https://arxiv.org/abs/2107.14110)

**Diffusion purification**
- DiffPure (Nie et al., ICML 2022): add noise up to a fixed timestep t*, then run reverse diffusion. Theory: the noise must be large enough to wash out the perturbation but small enough to keep label semantics. One global t* is used for all inputs. — [PMLR v162](https://proceedings.mlr.press/v162/nie22a/nie22a.pdf)
- UAD-RS (remote sensing): uses a pre-trained DDPM for purification on **both scene classification and semantic segmentation**, with an *adaptive noise-level selection* step. Reported to beat SOTA purification against 7 common perturbations on 4 remote-sensing datasets. — [ScienceDirect, JAG 2024](https://www.sciencedirect.com/science/article/pii/S1569843224004850)
- Faster variants: one-step consistency distillation (Lei et al., CVPR 2025, "Instant Adversarial Purification with Adversarial Consistency Distillation"). — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_Instant_Adversarial_Purification_with_Adversarial_Consistency_Distillation_CVPR_2025_paper.pdf). Also Purify++ (control of randomness). — [arXiv 2310.18762](https://arxiv.org/pdf/2310.18762)
- GDMP (Wang et al. 2022, guided diffusion for purification). [prior-knowledge link] — [arXiv 2205.14969](https://arxiv.org/abs/2205.14969)
- Lee & Kim (ICCV 2023), "Robust Evaluation of Diffusion-Based Adversarial Purification": DiffPure's robustness is overestimated under weak gradient approximations. [prior-knowledge link] — [arXiv 2303.09051](https://arxiv.org/abs/2303.09051)

**Energy / score-based purification**
- ADP (Yoon et al., ICML 2021, score-based purification with randomised denoising) → 33.7% under APGD+EoT vs. 69.71% reported. — [Croce et al. 2022](https://ar5iv.labs.arxiv.org/html/2202.13711); paper [arXiv 2106.06041](https://arxiv.org/abs/2106.06041) [prior-knowledge link]
- Hill et al. (ICLR 2021), "Stochastic Security": Langevin sampling under an EBM prior. [prior-knowledge link] — [arXiv 2005.13525](https://arxiv.org/abs/2005.13525)

**MAE-based repair (DRAM, Tsai et al. 2023, "Test-time Detection and Repair of Adversarial Samples via Masked Autoencoder")**
- Uses an MAE loss for both detection and repair. Repair improves robust accuracy by 6–41% (standard ResNet-50) and 3–8% (robust ResNet-50) over contrastive and rotation reverse attacks. Reports that "self-supervision tasks are not sensitive enough to detect adversarial attacks accurately" (i.e., contrastive/rotation losses are weak detectors). — [arXiv 2303.12848](https://arxiv.org/html/2303.12848)

### Inferences
- **Transferable idea A, intermediate-iterate BPDA:** Croce broke SOAP with BPDA on *averaged intermediate iterates*. The project's BPDA attack should add this variant. It is cheap: store the iterates the 20-step loop already produces.
- **Transferable idea B, transfer of unsuccessful perturbations from the static model:** a near-free attack baseline. Croce shows it beats naive AutoAttack on several defenses.
- **Transferable idea C, prediction averaging (TTE-style):** already planned (P5). TTE is the citation, and TTE alone is not novel.
- **Transferable idea D, a second SSL signal (MAE / contrastive) added to the equivariance loss:** see Q2. The best evidence for gains is modest.
- **Do not adopt a diffusion prior as the main contribution.** There is no pretrained Cityscapes-resolution diffusion model that is small enough. EoT through diffusion is expensive. Lee & Kim show that diffusion robustness is overestimated. Cost: high (weeks). Novelty risk: low for segmentation (only UAD-RS found), but evaluation risk is high.
- The measured "20 steps ≈ 1 effective step" problem is consistent with Croce's observation that randomness (noise injection) is what inflates many test-time defenses' numbers. The planned momentum + step-decay optimiser (APGD-style) is well motivated, but it may *lower* robustness under EoT if the gain was coming from noise. Report both.

### Gaps
- Croce et al.'s results for TTE and DENT were not in the retrieved section; verify in the PDF (the PDF could not be text-extracted here because poppler/pypdf are unavailable).
- Mao et al. 2023's own BPDA/adaptive numbers for segmentation were not retrieved in this session (only the claim that invariance baselines drop 4 points under BPDA). — [PMLR mao23d](https://proceedings.mlr.press/v202/mao23d/mao23d.pdf)
- The "Hedgehog" defense could not be identified as a distinct paper.

## Q2. Are multi-objective SSL purification losses (equivariance + contrastive + entropy/consistency) used, and do they help?

### Takeaway
Yes, but there is little evidence that they help. Lawhon, Mao & Yang (2022) combine contrastive + rotation + inpainting in a reverse attack and gain only +0.3 points robust / +1.1 points clean over single-task contrastive on CIFAR-10, with no dedicated adaptive-attack test. Mao et al. 2023 itself already argues that *dense* equivariance beats single "sparse" constraints. Adding entropy minimisation is untested for purification of segmentation, as far as was found.

### Cited Findings
- Lawhon, Mao, Yang (arXiv 2204.03714, 2022; no venue listed), "Using Multiple Self-Supervised Tasks Improves Model Robustness". The loss is L = L_contrastive + L_rotation + L_inpaint, with each task's step scaled by its share of L. Contrastive runs every iteration; rotation and inpainting alternate on even/odd iterations. — [arXiv 2204.03714](https://arxiv.org/html/2204.03714v1)
- Results (CIFAR-10, Carmon et al. backbone): none 63.9% attacked / 89.7% clean; single SSL 65.3 / 86.6; multi-task 65.6 / 87.7 (SE ≤ 0.4%). — [arXiv 2204.03714](https://arxiv.org/html/2204.03714v1)
- There is no dedicated adaptive-attack experiment. The authors argue that a sum of objectives forces a white-box attacker to balance them (an argument, not a measurement). — [arXiv 2204.03714](https://arxiv.org/html/2204.03714v1)
- Mao et al. 2023 frame prior work as "limited to a single constraint at inference time", and equivariance as a *dense* constraint holding for all natural images. — [arXiv 2212.06079](https://arxiv.org/abs/2212.06079); [PMLR](https://proceedings.mlr.press/v202/mao23d/mao23d.pdf)
- DRAM: an MAE loss beats contrastive/rotation for repair by 6–41% on ResNet-50. — [arXiv 2303.12848](https://arxiv.org/html/2303.12848)
- Related (not adversarial): equivariant test-time training uses an equivariance-consistency objective for imaging inverse problems. — [arXiv 2411.05771](https://arxiv.org/html/2411.05771)

### Inferences
- **Idea E, equivariance + prediction-entropy (or view-agreement KL) term:** cheap (~1 day; it reuses the existing views' logits). Novelty risk: medium. Entropy minimisation is TENT-style test-time adaptation, but as an *input-purification* term for segmentation combined with dense equivariance, no prior work was found. Expect a small gain (≈1 point by analogy to Lawhon et al.), and entropy minimisation can *reinforce* a wrong adversarial prediction. Must be checked against adaptive attacks.
- **Idea F, MAE/inpainting loss on the segmentation backbone:** requires training an MAE head. Cost: medium–high. It is out of scope for 4 weeks unless a pretrained MAE is used. Novelty risk: medium.

### Gaps
- No paper was found that combines equivariance with contrastive or entropy losses for segmentation purification, and no ablation was found that shows multi-objective gains under BPDA/EoT.

## Q3. Which purification methods adapt strength per sample or per pixel? (Novelty check for per-pixel ε(p) from dense equivariance error)

### Takeaway
*Per-sample* adaptive purification strength is now an active line of work (SSNI, ICML 2025; DGAP 2026; UAD-RS; MANI-Pure). The closest *per-region* method is "Divide and Conquer" (Pei et al. 2025). It uses a classifier-attention mask to inject strong diffusion noise inside the mask and weak noise outside, for **classification only**. None of these uses equivariance error, works with a PGD-style budget, or targets semantic segmentation. The novelty claim survives if it is worded narrowly. The image-level gate idea (defend or not) has close prior art in DGAP.

### Cited Findings
- **SSNI (Sun, Zhang, Ye, Xiao, Liu; ICML 2025 poster, arXiv 2506.06027):** a pre-trained score network estimates each sample's deviation from clean data (score norm), and a reweighting function maps it to a per-sample noise level t*. Cleaner inputs get less noise, which helps clean accuracy. Gains are reported on CIFAR-10 and ImageNet-1K when SSNI is plugged into existing diffusion purifiers. — [arXiv 2506.06027](https://arxiv.org/abs/2506.06027); [ICML 2025 poster](https://icml.cc/virtual/2025/poster/46361); [OpenReview](https://openreview.net/forum?id=KzokzKV4JK)
- **DGAP (arXiv 2610.10752, Oct 2026, audio deepfake detection):** a light forward-reverse diffusion probe, with the detector-score shift used as an adversarial indicator. Low-shift inputs pass *unchanged*, and flagged inputs get stronger purification. This is a *gate + adaptive strength* design in a different modality. — [arXiv 2610.10752](https://arxiv.org/html/2610.10752v1)
- **Divide and Conquer / heterogeneous noise (Pei et al., arXiv 2503.01407, 2025):** per-block classifier attention maps are thresholded (τ≈0.8) and unioned into a mask M. The noisy input is x(t_l)⊙M + x(t_s)⊙(1−M), with t_s < t_l, and masked regions are regenerated inpainting-style. CIFAR-10 WRN-28-10: 93.16% clean, 59.38% PGD+EOT, 80.45% AutoAttack. ImageNet RN-50 PGD+EOT: 60.06%. **No segmentation.** — [arXiv 2503.01407](https://arxiv.org/html/2503.01407)
- **UAD-RS:** adaptive noise-level selection for diffusion purification, evaluated on remote-sensing segmentation. It is per-image, not per-pixel, and not equivariance-based. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569843224004850)
- **MANI-Pure (arXiv 2509.25082):** magnitude-adaptive noise injection for purification. The method details were not retrieved. — [arXiv 2509.25082](https://arxiv.org/pdf/2509.25082)
- **SuperPure:** pixel-wise adversarial *masking* plus GAN super-resolution for patch attacks. It iterates down-up sampling and masking until few new pixels get masked. — [arXiv 2505.16318](https://arxiv.org/html/2505.16318)
- **Patch-defense masks (Segment-and-Complete CVPR 2022; Z-Mask AAAI 2023):** binary localisation of patches using segmentation or internal-feature Z-scores. These target patches, not ℓ∞ budgets. — [SAC CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Segment_and_Complete_Defending_Object_Detectors_Against_Adversarial_Patch_Attacks_CVPR_2022_paper.pdf); [Z-Mask](https://ojs.aaai.org/index.php/AAAI/article/download/26758/26530)
- Uncertainty-based *detection* of attacks on segmentation (Cityscapes, Pascal VOC; DeepLabv3+, BiSeNet, DDRNet, HRNet) uses per-pixel output uncertainty. It detects attacks but does not purify. — [arXiv 2305.12825](https://arxiv.org/pdf/2305.12825)
- DRAM reports that SSL losses are "not sensitive enough to detect adversarial attacks accurately". This is a direct risk for the P2 go/no-go AUROC check. — [arXiv 2303.12848](https://arxiv.org/html/2303.12848)

### Inferences
- **Safe novelty wording:** "first test-time defense for semantic segmentation that uses the dense equivariance error both to gate the defense and to set a per-pixel ℓ∞ purification budget ε(p)." Cite SSNI, DGAP, and UAD-RS as per-sample adaptive purification. Cite Pei et al. 2025 as region-adaptive diffusion noise for classification. Cite SuperPure/SAC as pixel masks for patches. Do **not** claim "first adaptive purification strength" or "first detect-then-purify gate".
- **Idea G, an SSNI-style continuous reweighting function** (map the image-level mean equivariance error to the global ε through a monotone function, instead of a hard gate): ~1 day. It gives a smooth version of the gate and a cleaner ablation. Novelty risk: low in this setting, but cite SSNI.
- **Idea H, attention/saliency-masked budget (Pei-style) as a baseline for ε(p):** replace the equivariance map with model-attention or output-entropy maps. ~1 day. This is *essential* as an ablation to show that equivariance error is the right per-pixel signal.
- **Idea I, combine per-pixel uncertainty (entropy) and equivariance error:** cheap. It positions the work against arXiv 2305.12825.
- Gate-aware adaptive attack (P6): DGAP-style gates are attackable by keeping the indicator low. The P6 attack should explicitly add a term that minimises the equivariance error on the adversarial input (a "stealth" term).

### Gaps
- MANI-Pure's mechanism was not retrieved; check that it is not per-pixel.
- Pixel-reweighted adversarial training (2024) and FaceGuard were not re-verified in this session (they were already known to the project).
- No search found any per-pixel ℓ∞ budget for *PGD-style* (non-generative) purification. Absence in a ~17-call search is not proof; run one more Scholar query ("spatially varying epsilon purification").

## Q4. Which works cite Mao et al. 2023 (arXiv 2212.06079) and extend or attack it?

### Takeaway
The citation footprint looks small. OpenAlex lists only **3** citing works for the record it matched, and none of them attacks the defense. Web search found no paper that breaks or adaptively evaluates the equivariance defense, and no follow-up that applies it to Cityscapes segmentation. This is an opportunity (the project's BPDA weakness finding is itself new evidence) and a risk (an incomplete citation list).

### Cited Findings
- OpenAlex `cites:` query (count = 3): "A geometric approach to robust medical image segmentation" (Medical Image Analysis, 2024); "Landscape Learning for Neural Network Inversion" (ICCV 2023); "Towards Adversarial Robustness and Backdoor Mitigation in SSL" (ICASSP 2025). — [OpenAlex API](https://api.openalex.org/works?search=Robust%20Perception%20through%20Equivariance) (queried 2026-10-26; OpenAlex may have matched only one version of the paper, so the count is likely an undercount)
- The awesome-test-time-adaptation list classifies it as "Equ-Defense" under instance-level attack/defense, next to Convolutional Visual Prompt (NeurIPS 2023, same group). — [GitHub list](https://github.com/tim-learn/awesome-test-time-adaptation/blob/main/TTA-TTBA.md)
- Related test-time defenses from 2023–2025 surfaced alongside it (citation relation not verified): Robust Feature Inference via spectral projections (arXiv 2307.11672); Interpretability-Guided Test-Time Adversarial Defense (ECCV 2024, arXiv 2409.15190); Test-time Adversarial Defense with Opposite Adversarial Path (arXiv 2410.16805); Stochastic Resonance of Latent Ensembles (arXiv 2510.03224); ATAC, augmentation-based test-time adversarial correction for CLIP (arXiv 2511.17362). — [2307.11672](https://arxiv.org/pdf/2307.11672); [2409.15190](https://arxiv.org/pdf/2409.15190); [2410.16805](https://arxiv.org/pdf/2410.16805); [2510.03224](https://arxiv.org/html/2510.03224); [2511.17362](https://arxiv.org/pdf/2511.17362)
- One Cityscapes robustness study (CNN/ViT/SAM) states that it lacks a test-time defense module and lists one as future work. This suggests that test-time defense for segmentation is still open. — [arXiv 2408.09839](https://arxiv.org/html/2408.09839v1)
- Mao et al. 2023 report that invariance-based baselines drop 4 points under BPDA, which they read as evidence of obfuscated gradients, and that their defense still improves robustness under two adaptive attacks. — [PMLR mao23d](https://proceedings.mlr.press/v202/mao23d/mao23d.pdf)

### Inferences
- The project's finding that the defense is weaker under BPDA than claimed appears to be **unreported in the literature**. Present it as a contribution, using Croce's checklist (intermediate-iterate BPDA, transfer of unsuccessful perturbations, EoT).
- Convolutional Visual Prompt (Tsai, Mao et al., NeurIPS 2023) is the closest same-group extension. It learns a small convolutional prompt at test time with an SSL loss. Cite it as related work. [prior-knowledge link] — [arXiv 2303.00198](https://arxiv.org/abs/2303.00198)

### Gaps
- Semantic Scholar's "cited by" list could not be retrieved (HTTP 429/403). Retry with an API key or by hand. Google Scholar likely shows more citations (expected tens, not 3). Each of the five 2023–2025 test-time defenses listed above needs a check for whether it actually cites Mao 2023.

---

## Consolidated idea list (cost on 1 GPU, novelty risk)

| # | Idea | Source | Applied to seg / with equivariance? | Cost | Novelty risk |
|---|---|---|---|---|---|
| A | BPDA on averaged intermediate iterates + transfer of unsuccessful static-model perturbations + Square black-box | [Croce 2022](https://ar5iv.labs.arxiv.org/html/2202.13711) | No / No | 1–2 days | Evaluation, not a novelty claim; strengthens the paper |
| B | APGD-style optimiser (momentum, step-size halving, checkpointing) for the *defense* | APGD [arXiv 2003.01690](https://arxiv.org/abs/2003.01690) [prior-knowledge link] | Not for equivariance purification (none found) | 1–2 days | Low as a component; cite APGD/MI-FGSM |
| C | TTE-style prediction averaging over views | [TTE arXiv 2107.14110](https://arxiv.org/abs/2107.14110) [prior-knowledge link] | Seg: standard TTA; not novel | <1 day | High if claimed as novel; supporting part only |
| D | Equivariance + entropy / view-agreement KL purification loss | [Lawhon 2022](https://arxiv.org/html/2204.03714v1) (multi-SSL) | Not seg; not with equivariance | ~1 day | Medium; expect ≤1 point |
| E | MAE/inpainting loss term | [DRAM 2023](https://arxiv.org/html/2303.12848) | No seg | 1–2 weeks (needs MAE head) | Medium; likely out of scope |
| F | Continuous SSNI-style mapping from mean equivariance error to global ε (soft gate) | [SSNI ICML 2025](https://arxiv.org/abs/2506.06027) | No seg; score norm, not equivariance | ~1 day | Low in this setting if SSNI is cited |
| G | Detect-then-purify gate with pass-through | [DGAP 2026](https://arxiv.org/html/2610.10752v1) | Audio only | Already planned (P4) | Gate alone is **not** novel; combine with per-pixel ε(p) |
| H | Attention/entropy-masked budget as an ablation of ε(p) | [Pei 2025](https://arxiv.org/html/2503.01407); [2305.12825](https://arxiv.org/pdf/2305.12825) | Classification / seg-detection | ~1 day | Needed to support the claim |
| I | "Stealth" gate-aware attack that also minimises the equivariance error | Croce checklist item 6 | — | 1–2 days | Evaluation |
| J | Diffusion purification baseline (DiffPure/consistency one-step) | [DiffPure](https://proceedings.mlr.press/v162/nie22a/nie22a.pdf); [Lei CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_Instant_Adversarial_Purification_with_Adversarial_Consistency_Distillation_CVPR_2025_paper.pdf); [UAD-RS](https://www.sciencedirect.com/science/article/pii/S1569843224004850) | UAD-RS on remote-sensing seg | High (no small Cityscapes-res pretrained model; EoT cost) | Skip, or mention only |
