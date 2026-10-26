# Adaptive-attack evaluation of test-time / purification defenses, and making iterative purification robust and efficient (2018–2026)

Scope: what a credible evaluation of the Mao et al. (ICML 2023) equivariance defense (DRN-D-22, Cityscapes, 680×340) plus the planned gate / per-pixel budget ε(p) / faster optimiser / view-averaging extension must include, and which purification-side changes have survived adaptive attacks. Search budget was ~15 tool calls, so some claims rest on abstracts or ar5iv extractions (flagged where relevant).

---

## Q1. What attack protocol should the project use to be credible, and which sanity checks show gradients are not obfuscated?

### Takeaway
Use the Croce et al. (ICML 2022) six-point checklist, adapted to segmentation. That means: transfer attacks from the undefended model; a black-box score-based attack (Square); a strong white-box attack on the full defense (APGD/PGD with several losses and restarts, or SEA for segmentation); BPDA with gradients averaged over intermediate iterates where needed; EOT for randomness; and a defense-specific adaptive attack. For stochastic defenses, report the worst case over repeated submissions. Raw PGD+EOT was stronger than AutoAttack on purifiers (Lee & Kim 2023). The headline number should be the minimum over all of these attacks.

### Cited Findings
**Foundational papers**
- Athalye, Carlini & Wagner, "Obfuscated Gradients Give a False Sense of Security", ICML 2018 (best paper). 7 of 9 non-certified ICLR-2018 white-box defenses relied on obfuscated gradients; the authors fully circumvented 6 and partially circumvented 1. They introduced BPDA (for shattered gradients), EOT (for randomised defenses) and reparameterisation (for vanishing/exploding gradients). — [arXiv 1802.00420](https://arxiv.org/abs/1802.00420v4); [code](https://github.com/anishathalye/obfuscated-gradients)
- Tramèr, Carlini, Brendel & Mądry, "On Adaptive Attacks to Adversarial Example Defenses", NeurIPS 2020. They circumvented 13 defenses from ICLR/ICML/NeurIPS whose original evaluations already used adaptive attacks. The paper is a methodology tutorial: some strategies generalise, but no single attack strategy works against every defense, so attacks must be tailored. — [arXiv 2002.08347](https://arxiv.org/pdf/2002.08347); [NeurIPS](https://papers.nips.cc/paper/2020/hash/11f38f8ecd71867b42433548d1078e38-Abstract.html)
- Carlini & Wagner, "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods", AISec 2017. All 10 detectors (from 7 papers) were defeated with new loss functions, in some cases without knowing the detector's parameters. — [arXiv 1705.07263](https://arxiv.org/pdf/1705.07263)

**Croce, Gowal, Brunner, Shelhamer, Hein, Cemgil, "Evaluating the Adversarial Robustness of Adaptive Test-time Defenses", ICML 2022 (PMLR 162:4421–4435)**
- Conclusion: none of the evaluated adaptive test-time defenses significantly beat their static counterparts under careful evaluation. Some weakened the static model, and all of them increased inference cost (2×–518×). — [PMLR](https://proceedings.mlr.press/v162/croce22a.html); [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.13711)
- Reported vs. re-evaluated robust accuracy (CIFAR-10, ℓ∞ 8/255): SODEF 57.76→52.2; Hedge Defense 65.70→61.0; Anti-adversaries 79.2→66.6; **SOAP 51.02→3.7 (518× cost)**; ATLD+IMF 65.07→12.6; AID-purifier 52.65→43.8; **Mao et al. 2021 (the same group's earlier self-supervised "reversal" defense) 63.83→58.4 (407× cost)**; ADP (Yoon 2021, Langevin/EBM) 69.71→33.7. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- Their recommendations, in order: (1) transfer attacks from the static model and other surrogates (e.g., RobustBench models), transferring the *unsuccessful* perturbations rather than clean points; (2) black-box attacks, Square (score-based) and RayS (decision-based), to confirm the defense beats the static model; (3) strong white-box attacks on the full defense, with APGD over multiple losses and restarts found reliable; (4) when gradients are missing or vanishing, APGD+BPDA with several backward approximations, including intermediate iterates; (5) EOT for explicit or implicit randomness, or fix the seed; (6) defense-specific adaptive attacks that are at least as strong as the non-adaptive ones. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- How SOAP was broken: APGD+BPDA on cross-entropy, averaging gradients over the **intermediate purification iterates** rather than only the final purified image, with 1000 iterations. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- For Mao et al. 2021, 16 EOT iterations gave 58.4%, while fixing the seed (no EOT) gave 56.4%. Fixing the seed can therefore be the stronger choice. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- Pitfall: AutoAttack returns the clean image when it fails, so transfer evaluations built on AutoAttack outputs can be inflated. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)

**Lee & Kim, "Robust Evaluation of Diffusion-Based Adversarial Purification", ICCV 2023 (oral)**
- Recommends **PGD+EOT over AutoAttack** for purifiers. DiffPure scored 46.84% under PGD+EOT vs. 63.60% under AutoAttack; ADP scored 33.48% vs. 59.53%. All attacks used 20 EOT samples. — [ar5iv 2303.09051](https://ar5iv.labs.arxiv.org/html/2303.09051); [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Robust_Evaluation_of_Diffusion-Based_Adversarial_Purification_ICCV_2023_paper.html); [code](https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification)
- Back-propagate through the full defense where possible, and otherwise use a **surrogate process** rather than the adjoint method. DiffPure scored 74.38% with the adjoint gradient vs. 46.84% with the surrogate; GDMP scored 75.59% under BPDA vs. 24.06% under the surrogate. BPDA therefore badly overestimated GDMP's robustness. — [ar5iv 2303.09051](https://ar5iv.labs.arxiv.org/html/2303.09051)

**Stochastic-defense evaluation**
- DiffHammer (Wang, Fu, Han, Xiang, NeurIPS 2024): EOT's global gradient averaging causes a "gradient dilemma", and single-evaluation reporting underestimates the risk that an attacker simply resubmits. They propose a selective attack on vulnerable purification samples plus **N-evaluation** (worst case over N runs of the stochastic defense). Effective in 10–30 iterations. — [mlanthology](https://mlanthology.org/neurips/2024/wang2024neurips-diffhammer)
- Liu et al., "Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective" (ICLR 2025): the intrinsic stochasticity is the main source of DBP robustness. DiffPure PGD-20 accuracy falls from 69.04% (EOT-1) to 55.96% (EOT-10). — [arXiv 2404.14309](https://arxiv.org/pdf/2404.14309)
- Controlling for randomness, the net CIFAR-10 gain from diffusion purification drops to 23.7%, vs. ~70% (46% with EOT) under full stochasticity. — [arXiv 2505.22839](https://arxiv.org/html/2505.22839v1)

**Segmentation-specific attacks**
- Croce, Singh & Hein, "Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models", ECCV 2024. **SEA** = worst case over 300-iteration red-ε APGD runs on three losses (MCE, MCE-Bal, JS). The worst-case mIoU over attacks is approximated greedily (5–10 rounds). SegPGD and CosPGD can significantly overestimate robustness; SEA improves on them by >17.2% accuracy and >10.3% mIoU. — [arXiv 2306.12941](https://arxiv.org/pdf/2306.12941); [code](https://github.com/nmndeep/robust-segmentation)

**What Mao et al. 2023 actually evaluated (Cityscapes, DRN-D-22, ℓ∞ 4/255)**
- Clean (non-adv-trained) model, vanilla → equivariance mIoU: clean 58.29→37.95; PGD 1.31→30.76; MIM 1.40→30.10; Houdini 0.00→30.52; AutoAttack 1.31→30.28; **BPDA 1.31→11.64**. BPDA used 10 outer steps × 20 inner reversal steps. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Adversarially trained model: equivariance 29.65 (PGD), 29.63 (AA), 29.83 (BPDA). — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Their "Adaptive Attack I" (Mao et al. 2021 style) adds λ_e × equivariance loss to the attack objective, with λ_e=1000 found most effective. The ar5iv extraction says "maximises task loss plus λ_e·equivariance loss", so check the sign in the paper. The logical attack *minimises* equivariance error. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Defense details: signed-gradient steps with Gaussian noise added to the normalised gradient, noise annealed linearly to zero (SGLD-style, MAP not sampling); step size η=2ε_v; T=20. The appendix reports random sampling was "ineffective". Runtime is ~1.7 s vs. 0.046 s per image. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)

### Inferences
- **Recommended protocol** (each value is the minimum mIoU over attacks; 3 seeds, mean±std; worst case over N=3–5 resubmissions as in DiffHammer):
  1. Clean and undefended PGD-50 baselines (already have).
  2. **Transfer**: PGD/SEA examples crafted on undefended DRN-D-22, then fed to the defense (the Mao table already implies this is the weakest attack).
  3. **BPDA+EOT** (identity backward), EOT 8–20 samples. Add a "**BPDA-through-iterates**" variant that averages the attack gradient over the predictions on all 20 purifier iterates, as Croce did to break SOAP.
  4. **Full-gradient / surrogate** attack through the unrolled purifier (see Q4 for why `sign()` makes this subtle).
  5. **Fixed-seed** attack: freeze RNG and attack the now-deterministic defense. Croce found this beat EOT on Mao 2021.
  6. **Square attack** (score-based black-box) on a small subset (e.g., 20 images, ~1000 queries, ℓ∞ 4/255), adapted to segmentation by using mean per-pixel CE as the score. It is a sanity check, not a headline number.
  7. **Gate-aware adaptive attack** (Q2).
  8. Optionally, SEA losses (MCE/JS) inside the APGD loop instead of plain CE, since SegPGD alone overestimates mIoU robustness (Croce 2024).
- **Sanity checks to report** (Athalye/Carlini checklist): attack success rises monotonically with steps (plot mIoU vs. 10/50/100/200 steps); success rises with ε and reaches ~0 mIoU at large ε (e.g., 32/255); white-box beats transfer and black-box; random-noise perturbations of the same ε do little; EOT samples 1→8→20 should not help a deterministic defense (and should help a random one); BPDA ≥ full gradient would indicate full-gradient masking.
- The project has already found that BPDA brings the clean-model defense close to the paper's 11.64. Credible headline robustness for the clean-model setting is therefore BPDA-level, not PGD-level. Any improvement claim must be made under the strongest attack.

### Gaps
- No published adaptive-attack evaluation of Mao et al. 2023 by a third party was found (no 2024–2026 follow-up on segmentation test-time defense under adaptive attack surfaced).
- Square attack adaptation to segmentation: no published reference protocol found; the above is a suggestion.
- The exact sign/form of Mao's "Adaptive Attack I" could not be verified from the extraction.

---

## Q2. How should a gated (detector + defense) system be attacked?

### Takeaway
Attack the composed system end-to-end with a loss that jointly fools the task and keeps the detector/gate score below threshold (Carlini & Wagner 2017 style). The stop-gradient used *inside* the defense must not be applied by the attacker: the attacker should differentiate through the error map, or through a smooth surrogate of the gate.

### Cited Findings
- Carlini & Wagner 2017 defeated all 10 detectors by building new loss functions that target the combined classifier+detector, including in limited-knowledge settings. — [arXiv 1705.07263](https://arxiv.org/pdf/1705.07263)
- Tramèr et al. 2020 recommend attacks that are simple, aligned with the actual decision of the full system, and that replace non-differentiable components (e.g., thresholds) with differentiable surrogates. No single strategy works for every defense. — [arXiv 2002.08347](https://arxiv.org/pdf/2002.08347)
- Mao et al. 2023's own adaptive attack adds λ_e·(equivariance loss) to the task loss (λ_e=1000 best on Cityscapes clean model). They argue this creates a "lose-lose situation" because reducing equivariance error weakens the attack. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Mao et al. 2023 propose detect-first (only defend when flagged) to reduce clean-accuracy loss and runtime. This is the same structure as the planned gate, so the planned gate is not itself novel. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)
- DiffHammer's "selective attack" targets the vulnerable subset of stochastic purification outcomes, which is the analogue of attacking whichever branch (defend/skip) is weaker. — [mlanthology](https://mlanthology.org/neurips/2024/wang2024neurips-diffhammer)

### Inferences
- **Gate-aware attack recipe:** minimise L = −CE_seg(f(D(x′)), y) + λ·max(0, ē(x′) − τ + κ), where ē is the image-level equivariance score (mean of the dense map), with **gradients flowing through ē**. Grid-search λ ∈ {0.1, 1, 10, 100, 1000} and margin κ. Report the best λ. Two attacker goals must both be tried:
  - **(a) Evade the gate:** keep ē below τ so the defense never runs. Undefended DRN-D-22 is then fooled outright, so success is measured on the undefended path.
  - **(b) Survive the defense:** let the gate fire, but shape ε(p) by keeping the *local* error low on the pixels/classes being attacked. The per-pixel budget then stays small exactly where the attack lives. This is the specific risk of per-pixel ε(p).
- Hard threshold: use a sigmoid surrogate σ((ē−τ)/T) in the backward pass (BPDA-style) and evaluate with the real hard gate.
- Report the **gate's TPR/FPR under the adaptive attack**, not just AUROC on non-adaptive PGD. Carlini & Wagner's lesson is that detector AUROC on non-adaptive attacks is close to meaningless.
- Expect a trade-off curve: the attack can drive ē down at some cost in attack strength (Mao's "lose-lose" claim). Plotting attack mIoU vs. ē for several λ is the convincing figure for examiners.

### Gaps
- No paper was found that attacks a dense, per-pixel-budgeted purifier. This is novel territory (which also means no reference attack exists).

---

## Q3. Which purification-side changes survived adaptive attacks, and is restart selection or loss-guided early stopping novel?

### Takeaway
Few purification tricks survive strong adaptive evaluation. Stochasticity gives real but EOT-reducible gains. Ensembling several purification runs plus a noise schedule (Lee & Kim) held up under their PGD+EOT surrogate attack. Self-supervised test-time optimisation (SOAP, Mao 2021) lost most or some of its claimed robustness. **Selecting the purification output with the lowest self-supervised loss is already published (SOAP, 11-ε sweep) and was broken (51%→3.7%)**, so restart selection is high novelty risk. Loss-guided early stopping had no direct prior found, but it is a weak contribution and attackable.

### Cited Findings
- **SOAP (Shi, Holtz, Mishne, ICLR 2021):** test-time input optimisation on a self-supervised auxiliary loss (reconstruction / rotation / label consistency) within a budget, using 5 PGD steps without random init. **It runs the purification for 11 values of ε_d and picks the one with the lowest auxiliary loss.** Croce 2022 reduced it from 51.02% to 3.7% (and to 0.0% for its static model). — [arXiv 2101.09387](https://arxiv.org/pdf/2101.09387); [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- **Mao et al. 2021** (self-supervised contrastive "reversal", 40 steps, random): 63.83% claimed vs. 58.4% found. That is a modest drop, the most robust survivor among self-supervised purifiers in Croce's table, but it cost 407×. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- **Hedge Defense** (Wu et al. 2021, 20 steps, random init): 65.70% vs. 61.0%. **Anti-adversaries** (Alfarra 2022, 2 steps): 79.2% vs. 66.6%. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- **Langevin / EBM purification:** Hill, Mitchell & Zhu (ICLR 2021) use long-run MCMC with an EBM and an EOT defense (averaging predictions over stochastic purifications) that they argue resolves theoretical ambiguity for stochastic defenses. — [arXiv 2005.13525](https://arxiv.org/abs/2005.13525v2). ADP (Yoon et al. 2021, Langevin-type) fell from 69.71% to 33.7% under Croce's evaluation. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)
- **Ensembling + noise schedule (Lee & Kim 2023):** "gradual noise-scheduling" ({30×4, 50×2, 125×2} forward steps over 8 purification steps) **plus an ensemble of 10 runs**. Under their PGD+EOT surrogate attack: CIFAR-10 WRN-28-10 55.82% vs. DiffPure 46.84%; ImageNet 42.15% vs. 38.71%. — [ar5iv 2303.09051](https://ar5iv.labs.arxiv.org/html/2303.09051)
- **Stochasticity:** robustness of diffusion purification comes mainly from intrinsic randomness. EOT-10 cuts it substantially (69.04→55.96), so randomness helps but is partly attackable. — [arXiv 2404.14309](https://arxiv.org/pdf/2404.14309); [arXiv 2505.22839](https://arxiv.org/html/2505.22839v1)
- **DiffAttack (Kang, Song, Li, NeurIPS 2023):** adds a deviated-reconstruction loss at intermediate purification steps and segment-wise forward/backward for memory (>10× saving, constant in chain length). It reduced robust accuracy by >20% (CIFAR-10) and >10% (ImageNet) vs. prior SOTA attacks. A *moderate* diffusion length was most robust. — [arXiv 2311.16124](https://arxiv.org/pdf/2311.16124)
- Mao et al. 2023 report that random sampling (posterior sampling instead of MAP) was "ineffective" in their appendix, and they do no ensembling or restarts. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)
- Early stopping: a targeted search found no work that stops purification by monitoring the self-supervised loss or sets the step count per input. SOAP and diffusion purifiers use fixed schedules. A summary page claims SOAP observed an "elbow" in the auxiliary-loss curve that indicates when adversarial noise is removed (secondary source, unverified). — [search: SOAP summary pages](https://liner.com/review/online-adversarial-purification-based-on-selfsupervised-learning)

### Inferences
- **Novelty risk, by idea:**
  - **Restart selection by lowest equivariance loss: HIGH risk.** SOAP is the direct precedent (selection over ε rather than random starts, but the same "argmin self-supervised loss" principle), and it was broken. The argmin also gives the attacker a target: drive one restart's output to have low equivariance error and wrong labels (DiffHammer-style selective attack). If used, frame it as an ablation, not the contribution.
  - **Ensembling / averaging purified outputs or views: LOW novelty** (Hill 2021 EOT defense; Lee & Kim 10-run ensemble). Fine as a cited supporting part, as the plan already says.
  - **Momentum + step decay in the purifier: LOW novelty.** It is an optimiser swap (MI-FGSM/APGD), so cite it. Its value is fixing the measured one-step collapse, which is an empirical finding worth reporting.
  - **Loss-guided early stopping: MEDIUM risk** (no prior found, but trivial in concept). A data-dependent stop is another non-differentiable selection, so attack it with fixed-seed full-gradient or BPDA-through-iterates.
  - **Dense-error-driven per-pixel budget ε(p): no prior found in this search** (consistent with the project's earlier novelty check), and it is the right thing to lead with.
- **Fixing the one-step collapse:** the measured issue (noise swamping the normalised gradient until the last step) suggests (i) anneal the noise faster or scale it relative to the gradient norm, (ii) use momentum so signal accumulates across steps, and (iii) the DiffAttack/Lee-Kim finding that a *moderate* chain is most robust. More effective steps are not automatically better, so sweep steps on tune100.
- **Raising BPDA robustness:** the only levers with evidence behind them are (a) genuine randomness that EOT has to average out (partial), (b) ensembling several stochastic purifications, and (c) larger/more varied transformations. For (c), Mao note that large rotations (≥90°) hurt because segmentation isn't equivariant to them, which bounds the transformation set. — [ar5iv 2212.06079](https://ar5iv.labs.arxiv.org/html/2212.06079)

### Gaps
- No paper was found that tests momentum/APGD-style optimisers *inside* a purifier under adaptive attack.
- No direct evidence found on "larger random transformation sets" improving BPDA robustness for equivariance/invariance purifiers specifically.

---

## Q4. Practical memory and time tricks for full-gradient attacks on unrolled defenses

### Takeaway
Options are gradient checkpointing, segment-wise backward (DiffAttack), shorter or larger-step surrogate chains (Lee & Kim), and AMP. For *this* defense there is a key structural fact: the update uses `sign(∇)`, whose derivative is zero almost everywhere. Exact autograd through the unrolled purifier therefore reduces to the straight-through path times the projection/clip masks. The "full gradient" is nearly BPDA, and it is cheap. A meaningful full-gradient attack needs a *soft surrogate* (e.g., tanh(g/τ) or raw normalised gradient) to expose the second-order path.

### Cited Findings
- Lee & Kim 2023: about 20 denoising steps was the most that fit in 40 GB for full backprop. One experiment (PGD-200, 20 EOT, 30 function calls) took nearly 10 days on one A100. Their remedy is a **surrogate process** (fewer steps with larger step size, back-propagated directly), which proved stronger than the adjoint. — [ar5iv 2303.09051](https://ar5iv.labs.arxiv.org/html/2303.09051)
- DiffAttack's segment-wise forwarding-backwarding keeps memory constant in chain length (>10× saving). — [arXiv 2311.16124](https://arxiv.org/pdf/2311.16124)
- Chen et al. (2023) used gradient checkpointing to compute full gradients through diffusion purification. ADBM's authors argue that evaluation used too few iterations or EOT samples, so checkpointing helps only if attack strength is kept adequate. — [ADBM arXiv 2408.00315](https://arxiv.org/pdf/2408.00315)
- The adjoint method saves memory but not time, and its accuracy depends on the solver. It gave over-optimistic robustness on DiffPure (74.38% vs. 46.84%) and on SODEF (57.76% adjoint vs. 49.28% full backprop). — [ar5iv 2303.09051](https://ar5iv.labs.arxiv.org/html/2303.09051)
- DiffHammer's gradient grafting gives efficient attacks within 10–30 iterations. — [mlanthology](https://mlanthology.org/neurips/2024/wang2024neurips-diffhammer)
- Croce 2022 attacked Hedge Defense (20 purification steps) with a 5-step version. Attacking a shortened purifier is an accepted surrogate. — [ar5iv 2202.13711](https://ar5iv.labs.arxiv.org/html/2202.13711)

### Inferences (not measured; verify with `torch.cuda.max_memory_allocated`)
- **Why sign matters:** x_{t+1} = Π_ε(x_t − η·sign(∇_x L_eq(x_t) + noise)). Since ∂sign/∂x = 0 a.e., autograd gives ∂x_{t+1}/∂x_t = diag(clip-mask). The exact gradient of the 20-step defense is thus BPDA-identity times projection masks. It is *not* stronger than BPDA, and an examiner may call it gradient masking. To probe the true sensitivity, run a **soft-surrogate full-gradient attack**: replace sign with tanh(g/τ) (τ≈ median|g|) or with the normalised gradient g/‖g‖∞, then double-backprop (`create_graph=True`). This mirrors Lee & Kim's surrogate-process idea.
- **Memory of the soft surrogate:** each inner step needs forward + backward over K views (7–9) with `create_graph=True`, so the graph holds ~2–3× forward activations per view per step. Unrolling 20 steps × 8 views at 680×340 is likely tens of GB, beyond a single consumer GPU. Practical options:
  1. Surrogate with **T′=2–5 inner steps** and step size scaled ×(20/T′). Given finding (a) that the real defense acts like ~1 step, a 1–3 step surrogate is arguably faithful, which is a strong argument to make in the write-up.
  2. `torch.utils.checkpoint` per inner step, recomputing forwards in the backward pass (~1.3–2× time, memory ~O(1 step)).
  3. Subsample views per attack iteration (EOT over view subsets).
  4. AMP (bf16/fp16 autocast) for forward passes. Keep the loss and gradient sign in fp32. Check the attack isn't weakened by underflow (compare fp32 vs. AMP mIoU on 10 images).
  5. Batch size 1, evaluation on test100 only, with full500 reserved for the final best attack.
- **Time budget:** at ~1.7 s per defended forward (Mao), PGD-50 with EOT-8 through the defense is ≈50×8×(defense+backward) ≈ 15–30 min per image. Run it on a 20–50 image subset and report it as such, with BPDA/transfer on test100/full500. Croce's and Lee-Kim's reported runtimes show this compromise is standard.

### Gaps
- No published memory numbers for unrolling equivariance-style purifiers on DRN-D-22. The estimates above must be measured.
- No paper found that explicitly discusses the sign()-induced zero second-order gradient in sign-gradient purifiers. The argument follows from calculus but should be shown empirically: the unrolled-autograd gradient and the BPDA gradient should be ~identical (cosine ≈ 1).
