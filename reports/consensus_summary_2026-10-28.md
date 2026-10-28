# Consensus AI literature summary (pasted by the user on 2026-10-28)

Kept verbatim as the source for `reports/plan_additions_consensus_2026-10-28.md`. The citations are as the tool gave them and have **not** been verified; check each against the paper before using it.

---

Novel Improvements for Equivariance Defense
The attached paper's core idea is strong: restore dense feature equivariance at test time to repair adversarially corrupted predictions, and it improves robustness across classification, semantic segmentation, and instance segmentation without retraining the base model [Mao 2023]. The best improvements are not "add more transformations" in the abstract, but making the constraints more informative, the optimization more trustworthy, and the evaluation much harder; the literature suggests that speedups or prompt-based additions alone are less novel unless they solve those three issues directly [Mao 2023; Croce 2022].

Base Paper Assessment
The attached paper argues that dense equivariance constraints are better than single self-supervised constraints because they give fine-grained feature supervision and can be applied at inference time without training an auxiliary predictor [Mao 2023]. It also gives a theoretical claim that more informative constraints should tighten accuracy bounds under attack, provided each added constraint contributes conditional information about the label [Mao 2023].

The paper's empirical strengths are breadth and task diversity: it reports gains on ImageNet, Cityscapes, PASCAL VOC, and COCO, including up to 15-point robust-accuracy improvements and up to 21 maskAP points for instance segmentation [Mao 2023]. It also shows that not all transformations are equally useful: flip and resize help more than large rotations, and rotations above 90° can hurt because the underlying segmentation models are not equivariant to them even on clean data [Mao 2023].

The attached paper also exposes its own practical weakness: the defense is expensive, about 40× slower on attacked COCO images, uses much more memory than feed-forward inference, and reduces clean accuracy unless the defense is gated or the perturbation budget is kept small [Mao 2023].

Strongest Novel Directions
A high-value improvement is adaptive transformation selection, because the base paper currently uses a fixed hand-designed set of transformations and already shows that some are much more useful than others [Mao 2023]. Test-time augmentation work outside adversarial defense has already shown that instance-specific transform choice can improve robustness over fixed augmentation sets, either by predicting which transforms will be useful or by learning per-sample policies [Kim 2020; Mittal 2026].

A more novel version would not just select transforms by classifier confidence; it would select the subset that maximizes independent equivariance evidence while minimizing redundancy. The attached paper's own theorem says gains depend on added information, not raw count [Mao 2023]. That makes "constraint diversity" a sharper contribution than simply scaling K, especially because nearby transforms can generate correlated gradients and repeated evidence rather than new evidence.

A second strong direction is uncertainty- or region-weighted equivariance repair. The attached paper optimizes a global equivariance objective, but later TTA work shows that adaptation is safer when it focuses on reliable or informative samples, regions, or uncertainties instead of updating uniformly [Tan 2024; Wang 2024]. For segmentation, this suggests weighting the equivariance loss toward spatial regions where transformed predictions disagree most, or where model uncertainty is high but reducible.

A third strong direction is held-out equivariance validation for stopping and defense gating. The base paper already suggests detect-then-defend and shows a clean/robustness trade-off controlled by the repair budget [Mao 2023]. A novel improvement would split transformations into optimization and validation sets, then stop repair when validation equivariance stops improving; this directly targets the known failure mode of test-time methods that over-adapt or degrade in-distribution performance when hyperparameters are mismatched [Boudiaf 2022].

Faster Variants Worth Pursuing
A practical improvement is to amortize the repair step. Across test-time purification work, iterative optimization is consistently the computational bottleneck, and newer papers increasingly replace long inner loops with one-step or lightweight learned modules [Croce 2022; Wang 2025; Singh 2023]. The attached paper's optimization takes 20 steps and explicitly uses SGLD-style noise, which helps robustness but is slow [Mao 2023].

The most publishable speedup would be a learned initializer or one-step proposal for the repair vector, followed by a few refinement steps under the original equivariance loss. That is more novel than fully replacing the method with generic diffusion purification, because it preserves the paper's central equivariance mechanism while addressing its main deployment weakness [Mao 2023; Lee 2023; Bi 2024].

A related variant is feature-space repair instead of input-space repair. Several later defenses argue that pixel-space purification can leave residual adversarial bias in deep features or logits, and dual-domain methods that constrain both pixel and feature spaces improve robustness-quality trade-offs [Wang 2025; Su 2026]. For this paper, that suggests updating intermediate features to restore equivariance rather than perturbing pixels alone.

Harder Evaluation You Should Add
The biggest methodological gap is evaluation. Broader studies of adaptive test-time defenses found that many robustness gains shrink or disappear under stronger white-box evaluation, especially when attacks target the full defense, use BPDA/EoT correctly, or exploit transfer from stronger surrogates [Croce 2022]. AutoAttack-style parameter-free evaluation also repeatedly lowers reported robust accuracy relative to original claims across defenses [Croce 2020].

The attached paper did evaluate adaptive attack and BPDA, which is a real strength [Mao 2023]. But later work shows that purification robustness depends strongly on attack-side hyperparameters, denoising-step matching, and whether gradients are taken through the actual stochastic or iterative process [Lee 2023; Wesego 2026].

For segmentation specifically, adding stronger task-aware attacks is important because segmentation can fail under attacks that exploit feature diversity and structured output weaknesses, not only standard pixel-space PGD variants [Xiao 2023]. A publishable improvement paper should therefore claim robustness only after attacking the full defense with APGD/BPDA/EoT and segmentation-specific objectives.

Novel Ideas Most Likely to Matter
The most promising new paper would combine four elements: adaptive transform selection, uncertainty-weighted dense equivariance, held-out validation for early stopping, and amortized initialization for fast repair. That package is well aligned with the attached paper's mechanism and directly addresses its known weaknesses of fixed constraints, occasional clean-accuracy loss, and high runtime [Mao 2023].

Less novel directions are simply adding more self-supervised tasks, adding generic diffusion purification, or claiming architectural equivariance alone solves the problem. Multi-task or diffusion methods already exist, and the broader symmetry literature is explicit that equivariance alone does not guarantee adversarial robustness or clean-performance preservation [Mao 2023; Nie 2022; Krumtunger 2026; +1 more].

A more ambitious frontier is certified or task-aware equivariant robustness. Recent work argues that robustness for equivariant tasks should be defined relative to the task symmetry, and that equivariance-preserving randomized smoothing can provide sound guarantees for such settings [Schuchardt 2023]. Bringing even partial certification to test-time equivariance repair would be genuinely novel because the current paper is entirely empirical.

Best Thesis-Style Project
If the goal is one strong, realistic extension, the best bet is: Reliability-Guided Adaptive Equivariance Repair. The method would choose a small, diverse subset of transforms per image, weight the equivariance loss by uncertainty or disagreement maps, initialize the repair with a learned one-step proposal, and stop using held-out transforms. This stays faithful to the attached paper's central idea while solving the three weaknesses that the literature keeps exposing: redundant constraints, negative clean-image adaptation, and prohibitive test-time cost [Mao 2023; Tan 2024; Boudiaf 2022; +1 more].

The single most important shift in the evidence is that test-time defenses help most when they use intrinsic structure selectively and are evaluated as full adaptive systems, not as preprocessing tricks [Mao 2023; Croce 2022]. The biggest open question for this base paper is still whether restored equivariance is a reliable proxy for restored correctness under the strongest adaptive attacks and across broader, less symmetry-aligned perturbations.
