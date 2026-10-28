# Plan additions from the Consensus literature summary (draft, 2026-10-28)

Source: a Consensus AI summary the user pasted on 2026-10-28, saved verbatim in `reports/consensus_summary_2026-10-28.md`. **Its citations are unverified.** Some (Mittal 2026, Su 2026, Krumtunger 2026, Wesego 2026) are too recent to check from memory. Nothing from it goes into the thesis until the paper itself has been read.

**Merged into `IMPLEMENTATION_PLAN.md` on 2026-10-28 after P6b; the plan is authoritative.** Changes at merge, in the light of P6b (the defense collapses under BPDA-50): A2 dropped and A1 made optional (a stronger attack can't change the conclusion); B became the central P5b question (is the failure visible?). P6b jobs stamp the git commit at start, so the plan isn't edited mid-run. None of it changes the locked P4c config or the running P6b queue.

## Triage: what the summary proposes vs what we already know

| Summary idea | Status in our project | Action |
|:--|:--|:--|
| Adaptive / diverse transform selection (Kim 2020; Mao's theorem: gains need *new* information, not more views) | M1 done: `up1.5` = full views at 1.5× less time; `no_up` hurts the clean model | **D1** (analysis): measure view redundancy to explain M1 with Mao's theorem |
| Uncertainty- / region-weighted equivariance loss (Tan 2024, Wang 2024) | Tested twice, negative: P4 placement test failed; M3 failed (inv_conf −6.1). Confidence is inverted on the clean model (entropy image AUROC 0.013–0.018) | None. Cite as consistent with our negative result |
| Held-out equivariance for stopping / gating (Boudiaf 2022) | Stopping is moot (`one_step`). The gate's 0.5× view is close to the defense's own 0.5× view (cnt 3), so it is not held out | **B1** (analysis), then optional **M5** (method) |
| Amortised one-step repair (Croce 2022, Wang 2025, Singh 2023) | Done without training: `one_step`, 24–29× faster than the original, equal under PGD and BPDA-10 | None. Use in related work: we reach a one-step defense with no learned component |
| Learned initialiser (Lee 2023, Bi 2024) | Needs training; breaks "no retraining" | Future work |
| Feature-space / dual-domain repair (Wang 2025, Su 2026) | Not tried; new loss and tuning round | Future work |
| Harder evaluation: full-defense gradients, BPDA/EOT, attack hyperparameters (Croce 2020/2022, Lee 2023), segmentation-specific attacks (Xiao 2023) | Mostly in P6b already: SegPGD, CosPGD, BPDA-10/50, EOT, fixed seed, transfer, noise control; smoothed full gradient planned | **A1** (attack hyperparameter sweep), **A2** (APGD via BPDA) |
| Certified, task-aware equivariant robustness (Schuchardt 2023) | Out of scope for 4 weeks | Future work |
| "Reliability-Guided Adaptive Equivariance Repair" (four combined parts) | Two of its four parts (region weighting, budget placement) already failed our tests; one (fast repair) is done | Not adopted as a package |

## A. Evaluation hardening (P6b, after the current queue)

**A1. Attack-hyperparameter sweep** (Lee & Kim 2023: purification robustness depends on attack settings).
- Setup: BPDA-50 against ours, clean model, 50 test100 images, seed 0.
- Step sizes 0.5, 1 and 2/255, plus 3 random restarts (per image, keep the restart that does the most damage).
- Report the minimum over settings as the BPDA-50 number. If it is > 1.5 below the default setting, re-run the 3-seed BPDA-50 row with that setting.
- Cost: about 4 × the BPDA-50 single-seed run.

**A2. APGD-CE through BPDA** (Croce & Hein 2020: no step-size choice for the attacker).
- 50 images, 100 iterations, against ours and `one_step` always-on, clean model.
- This is the first stage of the optional reduced SEA, so it replaces SEA if time is short.

Already covered in P6b; no change: segmentation-specific losses (SegPGD, CosPGD), EOT, fixed-seed, transfer, noise control, gate-aware (P6a).

## B. Is restored equivariance a proxy for restored correctness? (analysis, days 13–14 slot)

Mao et al. leave this open, and the summary names it as the main open question. We can answer it for our setting with forward passes only.

- **Held-out view:** a view the defense never optimised.
  - Clean model (7 views: resizes 2×/0.3×/2×/0.5× and two rotations): horizontal flip, and resize 0.75×.
  - Adv model (9 views, which include the flip): resize 0.75×.
- **Per image, after the defense:** the held-out consistency error (softmax L2 to the base output, as in `consistency_map`) and the image's own mIoU and pixel accuracy.
- **Inputs:** clean, PGD-50 and BPDA-50. Check whether `p6b_attacks.py` saves x_adv; if not, regenerate BPDA-50 on 50 test100 images (seed 0).
- **Report:**
  - The Spearman ρ between the held-out error and per-image mIoU.
  - The pixel-level AUROC of the held-out error for "defended prediction wrong".
- **Pre-registered reading:**
  - "Proxy holds" on an input type if ρ ≤ −0.5 and the pixel AUROC is ≥ 0.80.
  - Under BPDA-50, either outcome is a finding. If the proxy holds, the defense fails *visibly* and abstention or flagging is possible. If it fails, the attack restores equivariance without correctness, which is the counterexample to the paper's premise.
- **Cost:** under 30 min if x_adv is saved; about 1 h more if BPDA-50 has to be regenerated.

## C. M5 (optional, cut first): held-out-view budget selection

Only if B shows the proxy holds under PGD-50 on the clean model.
- For each image the gate lets through, run `one_step` at ε ∈ {ε_v/2, ε_v} and keep the result with the lower held-out error (B1 views).
- Compare against the locked `gate+perimage` on tune100, 3 seeds. It costs about 2× the defense time for gated images.
- **Pre-registered rule:** adopt only if it beats `gate+perimage` by > 1.5 on PGD+def or clean+def, losing ≤ 1.5 on the other, **and** it is not > 1.5 worse under the P6a gate-aware BPDA attack (λ = 1, 50 images).
- **Known risk:** choosing among candidates by a loss is the pattern that broke SOAP (Croce et al. 2022; already on the future-work list as "restart selection by lowest loss"). The BPDA condition in the rule is there for this reason. A negative result is reported as a confirmation of Croce et al.

## D. View redundancy (analysis, optional)

Mao's theorem says added constraints help only if they carry new information about the label.
- On 10 tune100 images (both models, clean and PGD inputs), compute each view's equivariance-loss gradient w.r.t. δ at the `one_step` start point.
- Report the pairwise cosine and the sign agreement (`one_step` uses only the sign).
- **Expected use:** explain M1. If the two 2× upscale views are near-duplicates of each other and of 1.5×, that explains why `up1.5` matches `full`. `no_up` failing on the clean model would mean the upscale direction carries information the downscale views don't.
- **Cost:** minutes.

## E. Future-work list (P7), additions

- Learned one-step initialiser for the repair (amortisation; Lee 2023, Bi 2024 per the summary, to verify)
- Feature-space or dual-domain repair (Wang 2025, Su 2026, to verify)
- Equivariance-preserving randomized smoothing for certification (Schuchardt et al. 2023)
- Per-input transform policies (Kim et al. 2020)

## F. Thesis rule (P7)

Every citation that comes from the Consensus summary is checked against the paper itself (title, authors, venue, the claim attributed to it) before it is used.

## Schedule fit and cut order

- **A1 / A2:** after the P6b queue (next free GPU slot).
- **B, then D:** in the days 13–14 slot, alongside P5.
- **C:** only if B passes, and only if days 13–14 aren't needed as buffer.
- **Cut order** (prepended to the plan's existing one): C, then D, then A2. **Not cut:** B, because it is cheap and answers the base paper's open question; A1, because it only lowers our numbers (it is honesty, not a feature).
