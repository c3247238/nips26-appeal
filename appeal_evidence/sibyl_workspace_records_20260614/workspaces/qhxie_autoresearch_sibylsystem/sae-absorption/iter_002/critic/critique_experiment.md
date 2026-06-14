# Experiment Critique — SAE Absorption Paper

## Experimental Design Quality

### Fatal Weakness: n_pos = 18 Ground Truth Labels

The entire paper rests on 18 exact absorbed features out of 24,576 (base rate 7.3×10⁻⁴). This is an extremely small positive class that creates multiple problems:

**Statistical power:**
- AUPRC = 0.00153, which is only 2.09× the base rate. The pre-registration required >3× base rate. This criterion was not met.
- Precision@50 = 0, Precision@100 = 0, Precision@500 = 0.008 (1 true positive). These values confirm EDA is not practically usable as a top-k shortlist at any k ≤ 500.
- The z = 2.49 significance is marginal. The permutation null standard deviation is 0.064, meaning the 95% confidence interval for EDA AUROC is approximately [0.524, 0.776]. This interval is compatible with AUROC = 0.55 (near-random) at the lower end.

**Recommendations:**
1. Run FeatureAbsorptionCalculator on all 26 letters rather than the 8 tested in C2. Expected n_pos: ~60 (consistent with Chanin et al.'s reported ~67 for GPT-2 Small). This would triple the positive class and make AUROC comparisons robust.
2. Report bootstrap confidence intervals for all AUROC values (not just permutation z-score). At n_pos=18, a 95% CI via bootstrap will show the range of plausible AUROC values.

---

### Cross-Domain Absorption (Phase C) — Methodological Failure

**Problem 1: n=20 words per semantic hierarchy.**
Both animate_inanimate and noun_proper were tested with n=20 words each. C2_child_suppression_absorption.json shows absorption_rate = 9.999e-9 (effectively zero) for both, with ratio_to_null = 1.0. However, with only 20 words, the measurement has essentially no power to detect absorption. The methodology requires child latent firing at baseline, child suppression when parent fires, and sufficient n to distinguish from random. At n=20, even a true absorption rate of 10% would yield 2 events out of 20, which is not distinguishable from noise with a single permutation control.

**Minimum sample size calculation:** To detect an absorption rate of 0.05 (5%) above null (0%) with 80% power and Bonferroni-corrected α = 0.0125 for 4 hierarchies, approximately n = 500 events are needed per hierarchy. The current design is underpowered by a factor of 25.

**Problem 2: Parent latent identification method is not validated.**
The top-5 probe-aligned latents are used as "parent latents" for semantic hierarchies. There is no verification that these latents actually correspond to the semantic parent concept in the way that first-letter latents correspond to letter categories. For animate_inanimate hierarchy, the parent latents are features 3152, 18287, 18617, 16169, 3278 — the paper provides no evidence these are genuine animacy features beyond probe alignment.

**Problem 3: The sae-spelling measurement method may not generalize.**
The first-letter hierarchy uses integrated gradients to identify absorption events — a behavioral, activation-based method. The C2 redesign uses "child latent suppression when parent fires" — a simpler proxy. These two methods are not equivalent. The first-letter result (absorption_rate = 0.0083, ratio_to_null = 10.0) uses IG-ablation; the semantic hierarchy results use the simpler suppression proxy. The null result for semantic hierarchies could reflect the proxy method's inadequacy rather than the absence of semantic absorption.

**Verdict:** The C2 semantic hierarchy experiments do not support the claim "semantic absorption is absent in GPT-2 Small." They support only "semantic absorption was not detected with the methods and sample sizes used in this experiment."

---

### Hysteresis Experiment (Phase E.2) — Uninterpretable Due to Ceiling

**Design problem:** The hysteresis test compares absorption rates before and after reducing sparsity. The test is only interpretable if there exists a non-absorbed state accessible by reducing sparsity. With absorption rates at 0.876–0.978 across all tested configurations (including the target low-sparsity regime), the experiment is operating entirely within the "both states absorbed" region.

**The numbers:** Baseline = 0.959, fine-tuned = 0.960, from-scratch = 0.964. These are all within 0.005 of each other — less than the likely measurement noise from the 15-word test set per letter.

**Conclusion from data:** The data are consistent with **three** hypotheses equally:
1. Hysteresis exists (absorbed state is metastable, can't escape by reducing sparsity)
2. All SAE configurations in this sparsity range are in the absorbed basin regardless of path
3. Absorption rate measurement is too coarse (20 words per letter, 8 letters) to detect 5% changes

The paper's claim that "data are consistent with the absorbed state being metastable" is correct but incomplete — it ignores hypotheses 2 and 3, which are equally consistent with the data.

---

### B3 Cross-Architecture Comparison — Confounded

The Standard/L1 SAE uses `blocks.6.hook_resid_pre` while the TopK SAE uses a different hook point. B3_cross_arch.json audit note flags: "L0 not matched (TopK k=32 gives exact sparsity; Standard L0~50). Different hook points: Standard uses resid_pre, TopK uses resid_post at layer 6."

The paper reports "TopK SAEs show lower mean letter-feature EDA (0.476) compared to the L1-penalized SAE at the same layer (0.676)" as evidence that "exact-sparsity constraints alter the gradient landscape." This comparison is confounded by both L0 mismatch (32 vs. ~51) and hook point mismatch (pre-MLP vs. post-MLP). The difference in EDA could be entirely explained by either confound independently.

---

### B2 Phase Transition (Pilot LRT p=0.027 → Full E1 LRT p=0.456)

The pilot found sigmoid-vs-linear LRT p=0.027 (marginally significant), which motivated the full Phase E. The full E1 experiment with all 11 configurations finds LRT p=0.456 (not significant). This reversal is a cautionary tale about pilot-to-full discrepancy, but the paper handles it correctly by reporting the full result.

However, the paper attributes the pilot's LRT p=0.027 to "5 data points from the same layer" (cross-layer comparison) while the full analysis uses "11 configs across different layers, widths, and training regimes." The full analysis mixes apples and oranges — a within-layer sparsity variation would be a cleaner test of H4 than the heterogeneous 11-config mix. The methodology.md correctly identifies this as a design revision requirement but the paper does not report whether the within-layer-only analysis was performed.

---

### F2 Mitigation Verification — Skipped

F2_mitigation_verification.json shows "skip_condition met: no alternative architecture SAE available." The theoretical analysis of mitigations (Matryoshka, OrtSAE, ATM SAE) is entirely theoretical — no empirical verification was performed. The paper's Section 5.6 "Connections to Architectural Mitigations" is presented with appropriate hedging ("These connections are theoretical predictions, not verified by our experiments"), which is correct, but the experimental plan included F.2 as a mandatory phase that was dropped.

**Risk:** A reviewer will ask why the predictions of Proposition 1 (mitigations should increase mean decoder angle) were not empirically tested against SAEBench data which is publicly available.

---

## What Works Well

1. **Exact Chanin labels used**: The paper correctly uses FeatureAbsorptionCalculator (IG-ablation) labels as ground truth rather than proxy labels. The proxy label comparison (n_pos=50, Jaccard=0.115) is presented as a secondary check, not as primary evidence.

2. **Tautological variant identified and excluded**: The D2 analysis identified that using absorbed feature decoders as parent candidates creates a tautological variant (AUROC=1.0). This was correctly flagged and excluded. This is good experimental hygiene.

3. **11 SAE configurations systematically analyzed**: The scaling suite (layers 2–10, widths 12k–98k, AJT architectures) is reasonably comprehensive for GPT-2 Small. The consistent absorption rate saturation across this range is a genuine finding.

4. **Statistical methods**: Bootstrap CIs, permutation null, DeLong test, Spearman correlations, and Cohen's d are all applied appropriately. The audit report confirms no fabricated numbers (with the four exceptions noted in the writing critique).
