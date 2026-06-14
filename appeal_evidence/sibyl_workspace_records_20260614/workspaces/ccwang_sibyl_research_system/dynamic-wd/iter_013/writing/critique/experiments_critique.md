# Critique: Experiments Section

**Overall Score: 6 / 10**

The section is structurally sound, covers the necessary components (setup, main results, ablations, analysis), and the reported numbers are internally consistent with the underlying raw data. However, several choices weaken the evidential force of the claims, and the statistical treatment is insufficient for a top venue. The analysis narrative sometimes overreaches beyond what the small sample size can support.

---

## 1. Experimental Setup Fairness

### 1.1 Hyperparameter Tuning Asymmetry (Major Issue)

The setup states that all baselines receive 50 Bayesian optimisation trials via Optuna, while EqWD uses its defaults without any tuning. This asymmetry is presented as a strength ("demonstrating that the defaults are competitive without search"), but it creates a logical tension that reviewers will immediately flag.

**Problem:** If EqWD is not tuned, the comparison is unfair in *one of two ways*:
- If the baselines genuinely benefited from tuning (i.e., untuned baselines would perform worse), EqWD's advantage is overstated because the baselines' tuned performance is compared against EqWD's untuned performance.
- If tuning made no difference, why tune the baselines at all? This raises the suspicion that the ablation table's $\beta = 5.0$ result (66.07%) was cherry-picked post-hoc to show EqWD has headroom.

**Recommendation:** Either (a) report tuned EqWD results alongside untuned to demonstrate robustness, or (b) run all methods untuned with the same default hyperparameter conventions, or (c) acknowledge that the 50-trial budget was chosen so baselines do not overfit to the held-out test set, and show the sensitivity of baseline performance to their own hyperparameters.

### 1.2 Training Duration (Moderate Issue)

ImageNet is trained for only 45 epochs, substantially below the standard 90-epoch convention for ResNet-50 (He et al. 2016, torchvision reference). All absolute accuracy numbers are correspondingly low (FixedWD 71.89% vs. ~76–77% at 90 epochs). This is an acceptable design choice for a fast comparison, but it must be explicitly defended, because:

- Differences between methods may be larger or smaller under full-length training.
- Reviewers familiar with standard ImageNet ResNet-50 numbers (76%+) will immediately notice the discrepancy.
- The claim that "ratio deviations are larger during prolonged transitional phases" is harder to assess when the training run is cut short.

**Recommendation:** Add a sentence explicitly justifying 45 epochs (e.g., "We use 45 epochs to enable multi-seed comparison within a fixed compute budget, following the accelerated-training literature [cite]. All methods are evaluated under identical conditions.") and consider adding a single-seed 90-epoch control for EqWD vs. FixedWD in an appendix.

### 1.3 No Adam/AdamW Evaluation

The method section claims EqWD is "compatible with any optimizer that supports decoupled weight decay, including SGD, Adam, and AdamW," yet all experiments use SGD. The lack of an AdamW result is a notable gap: a large fraction of modern practice (vision transformers, language fine-tuning) uses AdamW. A reviewer will ask why.

**Recommendation:** Add at least one AdamW experiment (e.g., ViT-S/16 or ResNet-50 with AdamW on ImageNet, even with a single seed) or explicitly scope the paper's claims to SGD-based training.

---

## 2. Statistical Significance

### 2.1 Three Seeds Are Insufficient for Significance Claims (Major Issue)

The section repeatedly makes comparative statements—"EqWD achieves the highest top-1 accuracy," "EqWD provides more stable training"—without reporting any formal statistical tests. With n=3 seeds, the standard deviation estimate is highly unreliable (effective degrees of freedom = 2), and the confidence intervals overlap substantially between EqWD and SWD on ImageNet.

**Concrete illustration from the data:**
- ImageNet EqWD: 72.27 ± 0.20 (values: 72.456, 72.064, 72.294)
- ImageNet SWD: 72.04 ± 0.40 (values: 72.324, 72.224, 71.584)

Seed 42 of SWD (72.324%) outperforms seed 42 of EqWD (72.456% — actually EqWD is better here, but the margin is 0.23% at the mean level). With overlapping 1-std bands and n=3, a two-sample t-test has essentially no power. The claim that EqWD's advantage is real cannot be statistically justified from this data alone.

**Recommendation:**
- Report 95% confidence intervals in addition to (or instead of) ±std.
- Run paired t-tests or Wilcoxon signed-rank tests across seeds and report p-values for the key EqWD vs. FixedWD and EqWD vs. SWD comparisons.
- Alternatively, increase to 5 seeds for the key ImageNet comparison.
- At minimum, add a caveat: "Given n=3 seeds, differences less than ~0.5% should be interpreted cautiously."

### 2.2 The CIFAR-100 Result Does Not Support the Main Claim

The abstract/intro (implied) and the cross-dataset analysis claim EqWD is generally better. On CIFAR-100, EqWD (65.05%) is third, behind both FixedWD (65.19%) and CPR (65.19%), and the gap is 0.14% in the wrong direction. The section attempts to explain this away as a "task complexity" effect, but:

- The explanation is post-hoc and untested. There is no ablation varying task complexity systematically.
- FixedWD being best on CIFAR-100 is actually consistent with EqWD being over-engineered for simple tasks—but this point is not made clearly.
- CPR also ties FixedWD on CIFAR-100, so the "simpler tasks favor simpler methods" narrative would apply equally to CPR.

**Recommendation:** Either (a) strengthen the task-complexity hypothesis by providing a quantitative measure of "complexity" (e.g., gradient norm variance, ratio deviation statistics) and showing EqWD's advantage correlates with it, or (b) adopt a more honest framing: "EqWD matches or exceeds baselines across settings, with the largest gains on complex tasks."

---

## 3. Data Presentation

### 3.1 Missing Standard Deviations in the Ablation Tables

The ablation tables (beta sensitivity, EMA sensitivity) use a single seed, which is appropriate for exploration. However, the tables report only point estimates with no uncertainty. The paper then makes monotonic trend claims ("performance degrades monotonically as alpha increases") that could be coincidental with a single seed.

**Problem:** The ablation data (from `cifar_analysis_summary.md`) shows:
- beta=1.0: 65.39% (single seed 42)
- beta=0.5: 65.07%
- beta=0.1: 65.21%

The non-monotonicity at the low end (0.1 → 0.5: -0.14%, then 0.5 → 1.0: +0.32%) is consistent with noise, not a meaningful trend. Yet the discussion treats the full beta=0.1 to 5.0 sweep as showing a clean monotonic pattern.

**Recommendation:** Run at least 2-3 seeds for the key ablation points (beta=1.0 vs. 5.0) that are discussed in the main text, since these are used to justify the default choice and the claim of "robust performance."

### 3.2 Inconsistency Between Table and Text on Rounding

The paper table shows:
- EqWD CIFAR-100: 65.05 ± 0.36

The raw data shows:
- mean: 65.053, std: 0.362

This is fine. However, the text says "EqWD achieves 65.05%, ranking third behind FixedWD (65.19%) and CPR (65.19%)," but CPR's raw mean is 65.187%, which rounds to 65.19%, while FixedWD's raw mean is 65.193%, also rounding to 65.19%. The table shows both as 65.19%, making it appear tied, but EqWD is actually 0.14% behind. This should be acknowledged more precisely—the tie between FixedWD and CPR is itself a presentation artifact of two-decimal rounding.

### 3.3 The Layer-Type Ablation Is Underpowered

The layer-type ablation (Uniform EqWD: 62.81 ± 1.31 vs. Layer-aware EqWD: 62.32 ± 1.19) uses VGG-16-BN on CIFAR-100. The standard deviations are 1.19–1.31%, which is enormous relative to the 0.49% difference between variants. This difference is not statistically meaningful. Yet the section uses it to make a theoretical claim about scale-invariance of BN layers.

**Recommendation:** Either (a) run more seeds to reduce variance and strengthen the conclusion, or (b) reframe this ablation as "the simpler uniform variant is not harmed by skipping layer-specific logic," dropping the theoretical claim.

### 3.4 Missing Training Curves

The section refers to Figure~\ref{fig:ratio_trajectories} and Figure~\ref{fig:wd_heatmap} for qualitative analysis, but there are no training accuracy/loss curves comparing EqWD vs. baselines. Seeing whether EqWD converges faster, more stably, or only differs at convergence would materially strengthen the "more stable training trajectory" claim.

**Recommendation:** Add training curve figures, or at minimum include the final training accuracy/loss alongside test accuracy in the main table. The raw data includes `final_train_loss` and `final_train_acc` for all runs—these could be reported in an appendix.

---

## 4. Whether Results Support Claims

### 4.1 "EqWD's advantage scales with task complexity" — Partially Supported

The ImageNet result (+0.38% over FixedWD, +0.23% over SWD) is real and consistent across all three seeds. The CIFAR-100 result (-0.14% vs. FixedWD) is also consistent. The directional claim is supported.

However, the paper currently relies on only two data points (one "complex" and one "simple" task) to establish a scaling relationship. The claim as written ("a clear pattern") is too strong for two data points—especially when the two data points differ on many dimensions simultaneously (dataset size, resolution, architecture scale, training length).

### 4.2 "Reduced variance indicates more stable training" — Weakly Supported

EqWD's lower std on ImageNet (0.20% vs. SWD's 0.40%) is cited as a key advantage. But with n=3, the std estimate has 2 degrees of freedom and is highly noisy. The true population std of EqWD could be anywhere from 0.05% to 0.60% at the 90% confidence level. Presenting this as an established advantage is premature.

### 4.3 "CAWD underperforms FixedWD, isolating ratio deviation as the key signal" — Well Argued but Circular

The CAWD ablation (cosine alignment modulation) vs. EqWD (ratio deviation modulation) is the most important ablation in the paper. Its conclusion—that ratio deviation is the effective signal—is the key contribution claim. However:

- CAWD uses cosine alignment as a raw scalar, not normalized. The comparison may be conflating "alignment signal" with "poor implementation of alignment signal."
- The paper does not report whether CAWD uses the same EMA structure and sensitivity parameter as EqWD, or whether it has been equivalently tuned.
- If CAWD was also not tuned (same treatment as EqWD), then its underperformance is evidence for the claim. But if CAWD was tuned via the same 50-trial budget, then FixedWD's outperformance of CAWD is actually the more puzzling finding.

**Recommendation:** Explicitly state the CAWD hyperparameter convention (tuned or untuned, which defaults), and clarify whether the comparison controls for modulation magnitude (i.e., are EqWD and CAWD designed to produce similar average modulation amplitudes?).

### 4.4 The "CWD and CPR fail at ImageNet scale" Narrative Is Plausible But Speculative

The paper attributes CWD and CPR's poor ImageNet performance to "discrete modulation signals" and "higher dimensionality." This is a reasonable hypothesis, but:

- CPR uses a smooth augmented Lagrangian, not a binary signal—the characterization of it as "binary/threshold-based" in the same sentence as CWD is imprecise.
- The paper provides no direct evidence (e.g., measuring gradient noise in binary decisions, or showing that continuous-signal variants of CWD perform better at scale).

**Recommendation:** Separate the CWD (binary masking) and CPR (norm-constrained) explanations. CPR's underperformance may be due to different reasons (e.g., the norm constraint becoming binding too frequently at large scale), and conflating them under "discrete modulation" is inaccurate.

---

## 5. Summary of Priority Fixes

| Priority | Issue | Fix |
|----------|-------|-----|
| High | No significance tests; n=3 insufficient for comparative claims | Add t-tests / CIs; increase seeds for key comparison |
| High | Hyperparameter tuning asymmetry between EqWD and baselines | Run tuned EqWD OR explain the asymmetry's implications explicitly |
| High | CIFAR-100 EqWD loses to FixedWD — main claim weakened | Reframe as "scales with complexity" with supporting evidence |
| Medium | 45-epoch ImageNet training not justified | Add justification sentence; add appendix 90-epoch control |
| Medium | Ablation tables use single seed | Add 2–3 seeds for the key beta and alpha ablation points |
| Medium | Layer-type ablation is statistically underpowered | Run more seeds or reframe conclusion more conservatively |
| Medium | No AdamW experiments despite claiming optimizer compatibility | Add one AdamW experiment or explicitly scope claims to SGD |
| Low | CPR mischaracterized as "binary/threshold-based" | Separate the CWD and CPR explanations |
| Low | Missing training curves | Add curves or appendix table with train loss/acc |
| Low | Two-decimal rounding obscures FixedWD vs. CPR tie | Use three decimal places or note the tie is rounding artifact |
