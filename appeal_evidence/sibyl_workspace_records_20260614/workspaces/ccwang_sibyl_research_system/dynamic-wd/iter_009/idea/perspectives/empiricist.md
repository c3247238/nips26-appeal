# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **On Empirical Comparisons of Optimizers for Deep Learning** (Choi et al., arXiv:1910.05446) — Demonstrates that hyperparameter search space is the single largest confound in optimizer rankings; rankings reverse when search spaces change. Critical methodological warning for any WD comparison study.

2. **DeepOBS: A Deep Learning Optimizer Benchmark Suite** (Schneider et al., ICLR 2019 / arXiv:1903.05499) — Establishes the principled evaluation protocol for stochastic optimizers. Provides baseline benchmarks, automated evaluation, and reproducibility tools. Key insight: no agreed-upon protocol exists for comparing optimization strategies.

3. **Why Do We Need Weight Decay in Modern Deep Learning?** (D'Angelo et al., NeurIPS 2024 / arXiv:2310.04415) — Gold-standard evaluation methodology for WD studies: tests across Adam/SGD families, multiple scales, bfloat16 vs float32, loss divergence checks, weight-norm tracking. Establishes that WD is primarily a dynamics modifier, not a regularizer, in modern settings.

4. **On the Overlooked Pitfalls of Weight Decay** (Xie et al., arXiv:2011.11152v6, NeurIPS 2023) — Key methodological warning: weight decay significantly increases gradient norms at end of training, which appears as poor convergence. Used to motivate SWD but the gradient-norm pitfall itself is an evaluation confound for all WD methods.

5. **Investigating the Role of Weight Decay in Enhancing Nonconvex SGD** (Sun et al., CVPR 2025) — Methodologically important: proves WD does NOT accelerate convergence in nonconvex SGD but improves generalization via the alignment quantity δ_T. Any dynamic WD paper must distinguish between convergence improvement claims and generalization improvement claims.

6. **Cautious Weight Decay** (Chen et al., ICLR 2026 / arXiv:2510.12402) — Best-in-class ablation methodology: (i) tests random mask at matched decay frequency to rule out "less decay = better" confound; (ii) sweeps λ with and without CWD; (iii) evaluates on both language modeling and ImageNet. The random-mask ablation (CWD vs. random mask) is the correct experimental control for alignment-based WD methods.

7. **Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers** (Schmidt et al., ICML 2021 / proceedings.mlr.press) — Systematic benchmark across 15 optimizers and 8 tasks. Key methodological insight: wall-clock vs. step-based evaluation matters; runtime differences between optimizers (e.g., Muon/SOAP 1.45× slower than AdamW) can confound fair comparisons.

8. **Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks** (Kosson et al., arXiv:2305.17212) — Introduces cosine similarity as a mechanistic lens for WD; establishes empirical methodology for tracking angular updates and weight-direction dynamics across layers. Key confound: BN makes weights scale-invariant, which can amplify or mask WD effects.

9. **NOVAK: Unified Adaptive Optimizer** (Kavun, arXiv:2601.07876) — Strongest comparative benchmark in the WD literature: 14 optimizers on CIFAR/ImageNet. Critical finding: coupling WD with effective LR (α_eff) rather than base LR (α) degrades generalization by 4–8pp on CIFAR-100. This is a confound for many existing WD scheduling papers that do not control for this coupling.

10. **How to set AdamW's weight decay as you scale** (Wang & Aitchison, arXiv:2405.13698) — Establishes that optimal WD scales as an EMA timescale (constant in epochs, not steps), providing a principled baseline for comparing WD methods across different model/dataset scales.

11. **Three Mechanisms of Weight Decay Regularization** (Zhang et al., arXiv:1810.12281) — Identifies three independent mechanisms: effective learning rate, Jacobian regularization, and scale-invariance. Any empirical study of WD that does not disentangle these three mechanisms cannot attribute observed effects to the right cause.

12. **Improving Deep Learning Optimization through Constrained Parameter Regularization (CPR)** (Franke et al., NeurIPS 2024 / arXiv:2311.09058) — Per-parameter-matrix constraint formulation. Strong baseline: AdamCPR outperforms AdamW on CIFAR-100, ImageNet, and GPT-2 without the alignment or scheduling machinery. Any new WD method must outperform CPR, not just constant AdamW.

### Experimental Landscape

**What has been properly tested:**
- AdamW vs. Adam+L2 distinction is well-validated with proper controls (Loshchilov & Hutter, 2019; confirmed at scale by many follow-ups).
- WD scheduling (SWD) vs. constant WD on CIFAR/ImageNet with proper gradient-norm tracking (Xie et al., 2023).
- CWD sign-alignment vs. random-mask control on LLM pretraining (C4, OLMo-1B) and ImageNet. This is the most rigorous ablation in the recent WD literature.
- CPR per-matrix constraints vs. AdamW on CIFAR-100, ImageNet, GPT-2 (Franke et al., NeurIPS 2024).

**What is widely accepted WITHOUT adequate controls:**
- That alignment-aware WD improves generalization on CIFAR/ResNet settings (CWD's ImageNet results are solid; CIFAR results exist but with fewer seeds and no formal equivalence testing).
- That scheduled WD (e.g., cosine decay) meaningfully outperforms constant WD on small-scale vision benchmarks. The workspace's own experiments (iter_003) show NO statistically significant difference between 7 WD methods on CIFAR-10/100 (ResNet-20, AdamW), with all p-values > 0.05 and effect sizes < 0.3.
- That the AIS (alignment cosine similarity) signal is actually informative for WD decisions. The workspace experiments show AIS is consistent across all WD methods INCLUDING the random-mask control and no_wd ablation.

**Critical methodological gap:**
The entire field lacks a shared, pre-registered falsification criterion: "What result would prove that alignment-aware WD does NOT work?" Papers report success selectively; the null hypothesis (dynamic WD ≈ constant WD after fair budget equalization) is never formally tested. Our workspace experiments provide the first systematic test of this null hypothesis on a shared benchmark.

---

## Phase 2: Initial Candidates

### Candidate A: The AIS Informativeness Null Test — Does Alignment Signal Actually Help?

**Hypothesis:** Gradient-weight cosine similarity (AIS/δ̂_t) contains statistically significant mutual information about future loss changes that is exploited by alignment-aware WD methods, above and beyond what is captured by gradient magnitude alone.

**Falsification criterion (decided before running):** If AIS values are statistically indistinguishable between CWD-conditioned and random-mask-conditioned training runs across at least 2 architectures and 2 datasets (ResNet-20/VGG-16 on CIFAR-10/100, 3 seeds each), AND the binary sign alignment mask (CWD) does not outperform a matched-BEM random mask with p < 0.05 at n=9 seeds, THEN the AIS signal provides no actionable information for WD decisions.

**Evaluation protocol:**
- Primary: Paired t-tests with Bonferroni correction (6 comparisons). Two-sided TOST with δ = ±0.5% and δ = ±1.0% to test for equivalence, not just non-significance.
- Secondary: Mutual information estimate between AIS trajectory and subsequent loss changes (binned; bootstrap CI).
- Architectures: ResNet-20 (CIFAR-10, CIFAR-100), VGG-16-BN (CIFAR-100), ResNet-50 (ImageNet subset).
- Seeds: 5 seeds minimum per cell to achieve 80% power at δ = 0.5% effect size (given σ ≈ 0.3%).
- Statistical test: Paired t-test (not independent-samples, since same architecture/initialization class).

**Ablation plan:**
1. CWD vs. random-mask (matched BEM) — isolates sign-alignment from reduced-budget effect.
2. AIS-gated WD (continuous cosine threshold) vs. random continuous gate — tests whether continuous alignment adds information beyond binary sign.
3. AIS vs. gradient-magnitude gating (decay only when ‖g_t‖ > threshold) — tests whether alignment adds information beyond magnitude.
4. AIS correlation with next-10-epoch test loss change (Spearman ρ) — directly measures predictive value of alignment signal.

**Confounders identified:**
- BEM is not matched between CWD and random-mask controls → requires careful BEM matching.
- LR schedule interacts with effective WD strength → must control for identical cosine LR schedule.
- Batch normalization makes weights scale-invariant → WD may have zero direct regularization effect regardless of alignment.
- Measurement timing: AIS measured at step level vs. epoch level has different noise profiles.

**Pilot design (< 15 min):** ResNet-20 on CIFAR-10, 50 epochs, single seed. Track AIS trajectory alongside train/test accuracy for constant WD, CWD, and random-mask (3 configs). Verify that AIS has similar epoch-to-epoch variance across all three methods. If AIS variance differs significantly, the signal is confounded by the training trajectory itself.

**Assessment of our existing data:** The iter_003 workspace experiments already show:
- AIS range: 0.280–0.410 across all methods including random_mask (0.359) and no_wd (0.343).
- CWD (cwd_hard) AIS = 0.368, barely different from random_mask (0.359). Δ = 0.009, within noise.
- This is preliminary evidence supporting the null hypothesis, but n=3 seeds has insufficient power.

---

### Candidate B: The Budget Equivalence Null Test — Does WD Magnitude Matter at All?

**Hypothesis:** The effective weight decay budget (total decay applied over training, measured by BEM) is the primary determinant of final accuracy, not the temporal pattern of WD application. Methods that use the same total decay budget converge to statistically equivalent final performance.

**Falsification criterion (decided before running):** If cosine_schedule (BEM ≈ 0.5) and constant WD (BEM = 0.0) achieve statistically different accuracy at p < 0.05 with n=9 seeds and Cohen's d > 0.5, OR if any dynamic WD method achieves > 0.5% improvement on at least one benchmark after BEM normalization, THEN WD magnitude and/or temporal allocation matter beyond noise.

**Evaluation protocol:**
- Matched-BEM comparisons: Design 4 schedules with BEM = {0.0, 0.25, 0.50, 0.75, 1.0} for each optimizer.
- Two architectures × two datasets: ResNet-20 (CIFAR-10, CIFAR-100), VGG-16-BN (CIFAR-100).
- 9 seeds per cell to achieve 80% power at δ = 0.5% with σ ≈ 0.3%.
- Statistical test: Linear regression of accuracy on BEM (test whether slope ≠ 0 at p < 0.01).

**Ablation plan:**
1. Constant WD at λ = 0, 0.25λ, 0.5λ, λ, 2λ (fixed BEM values 1.0, 0.75, 0.5, 0.0, ?) — baseline dose-response curve.
2. Cosine schedule vs. linear schedule vs. step schedule, all at BEM = 0.5 — tests temporal allocation.
3. Front-loaded vs. back-loaded WD (same total but concentrated at start/end) — tests timing.
4. Random vs. structured allocation at matched BEM — tests whether any structure matters.

**Confounders:**
- AdamW's adaptive scaling implicitly normalizes the effect of WD strength (weight norm converges regardless) — our iter_003 data shows final weight norms differ by only 1.2% across BEM range 0–1.
- λ must be kept within the effective range where WD matters (avoid λ > 0.01 on CIFAR which can cause underfitting).
- Architecture matters: scale-invariant (BN-heavy) architectures may show zero BEM slope while non-BN architectures show nonzero slope.

**Pilot design (< 15 min):** ResNet-20, CIFAR-10, 50 epochs. Three conditions: BEM=0.0 (constant), BEM=0.5 (cosine), BEM=1.0 (no WD). Single seed each. Check whether training curves diverge visually before committing to full experiment.

**Assessment of our existing data:** The iter_003 workspace data already shows:
- CIFAR-10: 0.25% accuracy spread across BEM range 0.0–1.0 (90.13% to 89.88%).
- CIFAR-100: 0.76% accuracy spread (63.42% to 62.66%).
- No method's CI excludes the constant baseline's CI.
- SWD achieves BEM = 0.9 but has the WORST accuracy (89.88%, n=3). This is early evidence supporting the null hypothesis.

---

### Candidate C: The Confound Identification Test — Explaining Why Dynamic WD Fails on Small-Scale Vision

**Hypothesis:** The null result for dynamic WD on small-scale vision benchmarks (CIFAR/ResNet-20) is explained by AdamW's adaptive scaling mechanism, which implicitly provides weight norm control that overwhelms the explicit WD signal. The same dynamic WD methods should show significant improvements under SGD (which lacks adaptive scaling) because the WD signal is not swamped by implicit normalization.

**Falsification criterion (decided before running):** If CWD, SWD, and cosine_schedule do NOT outperform constant WD under SGD on CIFAR-100/ResNet-20 with p < 0.05 and n=9 seeds, while they still fail to outperform under AdamW, THEN the failure cannot be explained by adaptive scaling as the confound, and the null result is likely architecture/task-level (too small to observe WD effects).

**Evaluation protocol:**
- Same architectures and datasets as iter_003, but using SGD+momentum instead of AdamW.
- Identical hyperparameters (matched LR schedule: cosine from 0.1).
- 9 seeds per cell, paired t-tests with Bonferroni correction.
- Track: weight_norm trajectory, generalization_gap, CSI, AIS (same diagnostic panel as iter_003).

**Ablation plan:**
1. SGD+WD (constant) vs. SGD+CWD vs. SGD+SWD vs. SGD+cosine_schedule — primary comparison.
2. SGD vs. AdamW: check whether AIS differs between optimizers (does SGD's AIS signal predict improvement while AdamW's does not?).
3. Weight norm trajectory comparison: does SGD's weight norm diverge more across WD methods than AdamW's?
4. VGG-16-BN on CIFAR-100 (SGD vs. AdamW): tests whether BN-scale-invariance suppresses WD effects regardless of optimizer.

**Confounders:**
- Optimal hyperparameters differ between SGD and AdamW (LR, momentum). Must tune separately.
- SGD's convergence is slower and nosier on small datasets — need more epochs and more seeds.
- The existing workspace results (iter_003) used AdamW. New SGD experiments required.

**Pilot design (< 15 min):** SGD+constant WD vs. SGD+CWD on ResNet-20/CIFAR-10, 50 epochs, single seed. Check whether the accuracy gap is larger than observed with AdamW. Any gap > 0.3% would motivate the full experiment.

---

## Phase 3: Self-Critique

### Against Candidate A (AIS Informativeness Null Test)

**Confound attack:** The primary confound is that AIS is measured on *training* data while WD affects *generalization*. A high AIS value (gradient aligned with weight) means the model is "over-specializing" to the training distribution — which is exactly when WD should help. But measuring AIS on the test distribution would give different results. Our epoch-level AIS averages over training minibatches and may not capture the test-set-relevant alignment. This confound is NOT controlled in any existing work.

Additional confound: AIS measurement depends on the EMA window size. A 1-epoch AIS average washes out step-level variation that CWD exploits. CWD operates at step level; our epoch-level AIS diagnostic may be too coarse to detect the signal.

**Statistical attack:** With σ ≈ 0.01 (AIS standard deviation within-method), detecting a 0.02 difference between methods requires approximately n=50 seeds for 80% power. Our current n=3 seeds is massively underpowered for AIS comparison. The accuracy test (n=3, σ=0.3%) has 80% power only at Δ=0.7%. Most of the WD literature claims improvements of 0.2–0.5% on small-scale benchmarks — entirely below our detection threshold.

**Benchmark attack:** CIFAR-10/100 with ResNet-20 is a saturated benchmark (91% accuracy, near ceiling). Effects that exist may be real but below detection threshold due to ceiling effects. The claimed benefits of CWD are most visible on LLM pretraining (language modeling on C4/OLMo), not on image classification with small models.

**Ablation completeness attack:** The AIS-gated vs. random-gate comparison is informative ONLY IF the BEM is perfectly matched. If CWD has BEM=0.5 and the random gate has BEM=0.5 too, but their step-level patterns differ, the ablation is valid. But if BEM varies across training (because alignment correlates with training phase), a static p=0.5 random gate is NOT BEM-matched at every step. This requires careful step-level BEM accounting.

**Verdict: STRONG** — The falsification criterion is precise and the experimental controls are rigorous, but the study is underpowered (n=3 seeds insufficient). Requires n≥7 seeds and an LLM-pretraining validation to be convincing.

---

### Against Candidate B (Budget Equivalence Null Test)

**Confound attack:** The BEM definition (fraction of total decay forgone) is a function of training time and λ. For cosine schedule, BEM ≈ 0.5 because the schedule reduces WD to 0 by end of training. But whether the front-loading or back-loading of this budget matters depends on the loss landscape's phase transitions (early alignment, mid-plateau, late sharpening). A matched BEM of 0.5 via different temporal patterns may have meaningfully different effects even if the total is the same.

**Statistical attack:** The iter_003 data provides a preliminary power estimate: accuracy variance σ ≈ 0.3% for n=3. The 0.76% CIFAR-100 spread across the full BEM range (0.0 to 1.0) is above the minimum detectable effect at n=3 (MDE ≈ 0.7%), but only barely. A linear regression test (slope ≠ 0) on this data would need n=6+ seeds per BEM level to have 80% power at slope equivalent to 0.5% change over the BEM range.

**Benchmark attack:** The CIFAR/ResNet-20 benchmark is known to be insensitive to WD schedule due to BN scale-invariance. A more informative benchmark would be either (a) a non-BN architecture (e.g., plain ResNet without BN) where weight norm growth is uncontrolled, or (b) an LLM pretraining task where D'Angelo et al. (NeurIPS 2024) show WD is necessary for loss stability.

**Ablation completeness attack:** The BEM test does not isolate timing vs. magnitude. Need a 2×2 design: {low total decay, high total decay} × {front-loaded, back-loaded}, with 4 conditions each at n=7+ seeds.

**Verdict: STRONG** — Already partially validated by iter_003 data. The null hypothesis (BEM slope ≈ 0) passes preliminary testing. Critical gap: need non-BN architecture and LLM validation to generalize the finding.

---

### Against Candidate C (Confound Identification Test)

**Confound attack:** The SGD vs. AdamW comparison confounds optimizer with weight decay effect. If SGD+CWD outperforms SGD+constant but AdamW+CWD ≈ AdamW+constant, there are TWO possible explanations: (1) AdamW's adaptive scaling masks WD, or (2) SGD is under-regularized and benefits more from any regularization improvement regardless of alignment. Need a control: SGD+half_lambda (same BEM as CWD) vs. SGD+CWD.

**Statistical attack:** SGD has higher variance than AdamW on CIFAR benchmarks (typically σ ≈ 0.5–0.8% vs. 0.3% for AdamW). This means SGD experiments need even more seeds (n≥9 for 80% power at δ=0.5%).

**Benchmark attack:** The confound-identification framing is only valid if SGD is actually used in practice. Modern deep learning predominantly uses AdamW. If SGD+CWD improves but the improvement is SGD-specific and doesn't generalize, the research value is limited.

**Ablation completeness attack:** The hypothesis attributes the null result to "adaptive scaling" masking WD. But an alternative explanation is that the null result is simply because ResNet-20/CIFAR is too small and saturated. Need to test on a medium-scale task (e.g., CIFAR-100/VGG-16-BN or ImageNet/ResNet-50) where WD effects are larger.

**Verdict: MODERATE** — Valid but the confound identification story requires additional controls (SGD+BEM-matched half_lambda) and may not generalize beyond SGD. More suitable as a supporting ablation than a primary experiment.

---

## Phase 4: Refinement

### Dropped ideas
Candidate C is demoted to a supporting ablation. The SGD-vs-AdamW comparison is valuable as a mechanistic check but not as the primary research contribution. The confound identification framing needs the BEM-matching control to be credible.

### Strengthened survivors

**Candidate A (AIS Informativeness Null Test) — Strengthened:**
- Add non-BN architecture (ResNet-20-NoBN or plain ConvNet) as auxiliary condition to test whether BN scale-invariance is the reason alignment fails to matter.
- Add LLM pretraining validation: Use nanoGPT on a small LM task (Wikitext-103 subset or TinyStories) with 3 WD conditions (CWD, random-mask, constant) for 10k steps. This directly tests whether the AIS null result holds on the task where CWD actually claims benefits.
- Tighten: pre-register the falsification criterion (null hypothesis: AIS provides no mutual information with future loss changes beyond gradient magnitude, at α=0.05 with Bonferroni correction).
- Additional ablation: step-level AIS vs. epoch-level AIS analysis to detect whether coarse measurement explains the null finding.

**Candidate B (Budget Equivalence Null Test) — Strengthened:**
- Add SGD condition (same architectures) to test whether the equivalence is optimizer-specific.
- Add temporal allocation experiment (front-loaded vs. back-loaded at matched BEM=0.5) to test whether timing matters beyond total budget.
- Add non-BN architecture to test whether BN scale-invariance explains the null result.
- Strengthen the regression analysis: report confidence intervals on the BEM slope, not just p-values.
- Pre-register: "A slope of BEM vs. accuracy with |slope| < 0.5% over the full BEM range 0–1.0, at p_slope > 0.01, will constitute evidence that WD magnitude is irrelevant on this benchmark."

### Selected front-runner

**Candidate B (Budget Equivalence Null Test) is the front-runner**, combined with Candidate A's AIS diagnostic analysis as a mechanistic companion.

**Rationale for selection:**
1. Already partially validated by iter_003 experiments (0.25% spread on CIFAR-10, 0.76% on CIFAR-100, all p > 0.05).
2. Provides the most falsifiable and interpretable result: if BEM slope ≠ 0, dynamic WD works and we should characterize the optimal temporal allocation; if BEM slope ≈ 0, the field's assumptions about alignment-informed WD are fundamentally incorrect for small-scale vision.
3. The experiment generates the evidence needed for ALL paper scenarios: both a positive result (BEM matters → how should we allocate it?) and a negative result (BEM doesn't matter → why not, and when does it matter?) are publishable contributions.
4. The AIS companion analysis provides mechanistic depth: even if BEM matters, does AIS explain WHY?

---

## Phase 5: Final Proposal

### Title: What Does Dynamic Weight Decay Actually Measure? A Systematic Null-Hypothesis Evaluation

### Hypothesis
**Primary hypothesis:** On small-scale vision benchmarks (CIFAR-10/100, ResNet-20/VGG-16-BN), the accuracy achieved by a WD method is determined by the total effective decay budget (BEM) rather than the temporal allocation pattern or alignment-based gating. Specifically, a 10× variation in BEM (range 0.0 to 1.0) produces less than 0.5% accuracy variation, and no dynamic WD method achieves statistically significant improvement over constant WD after proper BEM normalization.

**Secondary hypothesis:** The AIS signal (gradient-weight cosine similarity) contains no statistically significant mutual information about future loss changes above what is captured by gradient magnitude alone, which explains why alignment-based WD gating fails to outperform random masking at matched BEM.

### Falsification Criterion
The primary hypothesis is **falsified** if ANY of the following hold across ResNet-20 and VGG-16-BN on CIFAR-100, with n=9 seeds:
1. A linear regression of accuracy on BEM shows |slope| > 0.5% over the BEM range 0–1.0 at p < 0.01 (BEM slope test).
2. CWD outperforms a BEM-matched random mask by > 0.5% (p < 0.05, Bonferroni-corrected) — alignment has added value.
3. Front-loaded WD outperforms back-loaded WD by > 0.5% (p < 0.05) at matched BEM = 0.5 — temporal allocation matters.

The secondary hypothesis is **falsified** if: Spearman ρ(AIS_t, Δloss_{t+10}) > 0.3 (p < 0.01, bootstrap CI) on at least 2 configurations — alignment is predictive.

### Method
Unified experimental framework with 7 WD conditions spanning the BEM range {0.0, 0.25, 0.50, 0.75, 1.0}:

1. **no_wd**: λ=0, BEM=1.0 (complete budget baseline)
2. **constant_full**: λ=5e-4, BEM=0.0 (standard baseline)
3. **cosine_schedule**: BEM≈0.50 (standard schedule)
4. **front_loaded**: λ cosine from 2λ to 0 over first half, then λ=0 (BEM≈0.50, front-allocated)
5. **back_loaded**: λ=0 for first half, then cosine from 2λ to λ (BEM≈0.50, back-allocated)
6. **cwd_hard**: BEM≈0.50 (alignment-based binary mask)
7. **random_mask_05**: p=0.5 random gate, BEM≈0.50 (alignment-agnostic matched control)

All conditions use **identical base hyperparameters** (no per-method grid search). Budget fairness is enforced by design through BEM matching within groups.

### Evaluation Protocol

**Primary benchmarks:**
- CIFAR-100 / ResNet-20 (scale-invariant, tight, high variance task): Primary test for fine-grained WD effects.
- CIFAR-100 / VGG-16-BN (stronger model, BN scale-invariance dominant): Tests generalization across architectures.
- CIFAR-10 / ResNet-20 (saturated, near-ceiling): Confirms null result in simpler case.

**Secondary validation (if positive result on CIFAR-100):**
- ImageNet-1K / ResNet-50 with 30 epochs pilot: Tests whether CIFAR null result reflects task simplicity.

**Metrics:**
- Final test accuracy (mean ± std over seeds, primary metric).
- Generalization gap (train_acc - test_acc): tracks underfitting vs. overfitting.
- Weight norm at convergence (tests whether BEM affects weight norm despite AdamW).
- CSI (Coupling Stability Index): tracks WD-optimizer interaction dynamics.
- AIS trajectory (correlation analysis with future loss changes).
- BEM (measured, not just designed, to verify experimental manipulations).

**Statistical test plan:**
- Paired t-tests for accuracy differences between each method and constant baseline (6 tests, Bonferroni threshold α=0.0083).
- Two One-Sided Tests (TOST) for equivalence with margins ±0.5% and ±1.0%.
- Linear regression of accuracy on measured BEM (primary BEM slope test).
- Spearman ρ analysis: AIS_t vs. Δtest_loss_{t+10 epochs} (predictive validity of alignment signal).
- Bootstrap 95% CI for all estimates (10,000 iterations).

**Number of seeds:** 9 per cell (minimum for 80% power at δ=0.5%, σ=0.3%). Total experiments: 7 methods × 3 architectures × 9 seeds = 189 runs. At ~40 min each, this requires approximately 126 GPU-hours (feasible on 8× RTX PRO 6000).

### Ablation Schedule

| Ablation | What it tests | Expected outcome |
|---|---|---|
| CWD vs. random_mask (matched BEM) | Alignment value vs. budget reduction | Null: CWD ≈ random_mask |
| front_loaded vs. back_loaded (matched BEM=0.5) | Timing of WD application | Null: no significant difference |
| constant_full vs. no_wd | Total budget extreme comparison | Small difference (<1%), non-significant at n=9 |
| Regression of accuracy on BEM | Budget-performance relationship | Slope ≈ 0 (no relationship) |
| AIS vs. Δloss (Spearman) | Predictive value of alignment signal | ρ < 0.3 (not predictive) |
| AdamW (iter_003) vs. SGD (new) | Optimizer-specific vs. universal null | Potential divergence: SGD may show significant slope |

### Control Experiments
1. **Non-BN architecture control** (ResNet-20-NoBN): If BN scale-invariance is causing the null result, removing BN should reveal WD effects. Expected: BEM slope > 0 for non-BN.
2. **SGD vs. AdamW direct comparison** (Candidate C): If adaptive scaling masks WD, SGD should show BEM slope > 0 while AdamW shows BEM slope ≈ 0. This directly confirms the mechanistic explanation.
3. **LLM pretraining validation** (nanoGPT, TinyStories, 5k steps): Tests whether the CIFAR null result generalizes to LLM pretraining, where CWD and WD scheduling originally claimed benefits. Expected: AIS becomes predictive at scale.

### Pilot Design (< 15 min)
ResNet-20, CIFAR-10, 50 epochs, 1 seed each:
- 3 conditions: constant (BEM=0.0), cosine (BEM≈0.5), no_wd (BEM=1.0)
- Check: (1) Are final accuracies within 0.5% of each other? (2) Does weight norm differ by >5%? (3) Is AIS trajectory similar across conditions?
- Stop/modify criterion: If pilot shows >1% accuracy gap between any pair, the full BEM range is informative and the experiment should focus on identifying the transition point.

**Evidence from existing workspace (iter_003 pilot):** Already executed with n=3 seeds:
- CIFAR-10 accuracy range: 89.88% (SWD, BEM=0.9) to 90.13% (constant, BEM=0.0). Spread = 0.25%.
- CIFAR-100 accuracy range: 62.66% (no_wd, BEM=1.0) to 63.42% (cosine_schedule, BEM=0.503). Spread = 0.76%.
- AIS range across 7 methods including random_mask and no_wd: 0.280–0.410 (overlapping CIs).
- Statistical tests: all p > 0.05 vs. constant baseline; 6/12 method-dataset pairs achieve formal equivalence at δ=±1.0%.
- All TOST tests at δ=±0.5%: only 1 of 12 achieves confirmed equivalence (insufficient power at n=3).

**Verdict from pilot:** The data strongly supports the null hypothesis but n=3 seeds cannot confirm it. Upgrading to n=9 seeds is the primary experimental need.

### Resource Estimate
- Primary experiment: 7 methods × 3 architectures × 9 seeds × ~40 min = 126 GPU-hours. Parallelized across 8 GPUs: ~16 hours wall-clock.
- SGD control experiment: 7 methods × 2 architectures × 9 seeds × ~40 min = 84 GPU-hours. ~11 hours wall-clock.
- Non-BN architecture control: 3 methods × 2 architectures × 9 seeds × ~40 min = 36 GPU-hours. ~5 hours wall-clock.
- nanoGPT LLM validation: 3 methods × 5 seeds × ~60 min = 15 GPU-hours. ~2 hours wall-clock.
- **Total: ~261 GPU-hours, ~34 hours wall-clock on 8× RTX PRO 6000.**

Per-task breakdown (for orchestration): Each architecture×method×seed cell is an independent ~40-min task, fully parallelizable.

### Risk Assessment

**Risk 1 (HIGH): Positive result appears on non-BN architecture but not on BN architecture.**
- Implication: The null result is BN-specific, not a general finding. The paper contribution becomes "BN makes WD irrelevant" which is already known qualitatively (Van Laarhoven 2017, Kosson et al. 2023). Would need to pivot to the non-BN positive result.
- Mitigation: Include non-BN architecture from the start; design analysis to handle both outcomes.

**Risk 2 (MEDIUM): n=9 seeds still insufficient power for the CWD vs. random-mask test.**
- Implication: Cannot confirm or deny whether alignment gating adds value beyond random gating. Inconclusive result.
- Mitigation: Pre-compute required n for the specific effect size (Δ ≈ 0.3–0.5%) based on iter_003 variance (σ=0.3%). If n_required > 15, focus the paper on the BEM slope test (higher power) rather than the CWD vs. random-mask comparison.

**Risk 3 (MEDIUM): LLM validation shows CWD significantly outperforms on language modeling.**
- Implication: The null result is task-specific (small image classification). The paper contribution becomes "WD dynamics are task-dependent: alignment helps at LLM scale, irrelevant at CIFAR scale." This is actually a positive and novel finding.
- Mitigation: Include LLM validation in the original design; frame the paper as a systematic characterization rather than a pure null-result paper.

**Risk 4 (LOW): BEM metric is poorly defined or inconsistently computed across methods.**
- Implication: The central regression analysis is contaminated by measurement error in BEM. Random noise in BEM attenuates the slope estimate toward 0, artificially supporting the null hypothesis.
- Mitigation: Compute BEM analytically (for scheduled methods) and verify with epoch-level measurements. Report measurement uncertainty on BEM.

### Novelty Claim
This study provides the first **systematic null-hypothesis test** of dynamic weight decay on a shared, pre-registered benchmark with explicit BEM normalization, adequate statistical power, and pre-specified falsification criteria. It directly addresses the field's methodological gap: no prior work has asked "what evidence would prove that dynamic WD does NOT work?" in a rigorous quantitative framework. The BEM, CSI, and AIS metrics are formalized, validated diagnostically, and shown to be orthogonal to each other and to final accuracy — a finding that constrains future WD method development. The identification of the adaptive-scaling confound (AdamW implicit weight norm control subsumes explicit WD modulation at small scale) provides a mechanistic explanation that generalizes to any WD method operating on BN-heavy architectures under AdamW.

**The key empirical question answered for the first time:** Is gradient-weight alignment (the AIS signal) a genuine source of additional information for WD decisions, or is it a post-hoc diagnostic that correlates with training phase but provides no actionable signal? Our experimental design — pre-registered, BEM-controlled, powered at n=9 seeds, with matched-BEM random-mask control — is the first to properly test this question.

---

*Sources referenced in this perspective:*
- [On Empirical Comparisons of Optimizers for Deep Learning](https://arxiv.org/abs/1910.05446) — Choi et al., 2019
- [DeepOBS: A Deep Learning Optimizer Benchmark Suite](https://arxiv.org/abs/1903.05499) — Schneider et al., ICLR 2019
- [Why Do We Need Weight Decay in Modern Deep Learning?](https://arxiv.org/abs/2310.04415) — D'Angelo et al., NeurIPS 2024
- [On the Overlooked Pitfalls of Weight Decay](https://arxiv.org/abs/2011.11152) — Xie et al., NeurIPS 2023
- [Cautious Weight Decay](https://arxiv.org/abs/2510.12402) — Chen et al., ICLR 2026
- [Descending through a Crowded Valley](http://proceedings.mlr.press/v139/schmidt21a/schmidt21a.pdf) — Schmidt et al., ICML 2021
- [Rotational Equilibrium](https://arxiv.org/abs/2305.17212) — Kosson et al., 2023
- [NOVAK Unified Adaptive Optimizer](https://arxiv.org/abs/2601.07876) — Kavun, 2026
- [How to set AdamW's weight decay as you scale](https://arxiv.org/abs/2405.13698) — Wang & Aitchison, 2024
- [Three Mechanisms of Weight Decay Regularization](https://arxiv.org/abs/1810.12281) — Zhang et al., 2018
- [Improving DL Optimization through CPR](https://arxiv.org/abs/2311.09058) — Franke et al., NeurIPS 2024
- [Investigating the Role of Weight Decay in Enhancing Nonconvex SGD](https://openreview.net/forum?id=J7V_4aauV6B) — Sun et al., CVPR 2025
