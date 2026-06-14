# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

The following 10 resources are selected for their focus on experimental methodology, evaluation protocols, and known pitfalls in the dynamic weight decay literature:

1. **Sun et al. (CVPR 2025) — "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"** — Establishes the baseline convergence-vs-generalization trade-off for fixed WD in nonconvex SGD. Critical methodology point: proves WD does NOT accelerate convergence, only improves generalization. Any empirical claim of "faster convergence with dynamic WD" must be tested against this theoretical result. Falsification baseline for our work.

2. **D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415) — "Why Do We Need Weight Decay in Modern Deep Learning?"** — Key evaluation insight: WD is never useful as explicit regularization; its benefit comes through training dynamics modification. This reframing means experiments must track dynamics metrics (weight norm, gradient norm trajectories) not just final accuracy. Their experimental setup (ResNet/VGG/ViT on CIFAR/ImageNet with comprehensive tracking) is the gold-standard evaluation framework to adopt.

3. **Chen et al. (ICLR 2026, arXiv:2510.12402) — "Cautious Weight Decay (CWD)"** — The strongest existing alignment-aware WD baseline. Uses sign-alignment as a binary gate. Key pitfall: papers claiming continuous alignment outperforms CWD must verify the improvement is not merely due to hyperparameter tuning, since CWD already includes a learnable scaling factor. Any new alignment method needs to be compared with identical hyperparameter search budgets.

4. **Franke et al. (NeurIPS 2024, arXiv:2311.09058) — "Improving Deep Learning Optimization through Constrained Parameter Regularization (CPR)"** — Shows per-parameter-matrix adaptive WD (via augmented Lagrangian) outperforms uniform AdamW on CIFAR-100, ImageNet, GPT-2. Critical evaluation insight: CPR only has 2 hyperparameters vs. the many hyperparameters of hand-crafted adaptive schedules. Any proposed method must benchmark against CPR and use comparable hyperparameter search budgets — otherwise, apparent gains may be from tuning effort, not method quality.

5. **Xie et al. (NeurIPS 2023, arXiv:2011.11152) — "On the Overlooked Pitfalls of Weight Decay (SWD)"** — First WD scheduler. Critical pitfall: SWD uses gradient-norm-aware modulation, which risks conflating training-phase effects (gradient norm naturally decreases) with alignment-related effects. Any experiment that tracks gradient norm as a WD signal must use controls that separate temporal-schedule effects from adaptive-signal effects.

6. **Kosson et al. (arXiv:2305.17212) — "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks"** — Key evaluation insight: WD benefits can be largely explained by rotational equilibrium effects (balanced average rotation across layers). A methodological pitfall: papers claiming "alignment-aware WD improves performance" may be capturing rotational equilibrium dynamics rather than gradient-weight alignment specifically. Controlled ablations must include a "rotation-balanced WD" baseline.

7. **Wang & Aitchison (arXiv:2405.13698) — "How to set AdamW's weight decay as you scale"** — Establishes that optimal WD scales predictably with model/dataset size via EMA timescale. Critical for experimental design: any benchmark comparison must control for WD scale — methods tested with differently tuned base WD values are incomparable. Provides a principled WD normalization protocol that should be adopted in all comparisons.

8. **Defazio (arXiv:2506.02285) — "Why Gradients Rapidly Increase Near the End of Training"** — WD drives gradient-to-weight ratio ‖g‖/‖w‖ to a steady-state. Key evaluation insight: this ratio is a clean unifying quantity whose trajectory across training summarizes WD's effect. Any method comparison should plot and compare ‖g‖/‖w‖ per-layer trajectories. If two methods achieve similar ‖g‖/‖w‖ trajectories but claim different mechanisms, the claimed mechanism difference needs independent verification.

9. **Steiner et al. (TMLR 2022) — "How to Train Your ViT? Data, Augmentation, and Regularization"** — The most comprehensive ablation of WD interaction effects in ViT training. Critical pitfall: WD effects on ImageNet-scale ViT training are strongly confounded with data augmentation strategy. Any ImageNet experiment must control for augmentation protocol, otherwise WD comparisons are uninterpretable.

10. **Fernandez-Hernandez et al. (arXiv:2504.17160) — "OUI: Overfitting-Underfitting Indicator for WD selection"** — Provides a validation-free diagnostic tool for monitoring WD quality during training. Key evaluation use: OUI trajectories can reveal whether a dynamic WD method is responding to genuine overfitting signals or to noise in the training dynamics. Should be logged alongside all experiments as a confound-detection tool.

### Experimental Landscape

**What has been properly tested:**
- Fixed WD benefits in nonconvex SGD (Sun et al., CVPR 2025): rigorous theoretical analysis with empirical validation on VGG/ResNet
- Decoupled vs. coupled WD in Adam (Loshchilov & Hutter, ICLR 2019): definitive across many architectures
- Binary sign-alignment gating (CWD, ICLR 2026): well-ablated drop-in modification
- Per-parameter-matrix upper-bound constraint adaptation (CPR, NeurIPS 2024): compared against AdamW across CIFAR/ImageNet/GPT-2
- WD-augmentation interaction in ViT training (Steiner, TMLR 2022): exhaustive ablation

**What is accepted without proper controls:**
- Claims that "continuous alignment signal outperforms binary sign alignment" — CWD uses sign as a zero-cost proxy; continuous cosine similarity is more expensive and the marginal value has not been tested with matched compute budgets
- Claims that "dynamic WD achieves better convergence/generalization trade-off" — these are confounded with temporal schedule effects (SWD) unless explicit controls rule out "equivalent schedule without alignment signal" baselines
- Claims that alignment-aware WD works "across architectures" — most papers test 1-2 architectures and treat this as generalization; proper cross-architecture testing with statistical significance is rare

**Methodological gaps (where experimental evidence is weakest):**
- No paper has conducted a **Budget Equivalence** test: given equal training budget (FLOPs), does dynamic WD outperform optimally-tuned fixed WD?
- No paper has properly tested whether the **alignment signal** (gradient-weight cosine similarity) carries information above and beyond what is already captured by the gradient norm or weight norm alone — this is the key question for "alignment-aware" claims
- The proposed unified framework's theoretical metrics (BEM, CSI, AIS) have never been measured empirically — making them the highest-priority experimental contribution
- ImageNet-scale experiments with dynamic WD are rare and inconsistent across papers

---

## Phase 2: Initial Candidates

### Candidate A: Does Gradient-Weight Alignment Actually Contain Information Beyond Gradient Norm?

**Hypothesis:** The gradient-weight cosine similarity δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖) contains statistically significant additional information for predicting generalization performance beyond what is already captured by (a) the gradient norm ‖g_t‖ alone, and (b) the gradient-to-weight ratio ‖g_t‖/‖w_t‖ alone. Specifically: a regressor trained on (‖g_t‖, ‖w_t‖, δ̂_t) will achieve at least 10% lower MSE in predicting final test accuracy compared to a regressor trained on (‖g_t‖, ‖w_t‖) alone, across ResNet-20/VGG-16-BN on CIFAR-10/100, measured at 50% of total training steps.

**Falsification criterion:** If the partial R² of δ̂_t conditional on (‖g_t‖, ‖w_t‖) is < 0.05 (i.e., δ̂_t explains less than 5% additional variance in final generalization), then alignment-aware WD is theoretically unjustified as a distinct strategy from norm-only WD methods. This test must be conducted BEFORE proposing any alignment-aware WD algorithm.

**Evaluation protocol:**
- Metrics: partial R² and conditional mutual information I(δ̂_t ; test_acc | ‖g_t‖, ‖w_t‖) at multiple training checkpoints (20%, 50%, 80% of training)
- Statistical test: bootstrap confidence intervals (B=1000) on partial R²; Fisher z-transformation for correlation comparison
- Benchmarks: ResNet-20/CIFAR-10, ResNet-20/CIFAR-100, VGG-16-BN/CIFAR-10, VGG-16-BN/CIFAR-100
- Multi-seed: 3 seeds (42, 123, 456); report mean ± std
- Multiple WD regimes: λ ∈ {0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2} to sample diverse alignment trajectories

**Ablation plan:**
1. Track δ̂_t per layer vs. aggregated across layers — tests whether layer-specific alignment has additional information beyond global alignment
2. Compare minibatch δ̂_t vs. multi-step EMA of δ̂_t — tests whether noisy minibatch alignment provides usable signal
3. Compare cosine similarity vs. sign-alignment (0/1) vs. ‖g_t‖/‖w_t‖ ratio — identifies which alignment representation is most informative
4. Track the same metrics during early vs. late training phases — tests whether alignment signal quality varies over the training trajectory

**Confounders identified:**
- Gradient norm ‖g_t‖ is itself a predictor of generalization (via SWD); must partial out its effect
- Weight norm ‖w_t‖ correlates with WD strength and is the primary control variable for norm-matched WD
- Minibatch variance in δ̂_t may overwhelm the signal; must test EMA smoothing
- Architecture-specific effects (BatchNorm layers behave differently from Conv layers for alignment metrics)

**Pilot design:** Single VGG-16-BN training run on CIFAR-10 with standard hyperparameters. Log δ̂_t, ‖g_t‖, ‖w_t‖ per layer per step. Run partial regression at 50% checkpoint. Estimated time: 15 minutes on one GPU.

---

### Candidate B: Budget Equivalence Test — Does Dynamic WD Beat Optimally-Tuned Fixed WD Given Equal Compute?

**Hypothesis:** At equal training FLOPs, no existing dynamic WD method (CWD, SWD, alignment-aware SGDW, norm-matched WD) achieves a statistically significant improvement over a fixed WD baseline that has been optimally tuned with the same hyperparameter search budget. Specifically: with a grid search of 12 fixed WD values (log-spaced from 1e-5 to 1e-1) and 3 seeds each, the best fixed WD will match or exceed dynamic WD methods on CIFAR-10/100 (ResNet-20, VGG-16-BN) at 90%, 100%, and 110% of the standard training budget.

**Falsification criterion:** If any dynamic WD method achieves test accuracy improvement ≥ 0.3% (absolute, 95% confidence interval lower bound) over the best fixed WD baseline at matched compute — controlling for training budget and hyperparameter search effort — then dynamic WD provides genuine benefit beyond hyperparameter optimization. This is the "null hypothesis to reject" for the entire dynamic WD literature.

**Evaluation protocol:**
- Primary metric: test accuracy at matched FLOPs budget (standardized to ResNet-20/CIFAR-10 baseline)
- Statistical tests: paired t-test across seeds, Wilcoxon signed-rank test; p < 0.05 required; Bonferroni correction for multiple comparisons across architectures
- Benchmarks: ResNet-20/CIFAR-10, ResNet-20/CIFAR-100, VGG-16-BN/CIFAR-10, VGG-16-BN/CIFAR-100, ResNet-50/ImageNet-1K (the last being the most important per project constraints)
- Multi-seed: 3 seeds for each configuration; report mean ± std
- Fixed WD search: 12 log-spaced values from 1e-5 to 1e-1, 3 seeds each = 36 runs as hyperparameter search budget for the fixed WD baseline

**Ablation plan:**
1. Compare with different training budgets (50%, 100%, 150% of standard epochs) — tests whether dynamic WD advantage appears at longer/shorter training
2. Compare fixed WD + cosine LR schedule vs. fixed WD + step-decay LR — tests whether LR schedule interactions confound the comparison
3. Run all methods with identical augmentation protocols — controls for augmentation-WD interaction (Steiner et al.)
4. Report per-layer alignment metrics for all runs — tests whether "winning" dynamic WD methods achieve better alignment trajectories or just better hyperparameter configurations

**Confounders identified:**
- Dynamic WD methods have their own hyperparameters (scaling constant c, min/max clip values); their search budget must match fixed WD search budget
- LR schedule interacts strongly with WD (Ferbach et al. 2026); must use identical LR schedules
- Training stability at extreme WD values may cause failures for fixed WD baselines; must implement early stopping to prevent false failures
- Compute-equivalent comparison requires FLOPs counting, not epoch counting

**Pilot design:** Train ResNet-20/CIFAR-10 with 3 fixed WD values (1e-4, 5e-4, 1e-3) and CWD at same compute budget (100 epochs). Compute test accuracy and gradient-weight alignment trajectories. Estimated time: 12 minutes per run, 4 runs = ~50 minutes total on one GPU.

---

### Candidate C: The Alignment Informativeness Score — Can We Quantify When Alignment Signal Actually Helps?

**Hypothesis:** The utility of gradient-weight alignment as a WD signal is NOT constant across training. It follows a predictable pattern: (a) during early training (first 20% of steps), alignment signal is highly noisy (minibatch variance > signal) and using it actively hurts performance; (b) during mid-training (20-80% of steps), alignment signal is most informative and alignment-aware WD provides measurable benefit; (c) during late training (80-100% of steps), alignment signal reverts to noisy as the loss landscape flattens. Specifically, a phase-aware alignment WD that applies alignment gating ONLY during the middle phase achieves ≥ 0.2% (absolute) improvement over constant-alignment WD on VGG-16-BN/CIFAR-100.

**Falsification criterion:** If the variance of δ̂_t (minibatch alignment) exceeds the mean signal-to-noise ratio threshold during all training phases on at least 2 of 4 architecture-dataset pairs, then δ̂_t is not a usable per-step signal and alignment-aware WD cannot logically improve over temporal scheduling. We decide this threshold (SNR > 2.0 for "usable") before seeing results.

**Evaluation protocol:**
- Primary metric: SNR of δ̂_t (= mean[|δ̂_t|] / std[δ̂_t] over sliding window of 100 steps) as a function of training progress, across 4 architecture-dataset pairs
- Secondary metric: test accuracy of phase-aware alignment WD vs. full-training alignment WD vs. fixed WD (3 seeds each)
- Statistical tests: bootstrap CI on SNR; Mann-Whitney U test on final accuracy comparisons
- Benchmarks: ResNet-20/CIFAR-10, ResNet-20/CIFAR-100, VGG-16-BN/CIFAR-10, VGG-16-BN/CIFAR-100
- Compute the Alignment Informativeness Score (AIS) = I(δ̂_t > threshold ; final_acc) / I(epoch > threshold ; final_acc) — ratio of alignment-conditioned information gain to phase-alone information gain

**Ablation plan:**
1. Compare SNR of raw δ̂_t vs. EMA-smoothed δ̂_t (decay 0.9, 0.95, 0.99) — tests whether smoothing rescues the signal
2. Compare per-layer SNR vs. global (averaged across layers) δ̂_t — tests whether layer-level alignment is more informative
3. Test phase-boundary sensitivity: early/mid/late boundaries at (10%/80%), (20%/80%), (30%/70%) — tests robustness of phase-aware strategy
4. Replicate with SGD (momentum 0.9) and AdamW separately — tests whether alignment signal quality is optimizer-dependent

**Confounders identified:**
- Training phase is confounded with gradient norm phase (gradient norms typically decrease then spike at LR decay); must partial out gradient norm contribution to SNR
- BatchNorm rescaling affects effective alignment; must separate BN layers from Conv layers in analysis
- Minibatch size affects variance of δ̂_t; all experiments must use identical batch size (128)
- EMA smoothing introduces a bias-variance tradeoff that itself needs to be characterized, not just applied

**Pilot design:** Single ResNet-20/CIFAR-10 training run (200 epochs). Log δ̂_t per step, compute rolling SNR over 100-step windows. Plot SNR as function of training progress. Estimate training phase boundaries. Estimated time: 15 minutes on one GPU.

---

## Phase 3: Self-Critique

### Against Candidate A (Alignment Information Content Test)

**Confound attack:** The partial R² test assumes a linear relationship between δ̂_t and test_acc. If the relationship is nonlinear (e.g., threshold effects, where only extreme alignment values predict generalization), the partial R² will underestimate the information content of alignment. A kernel-based or mutual information estimator would be more appropriate. This is a design flaw that must be fixed.

Additionally, the choice of prediction target (final test accuracy at convergence) is problematic: different WD values produce convergence at different speeds, so "final accuracy" conflates convergence speed with asymptotic performance. A better target is test accuracy at fixed FLOPs budget.

**Statistical attack:** With 6 WD values × 4 architecture-dataset pairs × 3 seeds = 72 training runs, and multiple checkpoints (20%, 50%, 80%), the multiple comparisons problem is severe. A Bonferroni-corrected significance threshold of p < 0.001 per test (for 50 tests) requires very large effect sizes to survive. The 5% partial R² threshold may be too small to survive this correction. Should pre-register a primary analysis with a single checkpoint (50%) and treat others as exploratory.

**Benchmark attack:** CIFAR-10/100 with ResNet-20/VGG-16-BN is a reasonable starting point, but these architectures are small enough that alignment dynamics may be different from the ImageNet-scale models that practitioners care about. The finding may not generalize. Must include ResNet-50/ImageNet to validate the primary finding.

**Ablation completeness attack:** The ablation plan tests representation of alignment (cosine vs. sign vs. ratio) but does not test whether the layer at which alignment is measured matters — alignment in the first layer vs. the last layer may carry completely different information. This should be added.

**Verdict: MODERATE.** The core information-theoretic question is well-posed, but the implementation has statistical design flaws (linear assumption, multiple comparisons) and the benchmark scope is too narrow for high-impact publication. Fix: use mutual information estimators instead of partial R², pre-register primary analysis with pre-specified threshold, add ImageNet-scale validation.

---

### Against Candidate B (Budget Equivalence Test)

**Confound attack:** The "hyperparameter search budget" comparison is very difficult to equalize in practice. Fixed WD with grid search is embarrassingly parallel (36 runs all independent), while tuning dynamic WD methods typically requires sequential experimentation (tune c, then tune λ_min/λ_max, etc.). The search strategies have fundamentally different computational profiles. A fairer comparison uses Bayesian optimization with a fixed evaluation budget (e.g., 50 trials) for both fixed and dynamic WD methods — this removes the sequential vs. parallel confound.

**Statistical attack:** The 0.3% absolute accuracy threshold for significance is arbitrary and may be insufficient for CIFAR (where differences of 0.1-0.2% are common in the literature but are often noise). For CIFAR-10 where baseline accuracy is ~95%, the standard deviation across seeds is typically 0.1-0.2%, meaning 3 seeds gives approximately ±0.12% uncertainty. A 0.3% threshold is reasonable but must be validated with power analysis before the experiment.

**Benchmark attack:** ResNet-20 on CIFAR is quite saturated (state-of-the-art near ceiling); even a 0.3% improvement may be methodologically meaningful but practically unimportant. The more important benchmark is ResNet-50/ImageNet where the absolute performance range is larger and the WD choice matters more. This benchmark should be the PRIMARY comparison, not an afterthought.

**Ablation completeness attack:** The ablation tests different training budgets but does not test different hyperparameter search budgets. Specifically: what if the fixed WD baseline is given DOUBLE the search budget (72 vs. 36 runs)? If fixed WD then beats dynamic WD, it suggests dynamic WD's advantage is search-efficiency, not algorithmic superiority. This "asymmetric budget" test is a critical control that should be added.

**Verdict: STRONG.** This is the most important experiment in the whole space and it has not been done. It directly tests the null hypothesis for the entire dynamic WD literature. With the fixes (Bayesian optimization for fair comparison, ImageNet as primary benchmark, power analysis to set threshold), this is a publishable contribution even if the result is null (showing fixed WD matches dynamic WD would be a significant negative finding).

---

### Against Candidate C (Alignment Informativeness Score)

**Confound attack:** The hypothesis that alignment is "most informative in mid-training" could be entirely explained by the gradient noise schedule: early training has high gradient variance (noisy minibatch gradients), mid-training has medium variance (gradients more consistent), late training has low gradient variance but potentially chaotic near the sharp minima region. The proposed SNR measurement does not separate alignment noise from gradient noise. The correct test must compute SNR of δ̂_t with gradient noise CONTROLLED (e.g., by computing the alignment using a running EMA of the gradient and comparing SNR to a "gradient-only" benchmark).

**Statistical attack:** The SNR > 2.0 threshold for "usable" is arbitrary and not justified from theory. Different tasks, architectures, and WD regimes will naturally produce different baseline SNRs. A more rigorous criterion would be: "alignment-conditioned prediction of next-step generalization gap is statistically better than gradient-norm-only prediction," tested with a held-out test set and block bootstrap for time-series data.

**Benchmark attack:** Phase boundaries (20%/80%) are specified for CIFAR but may be completely wrong for ImageNet (which has a different loss landscape geometry and different typical gradient dynamics). The hypothesis should be stated architecture-agnostically: "there exists an informative alignment phase whose boundaries can be identified from the data."

**Ablation completeness attack:** The ablation tests EMA smoothing factors but does not test whether the phase-aware strategy outperforms a simple "threshold on δ̂_t" strategy (apply WD only when δ̂_t < threshold). If the threshold strategy performs as well as the phase-aware strategy, the "phase" interpretation is unnecessary — the alignment value itself is the relevant quantity, not its phase in training.

**Verdict: MODERATE.** The core question (when is alignment signal useful?) is highly relevant to designing alignment-aware WD, but the proposed measurement strategy conflates several effects. Fix: measure SNR relative to a gradient-norm-only benchmark; use non-parametric power analysis; add the direct "threshold vs. phase" ablation.

---

## Phase 4: Refinement

### Dropped candidates:
None dropped — all three address distinct and important empirical questions. However, Candidate C is demoted from "primary" to "supporting analysis" because its methodological issues are more difficult to resolve without a cleaner theoretical framework first.

### Strengthened survivors:

**Candidate A → Strengthened:**
- Replace partial R² with mutual information estimator (MINE or k-NN based estimator) to capture nonlinear alignment-generalization dependencies
- Pre-register primary analysis: single checkpoint at 50% training, primary test is I(δ̂_t ; test_acc | ‖g_t‖, ‖w_t‖) using k-NN estimator with B=1000 bootstrap CI
- Add ResNet-50/ImageNet as validation benchmark (mandatory per project constraints)
- Add layer-level ablation: test whether alignment in specific layer groups (early, middle, late) is more informative than global alignment
- Add missing control: "rotation equilibrium index" (variance of per-layer mean rotation) as an alternative alignment-like predictor

**Candidate B → Strengthened:**
- Switch hyperparameter search to Bayesian optimization (Optuna, 50 trials budget for both fixed and dynamic WD methods, identical evaluation protocol)
- Establish ImageNet/ResNet-50 as the PRIMARY benchmark (not secondary), per project constraints
- Power analysis: with 3 seeds and expected σ ≈ 0.15% on CIFAR-10 accuracy, effect size d = 0.3/0.15 = 2.0 is detectable with 3 seeds at 80% power (two-sided t-test, α=0.05) — threshold is acceptable
- Add "asymmetric budget" control: fixed WD with 5× search budget to test whether fixed WD can match dynamic WD with more tuning
- Add CPR as additional baseline (it already does per-matrix adaptive WD with only 2 hyperparameters, making it a very strong baseline)

**Candidate C → Supporting analysis (not primary):**
- Reframe as a diagnostic/visualization study rather than a hypothesis test
- Compute SNR relative to gradient-norm-only benchmark (partial SNR)
- Use the AIS metric definition as a standardized measure: AIS = I(δ̂_t ; Δgap | ‖g_t‖) where Δgap is train-test gap at end of training
- Produce per-layer, per-phase AIS heatmaps as visualization contribution
- Include this in the paper as Figure 2 (diagnostic analysis) rather than as the core claim

### Selected front-runner: Candidate B (Budget Equivalence Test)

**Rationale:** This experiment tests the most fundamental question: does dynamic WD actually work better than optimally-tuned fixed WD given equal resources? A positive result (dynamic WD wins) validates the entire research agenda; a negative or mixed result is itself a high-value finding that should be published. It is the experiment that every reviewer of a dynamic WD paper should demand but nobody has run. It is verifiable, pre-registerable, and applicable across the full scope of benchmarks including the mandatory ImageNet experiments.

Candidate A is incorporated as a prerequisite diagnostic to be run first (it determines WHICH alignment signal to use in dynamic WD methods); its results will inform the design of the dynamic WD methods tested in Candidate B.

---

## Phase 5: Final Proposal

### Title
**"Does Dynamic Weight Decay Actually Help? A Rigorous Budget-Equivalent Evaluation Framework with Alignment Informativeness Diagnosis"**

### Hypothesis
**Primary (Budget Equivalence):** Given identical training FLOPs and hyperparameter search budgets (50 Bayesian optimization trials), at least one dynamic WD method (CWD, SWD, alignment-aware SGDW with δ̂_t, norm-matched AdamWN) achieves a statistically significant improvement ≥ 0.3% absolute test accuracy over the best fixed WD baseline on ResNet-50/ImageNet-1K, and ≥ 0.2% absolute on ResNet-20/VGG-16-BN on CIFAR-100. Significance is defined as the 95% bootstrap CI lower bound exceeding 0 (paired comparison across seeds 42, 123, 456).

**Secondary (Alignment Informativeness):** The gradient-weight cosine similarity δ̂_t contains statistically significant additional information for predicting final test accuracy beyond what is captured by (‖g_t‖, ‖w_t‖) alone, quantified by mutual information I(δ̂_t ; test_acc_end | ‖g_t‖, ‖w_t‖) > 0 (95% bootstrap CI lower bound > 0), across all four CIFAR architecture-dataset pairs.

### Falsification Criterion (pre-specified)
- **Budget Equivalence is falsified if:** No dynamic WD method achieves a CI lower bound > 0 on any primary benchmark at matched compute budget. This would indicate dynamic WD's literature gains are artifacts of unequal hyperparameter tuning budgets, not algorithmic superiority.
- **Alignment Informativeness is falsified if:** The k-NN mutual information estimator yields CI lower bound ≤ 0 for at least 3 of 4 architecture-dataset pairs, indicating δ̂_t provides no incremental predictive value over norm-only signals.

Both criteria are decided NOW, before any experiments are run.

### Method
The experiment compares five WD strategies under a strict budget-equivalence protocol:
1. **Fixed WD (baseline):** Bayesian hyperparameter search over λ ∈ [1e-5, 1e-1] (log-uniform prior), 50 Optuna trials, 3 seeds each per trial — total budget: 50 × 3 = 150 training runs
2. **CWD (Chen et al., ICLR 2026):** Binary sign-alignment gating. Same Bayesian search budget (50 trials over λ_base and masking scale parameter)
3. **SWD (Xie et al., NeurIPS 2023):** Gradient-norm-aware scheduled WD. Same search budget (50 trials over base WD and gradient-norm scale factor)
4. **Alignment-aware SGDW:** λ_t = clip(c · γ_t · (1 - δ̂_t), λ_min, λ_max). Same search budget (50 trials over c, λ_min, λ_max)
5. **CPR (Franke et al., NeurIPS 2024):** Augmented Lagrangian per-matrix constraint. Same search budget (50 trials over 2 hyperparameters — CPR's natural advantage)

Each method gets EXACTLY the same wall-clock search budget (measured in GPU-hours), not just the same number of Optuna trials.

### Evaluation Protocol

**Primary benchmarks (in priority order per project constraints):**
1. ResNet-50 / ImageNet-1K (mandatory, highest priority)
2. VGG-16-BN / CIFAR-100 (secondary)
3. ResNet-20 / CIFAR-10 (secondary)
4. ResNet-20 / CIFAR-100 (secondary)

**Metrics with statistical test plan:**
- Top-1 test accuracy at final epoch (matched FLOPs budget)
- Top-5 test accuracy (ImageNet only)
- Gradient norm stability (variance of ‖g_t‖ over last 10% of training)
- Weight norm trajectory (‖w_t‖ at 25%, 50%, 75%, 100% of training)
- Gradient-to-weight ratio (‖g_t‖/‖w_t‖ per-layer at same checkpoints)
- OUI diagnostic (log throughout as confound detection)

**Statistical tests:**
- Paired t-test (3 seeds, paired across methods with same initialization) with p < 0.05 (two-sided)
- 95% bootstrap CI (B=1000) on the accuracy difference (dynamic WD - fixed WD)
- Bonferroni correction for 4 primary benchmark comparisons per method: α_corrected = 0.05/4 = 0.0125
- Pre-specified primary endpoint: ResNet-50/ImageNet-1K (no correction needed for this one)

**Minimum seeds:** 3 (42, 123, 456). Report mean ± std for all metrics.

### Ablation Schedule

| Ablation | Variable Isolated | Expected Outcome | Informative If |
|----------|-----------------|------------------|----------------|
| Fixed WD with 5× search budget (250 Optuna trials) | Whether dynamic WD advantage survives with more fixed WD tuning | Fixed WD gap narrows or closes | Dynamic WD advantage is large and robust vs. search effort |
| Training budget sensitivity: 50%, 100%, 150% of standard epochs | Whether dynamic WD advantage scales with training duration | Dynamic WD advantage appears/grows at longer training | Long-horizon experiments are where dynamic WD matters |
| Alignment signal granularity: global vs. per-layer δ̂_t | Whether layer-specific alignment adds value over global alignment | Per-layer alignment provides marginal improvement | Spatial resolution of alignment signal matters |
| EMA smoothing: none vs. EMA-0.9 vs. EMA-0.99 | Whether smoothing the alignment signal improves stability | EMA-0.9 is better than raw; EMA-0.99 is similar to scheduling | Noise level of δ̂_t is a limiting factor |
| CPR as additional baseline (no alignment signal) | Whether per-matrix constraint adaptation (no alignment) already captures dynamic WD benefits | CPR matches or exceeds alignment-aware methods | Alignment-awareness is not the source of dynamic WD benefit |

### Control Experiments

1. **Rotation equilibrium control:** Train the full benchmark suite with "WD that equalizes per-layer rotation" (matching the rotational equilibrium target of Kosson et al.) as an alternative adaptive WD baseline. If rotation-equalizing WD performs similarly to alignment-aware WD, then the alignment mechanism is not the operative explanation.

2. **Phase-schedule control:** For each dynamic WD method, create a "fixed schedule" version that plays back the average λ_t trajectory from a successful run as a fixed schedule — without computing the alignment signal in real-time. If fixed-schedule playback performs similarly to online adaptive WD, then the timing of WD application (not the alignment signal) is the actual driver of performance.

3. **Noise injection control:** Add Gaussian noise with σ = std(δ̂_t) to the computed alignment signal before using it for WD modulation. If noisy alignment performs similarly to clean alignment, the alignment signal is not actually being used effectively by the algorithm (the method works "despite" the alignment signal, not because of it).

4. **Gradient-norm-only control:** Replace δ̂_t with ‖g_t‖ normalized to [0,1] as the WD modulation signal (same functional form, same computational cost, but using gradient norm instead of alignment). This directly tests whether the alignment term in δ̂_t provides value beyond what the gradient norm alone provides.

### Pilot Design (< 15 minutes)

**Run on:** ResNet-20/CIFAR-10, 100 epochs (reduced from 200), batch size 128, lr=0.1 with step decay at epochs 50 and 75 (×0.2).

**What to measure:**
1. Log δ̂_t, ‖g_t‖, ‖w_t‖, ‖g_t‖/‖w_t‖ per step
2. Train 3 runs: fixed WD (λ=5e-4), CWD (λ=5e-4), alignment-aware SGDW (λ_t = clip(0.01 · γ_t · (1-δ̂_t), 1e-5, 1e-2))
3. Plot alignment signal trajectory and check SNR (is δ̂_t varying meaningfully or flat?)
4. Check final test accuracy gap between methods

**Go/no-go criterion:** If δ̂_t variance is < 0.01 (effectively flat, no usable signal), the alignment-aware WD methods cannot function and the experiment plan needs redesign. If the three methods all converge to within 0.1% of each other, this is early evidence supporting the null hypothesis (fixed WD matches dynamic WD).

**Estimated time:** ~12 minutes on one RTX PRO 6000 Blackwell (based on typical ResNet-20/CIFAR-10 throughput of ~200 epochs/hour for 3 parallel runs).

### Resource Estimate

| Experiment | Architecture | Dataset | Seeds | Optuna Trials | GPU-hours |
|------------|-------------|---------|-------|---------------|-----------|
| Pilot | ResNet-20 | CIFAR-10 | 3 | N/A | 0.5 |
| Alignment info test (Phase A) | ResNet-20, VGG-16-BN | CIFAR-10/100 | 3 each (6 configs × 3) | N/A | 2 |
| Budget equivalence (CIFAR) | ResNet-20, VGG-16-BN | CIFAR-10/100 | 3 per trial, 50 trials | 5 × 50 | ~30 |
| Budget equivalence (ImageNet) | ResNet-50 | ImageNet-1K | 3 per trial, 50 trials per method | 5 × 50 | ~200 |
| Controls (rotation, phase-schedule, noise, gradient-norm) | ResNet-20 | CIFAR-100 | 3 | N/A | 4 |
| **Total** | | | | | **~236 GPU-hours** |

With 8 × RTX PRO 6000 Blackwell GPUs: estimated wall-clock ~30 hours for CIFAR experiments (parallelizable), ~25 hours per ImageNet search configuration (dominant cost). Total ImageNet wall-clock: 5 methods × 50 trials × 3 seeds / 8 GPUs ≈ the largest cost driver. Recommend staged approach: run CIFAR experiments first to identify which 2-3 dynamic WD methods are competitive, then run only those on ImageNet.

**Per-experiment target:** CIFAR experiments: < 1 hour per training run. ImageNet (ResNet-50, 90 epochs): ~4-6 hours per run on 1 GPU. Use 4-GPU data-parallel for ImageNet to reduce wall-clock to ~1.5 hours per run.

### Risk Assessment

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|-----------|
| δ̂_t minibatch SNR too low to be useful as WD signal | High | Moderate | Pilot design detects this before committing to full experiment; fallback: test EMA-smoothed alignment |
| Budget-equivalent comparison shows null result (fixed WD ≈ dynamic WD) | Moderate | Moderate | Null result is publishable as a rigorous negative finding; reframe paper as "evaluation framework" contribution |
| ImageNet experiments exceed time budget | High | Low | Use 4-GPU data-parallel; reduce to 50 ImageNet epochs for search (full 90 epochs for final best configurations only) |
| CPR baseline dominates all alignment-aware methods | Moderate | Low | If CPR wins, it validates the "per-parameter adaptation without alignment signal" approach; shifts paper focus to CSI metric as CPR's characterization |
| Alignment dynamics different at ImageNet scale | Moderate | Moderate | Run alignment info diagnostic (Phase A) on ResNet-50/ImageNet with 3 WD values before committing to full search |

### Novelty Claim

This paper answers the following empirical questions for the first time:
1. **Does alignment-aware WD provide information beyond gradient norm and weight norm?** (Candidate A, operationalized via AIS metric)
2. **Does any dynamic WD method beat optimally-tuned fixed WD at matched compute budget?** (Candidate B, the Budget Equivalence test)
3. **What is the gradient-to-weight ratio trajectory under each major WD strategy, and does it explain the performance differences?** (Control experiment, operationalized via CSI metric)

The experimental contribution is a **standardized evaluation protocol** (BEM + CSI + AIS) that resolves the comparison fragmentation problem in the existing dynamic WD literature. The paper's claim is not "our method is better" but rather "here is how to correctly compare all methods, and here are the results of that correct comparison."

This is methodologically analogous to the role played by GLUE/SuperGLUE in NLP: not proposing a new method, but establishing the ground truth for what "better" means in a fragmented research area.
