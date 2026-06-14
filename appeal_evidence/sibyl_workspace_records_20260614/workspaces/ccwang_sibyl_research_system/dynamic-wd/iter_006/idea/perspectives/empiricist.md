# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Xie et al. (NeurIPS 2023) -- "On the Overlooked Pitfalls of Weight Decay"** -- First practical WD scheduler (SWD). Key evaluation insight: WD causes gradient norm inflation at end of training; evaluation must track gradient norms, not just final accuracy. Provides gradient-norm-aware dynamic WD baseline.

2. **D'Angelo et al. (NeurIPS 2024) -- "Why Do We Need Weight Decay in Modern Deep Learning?"** -- Most rigorous experimental study: WD is never useful as explicit regularization in ResNets/LLMs; it changes training dynamics (loss stabilization for SGD, bias-variance tradeoff for LLMs). **Critical confounder identified**: WD prevents bfloat16 loss divergence, so some WD benefits are numerical, not algorithmic.

3. **Holzl et al. (NeurIPS 2025) -- "Gradient-Weight Alignment as a Train-Time Proxy for Generalization"** -- GWA predicts generalization without validation sets. Stable signal, efficiently computable. **Key methodology contribution**: establishes that alignment cosine similarity is a meaningful metric, not just noise. Average pairwise gradient alignment has high variance; GWA is smoother.

4. **Sun et al. (CVPR 2025) -- "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"** -- First theoretical proof that WD improves generalization (not convergence). Experimental section uses fixed WD only; no ablation of dynamic schedules. Alignment quantity delta_T measured but not exploited.

5. **Chen et al. (ICLR 2026) -- "Cautious Weight Decay (CWD)"** -- Binary sign-alignment mask. Evaluation: reports final accuracy but **no trajectory analysis** of when/where CWD helps. Missing ablation: random mask baseline to test whether sign-alignment is causal or incidental.

6. **Wang & Aitchison (arXiv:2405.13698) -- "How to set AdamW's weight decay as you scale"** -- WD as EMA timescale; optimal timescale constant across scales. **Evaluation pitfall identified**: comparing WD methods at different compute budgets without normalization is meaningless.

7. **Kosson et al. (arXiv:2510.19093) -- "Weight Decay may matter more than muP for LR Transfer"** -- WD stabilizes update dynamics across widths. **Critical confounder**: WD's benefit may be primarily about making LR transfer work, not about regularization per se.

8. **Chou (arXiv:2512.08217) -- "Correction of Decoupled Weight Decay"** -- WD proportional to gamma^2 stabilizes weight norm. Total Update Contribution (TUC) analysis. **Methodological contribution**: introduces a principled way to decompose WD's effect on training dynamics.

9. **Defazio (arXiv:2506.02285) -- "Why Gradients Rapidly Increase Near the End of Training"** -- WD controls gradient-to-weight ratio ||g||/||w||; all normalized layers converge to same steady-state. **Critical evaluation insight**: For BN architectures, WD's role is purely an effective-LR mechanism, NOT regularization. This means alignment-aware WD may be uninformative for BN networks.

10. **Previous iteration experiments (iter_005)** -- VGG16-BN/CIFAR-10 results across 7 WD methods with 3 seeds each. **The most relevant empirical evidence we have.** Details below.

### Experimental Landscape

**What has been properly tested:**
- Fixed WD vs no-WD on standard benchmarks (extensively, by dozens of papers)
- CWD binary mask on CIFAR-10/100 and LLM training (Chen et al. ICLR 2026)
- SWD gradient-norm-aware scheduling on CIFAR-10/100 (Xie et al. NeurIPS 2023)
- WD scaling rules across model sizes (Wang & Aitchison 2024)

**What is accepted WITHOUT proper evidence:**
- That alignment-aware WD provides meaningful improvement over simpler schedules (CWD paper shows modest gains but no random-mask control)
- That continuous alignment modulation beats binary masking (never tested)
- That proposed evaluation metrics (BEM, CSI, AIS) actually discriminate between methods better than simple test accuracy
- That WD scheduling matters for BN architectures (our iter_005 data strongly suggests it does not)

**Critical methodological gap:**
- No study has simultaneously compared all major WD approaches (constant, cosine, SWD, CWD, alignment-aware continuous) on the same benchmarks with the same seeds and the same compute budget, with proper statistical testing.

---

## Phase 2: Initial Candidates

### Candidate A: "The Null Result Paper" -- Controlled Comparison Showing WD Method Choice Barely Matters with Batch Normalization

- **Hypothesis**: For architectures with batch normalization (the vast majority of modern CNNs), the choice of WD method (constant, scheduled, alignment-aware, cautious) does NOT significantly affect test accuracy (p > 0.05 on paired t-test across seeds), because BN reduces WD's role to an effective-LR modifier where alignment information is uninformative.
- **Falsification criterion**: If any dynamic WD method achieves statistically significant improvement (p < 0.05, paired t-test, N >= 5 seeds) over constant WD on at least 2 of 3 BN architectures (ResNet-20, VGG-16-BN, ResNet-50), the hypothesis is falsified.
- **Evaluation protocol**: CIFAR-10, CIFAR-100, ImageNet; ResNet-20, VGG-16-BN, ResNet-50; 5 seeds; paired t-test and bootstrap 95% CI. Methods: no-WD, constant, cosine, SWD, CWD, random-mask (control), alignment-aware continuous.
- **Ablation plan**: (1) Remove BN and re-run -- does WD method choice matter more? (2) Vary WD strength (5e-4 to 5e-2) -- does method choice interact with WD magnitude? (3) Random mask vs CWD -- is sign-alignment causal?
- **Confounders**: (a) LR schedule interaction -- different WD methods may need different LR schedules. (b) WD magnitude tuning -- some methods implicitly change effective WD. (c) Training length -- short training may mask WD effects.
- **Pilot design**: ResNet-20 / CIFAR-10, 3 methods (constant, CWD, random-mask), 50 epochs, 3 seeds. ~10 min.

### Candidate B: Alignment Informativeness Score Validation -- When Does Gradient-Weight Alignment Actually Help WD Decisions?

- **Hypothesis**: The informativeness of gradient-weight alignment for WD decisions depends critically on architecture (BN vs no-BN) and training phase (early vs late). Specifically: alignment is informative (AIS > 0.5) only for non-BN architectures in the mid-to-late training phase (epoch > 30% of total), and uninformative (AIS < 0.3) for BN architectures throughout training.
- **Falsification criterion**: If alignment-aware WD outperforms constant WD by > 0.5% test accuracy on BN architectures consistently across seeds, or if AIS > 0.5 is observed for BN architectures, the hypothesis is falsified.
- **Evaluation protocol**: Track per-epoch AIS (correlation between alignment signal and generalization gap change), CSI (weight norm trajectory stability), BEM (compute-normalized comparison). Run on ResNet-20 (BN and no-BN variants), VGG-16 (BN and no-BN), CIFAR-10/100.
- **Ablation plan**: (1) AIS computation: cosine similarity vs sign-agreement vs magnitude-weighted alignment. (2) Temporal resolution: per-epoch vs per-batch AIS. (3) Layer-wise vs global alignment.
- **Confounders**: (a) AIS may correlate with training phase, not actually predict optimal WD. (b) CSI may be dominated by LR schedule effects, not WD.
- **Pilot design**: ResNet-20 BN vs no-BN, CIFAR-10, constant WD, track all alignment metrics for 200 epochs. ~15 min per run.

### Candidate C: Budget Equivalence Metric -- Fair Comparison Framework for Dynamic WD Methods

- **Hypothesis**: When methods are compared at equal compute budget (FLOPs, wall-clock, or gradient evaluations), the apparent advantages of complex WD methods shrink dramatically. Specifically, the gap between best and worst method (excluding no-WD) is < 0.3% test accuracy on CIFAR-10/100 with BN architectures when budget is equalized.
- **Falsification criterion**: If budget-equalized comparison still shows > 0.5% test accuracy gap between methods on at least 2 benchmarks, the hypothesis is falsified.
- **Evaluation protocol**: Run all methods to identical FLOPs (accounting for CWD's ~25% overhead from mask computation). Normalize by wall-clock. Report BEM-adjusted accuracy. CIFAR-10, CIFAR-100, ImageNet with ResNet-20, VGG-16-BN, ResNet-50.
- **Ablation plan**: (1) FLOPs-equalized vs wall-clock-equalized vs gradient-evaluation-equalized. (2) Fixed total compute vs fixed epochs.
- **Confounders**: (a) GPU utilization varies between methods (CWD adds overhead). (b) Some methods converge faster to lower-quality solutions.
- **Pilot design**: Measure wall-clock per epoch for each method on ResNet-20/CIFAR-10. ~5 min.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Confound attack**: The LR schedule is fixed across all methods, but different WD methods may need different LR schedules to reach their full potential. CWD paper uses the same LR schedule as baseline, which is fair but may understate CWD's potential. **Mitigation**: Add a grid search over LR for each WD method, or use a small sweep.
- **Statistical attack**: With 5 seeds, the power to detect a 0.3% accuracy difference (plausible effect size) at p=0.05 is low. Standard deviation across seeds is ~0.1-0.3% based on iter_005 data, so detecting 0.3% differences requires ~10+ seeds. **Mitigation**: Increase to 10 seeds, or accept that we can only detect differences > 0.5%.
- **Benchmark attack**: CIFAR-10 may be too saturated to show WD effects. VGG-16-BN already hits 92%+ where improvements are marginal. ImageNet would be more informative but expensive. **Mitigation**: Include CIFAR-100 (more room for improvement) and at least one ImageNet run.
- **Ablation completeness attack**: The no-BN ablation is essential but may introduce confounders (different optimal LR, different convergence behavior). Need to carefully match hyperparameters. **Mitigation**: Use matched-rho protocol from previous iterations.
- **Verdict**: **STRONG** -- A well-designed null result paper would be genuinely valuable. The community needs to know if WD method choice matters or if it is a waste of research effort for BN architectures. The risk is that the paper is "boring" if the null hypothesis holds. But controlled negative results are publishable (see "Do Deep Nets Really Need Weight Decay and Dropout?").

### Against Candidate B

- **Confound attack**: AIS is a proposed metric without external validation. We are defining a metric and then testing it -- this is circular unless we establish external ground truth. **Mitigation**: Use generalization gap as external ground truth; AIS is useful only if it predicts generalization gap changes better than simpler proxies (e.g., weight norm).
- **Statistical attack**: Correlation between AIS and generalization gap may be driven by training phase (both change monotonically), not by genuine causal relationship. **Mitigation**: Use partial correlation controlling for epoch number, or use time-series methods.
- **Benchmark attack**: Testing AIS only on CIFAR-10/100 limits generalizability claims. But extending to ImageNet significantly increases cost. **Mitigation**: Start with CIFAR, extend to ImageNet if signal is promising.
- **Ablation completeness attack**: Multiple AIS definitions (cosine, sign, magnitude-weighted) make the analysis fragile -- easy to p-hack by choosing the definition that works best. **Mitigation**: Pre-register the primary AIS definition before running experiments. Use cosine similarity (most standard) as primary, others as secondary.
- **Verdict**: **MODERATE** -- The AIS validation is scientifically sound but risks being unfocused. The BN vs no-BN distinction is the most interesting finding, but it may already be predicted by the effective-LR theory (Defazio 2025). The novelty is empirical confirmation, not new insight.

### Against Candidate C

- **Confound attack**: Budget equivalence is tricky to define fairly. CWD adds ~25% wall-clock overhead due to mask computation; if you give CWD 25% fewer epochs to equalize wall-clock, it trains less and appears worse. But if you equalize epochs, CWD uses more compute. **Mitigation**: Report both FLOPs-equalized and epoch-equalized results; let the reader decide.
- **Statistical attack**: The 0.3% threshold is arbitrary. Why not 0.5% or 0.1%? **Mitigation**: Report actual effect sizes with CIs rather than testing against arbitrary thresholds.
- **Benchmark attack**: Same as Candidate A -- CIFAR-10 may be too saturated. **Mitigation**: Include CIFAR-100 and ImageNet.
- **Ablation completeness attack**: This candidate is primarily a benchmarking exercise, not a scientific investigation. It is useful but not novel enough for a top venue. **Mitigation**: Combine with Candidate A/B as the evaluation framework for a unified paper.
- **Verdict**: **MODERATE** -- BEM is a useful tool but insufficient as a standalone contribution. Best combined with other candidates.

---

## Phase 4: Refinement

### Dropped
- **Candidate C as standalone** -- Budget equivalence is important but not a paper on its own. Integrated into the evaluation framework.

### Strengthened and Merged

The strongest approach combines Candidates A and B into a single, rigorous empirical investigation:

**Merged Proposal: "Does Alignment-Aware Weight Decay Actually Help? A Controlled Empirical Study"**

Key strengthening:
1. **Added random-mask control** (from Candidate A) to test whether CWD's sign-alignment is causal. This is the single most important missing control in the CWD literature. If random masking performs equally well, CWD's alignment mechanism is not the source of its benefit -- the benefit comes from simply applying WD to a random subset (implicit regularization reduction).
2. **Added BN vs no-BN comparison** (from Candidate B) as the primary experimental axis. The hypothesis is that alignment-based WD is informative only without BN.
3. **Pre-registered AIS metric** using cosine similarity, with power analysis showing we need 5+ seeds to detect meaningful differences.
4. **Budget equivalence normalization** (from Candidate C) applied to all comparisons.
5. **ImageNet validation** on ResNet-50 to ensure findings generalize beyond CIFAR-scale.

### Selected Front-Runner: The merged proposal above.

**Key empirical predictions (to be decided BEFORE running experiments):**
1. On BN architectures (VGG-16-BN, ResNet-20-BN), no WD method will outperform constant WD by more than 0.3% (mean over seeds).
2. On non-BN architectures, alignment-aware WD will outperform constant WD by 0.5-1.0%.
3. Random-mask WD will perform within 0.2% of CWD on BN architectures (CWD's alignment is uninformative with BN).
4. Random-mask WD will perform 0.3-0.5% worse than CWD on non-BN architectures (alignment IS informative without BN).
5. AIS will be < 0.3 for BN architectures and > 0.5 for non-BN architectures in mid-to-late training.

---

## Phase 5: Final Proposal

### Title
Does Alignment-Aware Weight Decay Actually Help? A Controlled Empirical Study with Batch Normalization as the Critical Confounder

### Hypothesis
The effectiveness of alignment-aware weight decay methods (CWD, continuous alignment WD) depends critically on whether the architecture uses batch normalization. For BN architectures, alignment information is uninformative for WD decisions because BN projects weight dynamics into the angular domain where WD acts purely as an effective-LR modifier. For non-BN architectures, alignment information is informative and alignment-aware WD provides genuine generalization benefits.

**Precise falsification**: This hypothesis is falsified if alignment-aware WD outperforms constant WD by more than 0.5% test accuracy (paired t-test p < 0.05, N = 5 seeds) on at least 2 of 3 BN architectures (ResNet-20-BN, VGG-16-BN, ResNet-50).

### Falsification Criterion
- BN architectures: alignment-aware WD achieves > 0.5% improvement over constant WD with p < 0.05 on >= 2/3 architectures
- Non-BN architectures: alignment-aware WD fails to outperform constant WD (< 0.3% improvement) -- this would falsify the second part of the hypothesis

### Method
- **7 WD methods**: no-WD, constant, cosine-schedule, SWD (Xie et al.), CWD (Chen et al.), random-mask (control), alignment-aware continuous (lambda_t = c * gamma_t * (1 - cos_sim(g_t, w_t)))
- **4 architectures**: ResNet-20-BN, ResNet-20-noBN, VGG-16-BN, ResNet-50 (ImageNet)
- **3 datasets**: CIFAR-10, CIFAR-100, ImageNet
- **Training protocol**: SGD with momentum 0.9, initial LR 0.1, cosine LR schedule to 0, 200 epochs (CIFAR), 90 epochs (ImageNet)
- **WD base rate**: 5e-4 (standard), with sensitivity sweep at 1e-4 and 1e-3
- **Metrics tracked per epoch**: train/test loss, train/test accuracy, weight norm (global and per-layer), gradient-weight alignment (cosine similarity, global and per-layer), CSI, AIS, BEM, generalization gap, effective learning rate

### Evaluation Protocol
- **Primary benchmarks**: CIFAR-10 (ResNet-20-BN, ResNet-20-noBN, VGG-16-BN), CIFAR-100 (ResNet-20-BN, VGG-16-BN), ImageNet (ResNet-50)
- **Primary metric**: Test accuracy at final epoch; secondary: best test accuracy during training
- **Statistical tests**: Paired t-test (method vs constant-WD baseline) for primary comparison; Wilcoxon signed-rank test as nonparametric backup; bootstrap 95% CI for effect size estimation
- **Number of seeds**: 5 (seeds: 42, 123, 456, 789, 1024)
- **Multiple comparison correction**: Bonferroni correction for 6 method comparisons per architecture

### Ablation Schedule

| Ablation | Tests | Expected Outcome |
|----------|-------|------------------|
| Remove BN from ResNet-20 | BN vs no-BN effect on alignment informativeness | AIS increases from < 0.3 to > 0.5 without BN |
| Random mask vs CWD | Is sign-alignment causal or incidental? | Equivalent for BN; CWD wins without BN |
| Vary WD base rate (1e-4 to 1e-2) | Does method choice interact with WD magnitude? | Small WD: all equivalent; Large WD: alignment matters more |
| Layer-wise vs global alignment | Does per-layer AIS provide better signal? | Per-layer AIS more informative for deep networks |
| Cosine vs sign vs magnitude-weighted alignment | Which alignment definition is most predictive? | Cosine (pre-registered primary); sign is CWD's choice |
| Training length (50, 100, 200 epochs) | Does WD method choice matter more with longer training? | Effects diminish with longer training (convergence dominates) |

### Control Experiments

1. **Random-mask control**: Apply WD to a random 50% of parameters (matching CWD's average mask ratio of ~0.49 from iter_005 data). This directly tests whether CWD's benefit comes from alignment-based selection vs simply applying WD to fewer parameters.
2. **Shuffled-alignment control**: Compute alignment per parameter but shuffle the mask across layers. Tests whether layer-level alignment structure matters.
3. **Fixed-alignment control**: Compute alignment at epoch 0 and use that fixed mask for all training. Tests whether temporal adaptation of the mask is important.
4. **Matched effective-WD control**: Adjust constant WD to match the mean effective WD of each dynamic method. Tests whether dynamic methods simply find a better average WD strength.

### Pilot Design
- **Phase 1** (~10 min): ResNet-20/CIFAR-10, 50 epochs, 3 methods (constant, CWD, random-mask), 3 seeds. Goal: verify random-mask control is implementable and produces sensible results.
- **Phase 2** (~15 min): Add alignment-aware continuous WD to Phase 1. Goal: verify alignment tracking code works and produces stable AIS values.
- **Phase 3** (~15 min): ResNet-20-noBN/CIFAR-10, same methods, 50 epochs. Goal: verify BN vs no-BN hypothesis shows preliminary signal.

### Resource Estimate
- CIFAR-10/100 experiments: ~20 min per run x 7 methods x 5 seeds x 4 architectures = ~47 GPU-hours
- ImageNet experiments: ~4 hours per run x 7 methods x 5 seeds = ~140 GPU-hours
- Total: ~187 GPU-hours on 8x RTX PRO 6000 Blackwell = ~24 hours wall-clock
- Ablation experiments: additional ~50 GPU-hours
- **Total estimated**: ~240 GPU-hours (~30 hours wall-clock with 8 GPUs)

### Risk Assessment

1. **Biggest threat**: The null result on BN architectures is confirmed but the community considers it "obvious" and not publishable. **Mitigation**: Frame as a diagnostic study that saves the community from pursuing unproductive alignment-aware WD for BN architectures, while redirecting effort toward non-BN settings (ViTs, MLPs) where it matters.

2. **Second threat**: The effect sizes are so small that no conclusion can be drawn with affordable seed counts. Based on iter_005 data, VGG-16-BN test accuracy std across seeds is ~0.1-0.3%, so detecting 0.3% differences requires impractical sample sizes. **Mitigation**: Focus on non-BN architectures where effects are larger, and report effect sizes with CIs rather than p-values.

3. **Third threat**: Alignment-aware WD turns out to help on ImageNet even with BN, contradicting CIFAR results. This would complicate the narrative. **Mitigation**: This would actually be an interesting finding -- it would suggest alignment informativeness depends on dataset complexity, not just architecture.

4. **Fourth threat**: Our alignment-aware continuous WD implementation may be suboptimal (wrong clipping, wrong base rate). **Mitigation**: Test multiple parameterizations in ablation; compare against published CWD implementation.

### Novelty Claim
The experimental contribution answers three specific empirical questions for the first time:
1. **Is CWD's sign-alignment causal?** (Random-mask control experiment -- never reported in the literature)
2. **Does batch normalization render alignment-based WD uninformative?** (BN vs no-BN controlled comparison with alignment tracking -- predicted by Defazio's theory but never experimentally confirmed)
3. **How informative is gradient-weight alignment for WD decisions?** (AIS metric quantifying the predictive value of alignment for generalization gap changes -- first systematic measurement)

### Grounding in Prior Results

From iter_005 data (VGG16-BN, CIFAR-10, 200 epochs, seeds 42/123/456):

| Method | Test Acc (mean +/- std) | Weight Norm | Gen Gap |
|--------|------------------------|-------------|---------|
| no_wd | 91.96 +/- 0.09 | 661.3 | 8.02 |
| constant (5e-4) | 91.98 +/- 0.07 | 650.7 | 8.00 |
| cwd_hard | 92.02 +/- 0.28 | 656.2 | 7.96 |
| cosine_schedule | 91.89 +/- 0.30 | 653.6 | 8.08 |
| swd | 92.02 +/- 0.24 | 658.2 | 7.95 |
| half_lambda | 92.06 +/- 0.13 | 654.4 | 7.92 |
| random_mask | 91.96 +/- 0.36 | 657.8 | 8.02 |

**Key observation**: The spread across methods is 0.17% (91.89 to 92.06), while within-method seed variance is 0.07-0.36%. **No method is statistically distinguishable from constant WD.** This is the core empirical finding that motivates the entire proposal.

From iter_005 data (ResNet20-noBN, CIFAR-10):

| Method | Test Acc (mean +/- std) | Weight Norm | Alignment |
|--------|------------------------|-------------|-----------|
| no_wd | 87.35* | 50.4 | 0.095 |
| constant (5e-4) | 87.49 +/- 0.23 | 50.7 | 0.114 |
| cwd_hard | 87.46 +/- 0.19 | 50.7 | 0.088 |

(*incomplete data for some seeds)

**Key observation**: Without BN, alignment values are 2x higher (~0.10 vs ~0.05 for BN networks), supporting the hypothesis that alignment is more informative without BN. However, absolute accuracy differences remain small.

### What This Means for the Unified Framework Paper

If the empirical findings confirm the hypothesis:
- The "unified dynamic WD framework" paper should clearly delineate **when** each sub-approach matters (BN vs no-BN, dataset scale, training length)
- The proposed metrics (BEM, CSI, AIS) should be validated as diagnostic tools that predict WHEN to use which WD method, not just as abstract measures
- The theoretical contributions (Lyapunov certificates, alignment-weighted convergence) remain valid but their practical impact is limited to non-BN architectures
- The paper's primary value shifts from "here is a better WD method" to "here is a principled framework for understanding when WD method choice matters" -- which is arguably a more valuable contribution
