# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Choi, Shallue, Nado, Lee, Maddison & Dahl, 2019. "On Empirical Comparisons of Optimizers for Deep Learning." arXiv:1910.05446** -- The definitive paper on fair optimizer comparison. Key finding: "the hyperparameter search space may be the single most important factor explaining the rankings obtained by recent empirical comparisons." Shows that optimizer rankings can be completely reversed by changing search spaces. Establishes the "inclusion relationship" principle: more general optimizers (Adam) should never underperform the ones they approximate (momentum) if tuned properly. **Critical implication for our WD comparison**: every dynamic WD method must be compared under equal hyperparameter tuning effort, and methods that subsume others (e.g., CWD with mask=1 reduces to standard WD) must satisfy inclusion consistency.

2. **Semenov, Pagliardini & Jaggi, 2025. "Benchmarking Optimizers for Large Language Model Pretraining." arXiv:2509.01440** -- Most recent comprehensive optimizer benchmark. Systematically varies model size (125M-1.5B), batch size, and training duration. Emphasizes that "diverse experimental protocols used to validate claims make direct comparisons between methods challenging." Provides a reproducible evaluation framework. **Key lesson**: standardized experimental protocol matters more than any individual result.

3. **D'Angelo, Andriushchenko, Varre & Flammarion, 2024. "Why Do We Need Weight Decay in Modern Deep Learning?" NeurIPS 2024, arXiv:2310.04415** -- The most rigorous experimental study of WD mechanisms. Uses ResNet-18/VGG-16-BN on CIFAR-10/100 and ResNet-50 on ImageNet. Key experimental findings: (a) WD is never useful as explicit regularization -- it acts as a training dynamics modifier, (b) for SGD, WD provides "loss stabilization" preventing training divergence, (c) for LLMs, WD facilitates a bias-variance tradeoff, (d) WD prevents bfloat16 loss divergence. **Methodological model**: separate the mechanistic question (what does WD do?) from the performance question (does it improve accuracy?).

4. **Chen, Yi, Huang, He, Luo, Xiao, Lee & Sugiyama, 2025. "Cautious Weight Decay (CWD)." ICLR 2026, arXiv:2510.12402** -- Most methodologically thorough recent WD paper. Extensive hyperparameter grid search (24 LR values, 7 warmup ratios, beta sweep per optimizer) before adding CWD. Tests across 3 optimizers (AdamW, Lion, Muon), multiple scales (111M-2.3B), and reports mask activation patterns, parameter norm evolution, and per-layer behavior. **Methodological strengths**: fair baseline tuning before testing the modification. **Methodological weaknesses**: no confidence intervals or statistical tests on reported improvements; improvements of 0.1-0.6% may not be statistically significant at the reported scale.

5. **Xie, Ma, Li & He, 2020/2023. "On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them (SWD)." NeurIPS 2023, arXiv:2011.11152** -- First paper to systematically identify WD confounds in optimizer evaluation. Key finding: SGD with lambda=0.0001 (common default) is often a poor baseline; lambda=0.0005 shows better generalization on CIFAR. The "pitfall" is that unfair WD tuning creates misleading optimizer comparisons. **Critical implication**: any WD comparison must sweep WD values for each method independently, not use a shared default.

6. **Hernandez-Garcia & Konig, 2018/2019. "Do Deep Nets Really Need Weight Decay and Dropout?" arXiv:1802.07042** -- Ablation study concluding WD and dropout are unnecessary for object recognition with sufficient data augmentation. Demonstrates that WD benefits are confounded with data augmentation choices. **Methodological lesson**: data augmentation strategy is a major confound in WD experiments; it must be held constant across all comparisons.

7. **Xiong, Liu, Lan, You, Si & Hsieh, 2020. "How Much Progress Have We Made in Neural Network Training? A New Evaluation Protocol for Benchmarking Optimizers." arXiv:2010.09889** -- Proposes bandit-based hyperparameter tuning protocol and data-addition training efficiency metric. Shows no optimizer wins across all tasks. **Methodological contribution**: end-to-end efficiency (including tuning cost) is the right metric, not best-case performance after unlimited tuning.

8. **Wen et al., 2025. "Fantastic Pretraining Optimizers and Where to Find Them." arXiv:2509.02046** -- Reality check: optimizer speedups over well-tuned AdamW shrink from 1.4x at 0.1B to 1.1x at 1.2B. **Critical baseline**: any WD modification claiming improvements must be evaluated at sufficient scale; small-scale improvements may vanish.

9. **Fernandez-Hernandez et al., 2025. "OUI: Overfitting-Underfitting Indicator." arXiv:2504.17160** -- Proposes OUI metric for monitoring WD quality during training. Validated on DenseNet-BC-100/CIFAR-100, EfficientNet-B0/TinyImageNet, ResNet-34/ImageNet-1K. **Methodological contribution**: provides a validation-free diagnostic signal that could serve as one component of our proposed CSI metric.

10. **Debenedetti, Sehwag & Mittal, 2022. "A Light Recipe to Train Robust Vision Transformers." arXiv:2209.07399** -- Demonstrates through rigorous ablation that canonical ViT training recipes (strong data augmentation) become suboptimal under adversarial training, and larger weight decay becomes beneficial. **Methodological lesson**: the optimal WD depends on the full training recipe; changing any other component can shift the optimal WD.

11. **Wang & Aitchison, 2024. "How to Set AdamW's Weight Decay as You Scale." arXiv:2405.13698** -- Derives optimal WD from EMA timescale analysis. Finds optimal timescale constant in epochs across model/dataset scales. **Evaluation insight**: compute-normalized comparison must account for different optimal WD values at different scales.

12. **Schaipp, 2024. "How to Jointly Tune Learning Rate and Weight Decay for AdamW." Blog post.** -- Shows that in PyTorch's AdamW implementation, LR and WD are NOT truly decoupled; tuning strategy should be diagonal line search. **Critical implementation pitfall**: PyTorch AdamW behavior differs from the original paper's formulation. Any benchmark must verify the implementation matches the intended mathematical definition.

### Experimental Landscape

**What has been properly tested:**
- AdamW vs. Adam+L2: well-established that decoupled WD is superior for adaptive optimizers (Loshchilov & Hutter 2019, 10,000+ citations)
- WD mechanisms in ResNets/VGGs with BN on CIFAR: D'Angelo et al. (2024) provide the definitive study
- CWD on LLM pretraining: extensive evaluation at 111M-2.3B scale with fair baseline tuning
- WD scaling rules: Wang & Aitchison (2024) establish EMA timescale principle
- WD necessity: Hernandez-Garcia & Konig (2018) establish the data augmentation confound

**What is accepted without proper evidence:**
- That CWD's improvement over standard AdamW is statistically significant (no confidence intervals reported)
- That SWD's gradient-norm-aware scheduling generalizes beyond CIFAR (authors admit marginal ImageNet gains)
- That alignment signals (gradient-weight cosine) are informative at practical scale (this project's own pilots say no)
- That norm-matched WD (AdamWN) improves over fixed-target WD in practice (limited experimental validation in original paper)
- That any dynamic WD method provides meaningful improvement over well-tuned constant WD at scale > 1B params

**Where the methodological gaps are most severe:**
1. **No head-to-head comparison of all major WD variants** on the same codebase, same compute budget, same hyperparameter search protocol
2. **No statistical significance testing** in most WD papers -- improvements of 0.1-0.6% are reported without confidence intervals
3. **No compute-normalized comparison** -- some methods use more epochs or larger grids, making "improvements" an artifact of more tuning
4. **Confound between WD value and WD method** -- dynamic WD methods often use different effective WD strength (e.g., CWD applies less total decay because the mask blocks some parameters, so it may simply be equivalent to lower constant WD)
5. **No falsification protocol** -- papers test whether their method improves but never test the null hypothesis (their method is equivalent to properly tuned constant WD)


## Phase 2: Initial Candidates

### Candidate A: The Definitive Head-to-Head WD Benchmark with Null Hypothesis Testing

- **Core hypothesis (falsifiable)**: Under compute-controlled conditions (equal FLOPs, equal hyperparameter tuning budget, same codebase), no dynamic WD method achieves statistically significant improvement over optimally tuned constant WD on standard benchmarks (CIFAR-10/100, ImageNet) across multiple architectures and seeds.
- **Falsification criterion**: If any dynamic WD method achieves >0.3% improvement in mean test accuracy over the best constant WD baseline, with p < 0.05 (paired t-test across seeds), the null hypothesis is rejected.
- **Evaluation protocol**:
  - Benchmarks: CIFAR-10 (ResNet-20, VGG-16-BN), CIFAR-100 (ResNet-20, VGG-16-BN), ImageNet (ResNet-50, ViT-Small)
  - Methods: Constant AdamW, Constant SGD+WD, CWD-AdamW, SWD/AdamS, Cosine WD schedule, AdamWN (target-norm WD), CWD+Cosine-schedule combination
  - Metrics: Test accuracy (mean +/- std), training loss, weight norm trajectory, gradient-weight cosine alignment, effective learning rate, training FLOPs
  - Statistical tests: Paired t-test (accuracy across 5 seeds), bootstrap 95% CI, Wilcoxon signed-rank test as nonparametric backup
  - Seeds: 42, 123, 456, 789, 1024 (5 seeds for statistical power)
  - HP tuning: Grid search with **equal number of trials** (e.g., 20 configurations per method). For constant WD: sweep lambda in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} x LR grid. For dynamic methods: sweep their specific HPs with equal total grid size.
- **Ablation plan**:
  1. **WD strength ablation**: For CWD, measure the effective average WD applied (accounting for mask). Compare CWD at effective-average-WD vs. constant WD at same value. If they match, CWD's improvement is just from using less WD, not from alignment-awareness.
  2. **Random mask ablation**: Replace CWD's sign-alignment mask with a random binary mask (same sparsity level). If random mask performs comparably, the alignment signal is uninformative.
  3. **Schedule shape ablation**: For SWD/cosine WD, compare against constant WD with lambda = time-average of the schedule. If they match (as prior iterations suggest), scheduling adds nothing over the right constant value.
  4. **Architecture ablation**: Test whether WD method rankings change across architectures. If they do, there is no universally "best" dynamic WD method.
  5. **Normalization ablation**: Test with and without batch normalization. D'Angelo et al. show WD mechanisms differ fundamentally with BN. If method rankings change with/without BN, the methods address different phenomena and should not be unified.
- **Confounders identified**:
  1. **Unequal tuning effort**: Mitigate by fixing grid size per method
  2. **Effective WD strength**: Mitigate by measuring and comparing total applied WD
  3. **Data augmentation confound**: Hold augmentation constant (standard for each benchmark)
  4. **Implementation confound**: Use a single codebase (fork why-weight-decay) to eliminate implementation differences
  5. **LR schedule interaction**: Use identical LR schedule for all methods; separately test with cosine and step LR
  6. **BN/LN interaction**: Separately exclude bias and normalization parameters from WD for all methods
- **Pilot design**: CIFAR-10, ResNet-20, AdamW vs. CWD-AdamW vs. SWD, 3 seeds, 200 epochs. Test accuracy + effective WD + alignment cosine tracking. Time: ~15 min on 8 GPUs.

### Candidate B: Diagnostic Metric Validation -- Do BEM/CSI/AIS Predict Method Rankings?

- **Core hypothesis (falsifiable)**: The proposed standardized metrics (Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score) have predictive power: methods that rank higher on CSI achieve better generalization, and AIS values >0.1 (on a 0-1 scale) indicate that alignment-aware methods will outperform alignment-agnostic ones.
- **Falsification criterion**: If CSI and AIS do not correlate with test accuracy rankings (Spearman rho < 0.5 across all method-architecture combinations), the metrics are uninformative and should not be proposed as standards.
- **Evaluation protocol**:
  - Same benchmarks and methods as Candidate A
  - Additional metrics: BEM (normalize all comparisons to equal FLOPs), CSI (variance of weight norm trajectory + spectral condition number evolution + effective LR stability), AIS (mutual information between gradient-weight alignment signal and optimal WD direction, estimated via empirical frequency of correct CWD mask decisions)
  - Compute Spearman rank correlation between each metric and test accuracy across all methods
  - Perform cross-validation: compute metrics on CIFAR-10, predict rankings on CIFAR-100 and ImageNet
- **Ablation plan**:
  1. **CSI component ablation**: Which component of CSI (weight norm variance, spectral condition, ELR stability) is most predictive?
  2. **AIS sensitivity ablation**: How does AIS vary across training stages (early, mid, late)? Is alignment more informative at specific phases?
  3. **BEM sensitivity ablation**: Does BEM normalization change method rankings compared to raw accuracy?
  4. **Metric robustness**: Do metric rankings remain stable across seeds?
- **Confounders**: Metrics might correlate with accuracy simply because they capture WD strength (trivial correlation). Mitigate by partialing out total WD applied.
- **Pilot design**: Compute all three metrics for AdamW vs. CWD on CIFAR-10/ResNet-20 (same runs as Candidate A pilot). Check whether AIS > 0.1 and whether CSI correlates with final accuracy. Time: no additional compute (metrics computed from Candidate A pilot runs).

### Candidate C: The Controlled Falsification Experiment -- "Is CWD Just Lower Effective WD?"

- **Core hypothesis (falsifiable)**: CWD (ICLR 2026) achieves its reported improvements not through alignment-awareness but through the trivially equivalent mechanism of applying less total weight decay (since the mask blocks WD on some parameters). A constant WD with lambda_eff = lambda * mean(mask_ratio) will match CWD's performance.
- **Falsification criterion**: If CWD outperforms the effective-lambda-matched constant WD by >0.2% accuracy (p < 0.05), then alignment-awareness contributes genuinely beyond WD reduction. If they are statistically indistinguishable, CWD's contribution is trivially explained.
- **Evaluation protocol**:
  - Same benchmarks as Candidate A
  - Method pairs: (CWD with lambda=0.01) vs. (Constant WD with lambda = 0.01 * empirical_mean_mask_ratio)
  - Measure mask ratio (fraction of parameters receiving WD) at each step; compute running average
  - Report mean +/- std accuracy across 5 seeds with paired t-test
- **Ablation plan**:
  1. **Mask ratio tracking**: Plot mask_ratio over training. If it is approximately constant, the effective-lambda explanation is clean. If it varies significantly, the temporal dynamics of the mask may matter independently.
  2. **Random mask with matched sparsity**: Apply random binary mask with same mean sparsity as CWD's mask. If random mask matches CWD, it is the sparsity (reduced total WD) that matters, not the alignment signal.
  3. **Inverted mask**: Apply WD only where CWD does NOT apply it (anti-alignment). If anti-alignment performs worse, the alignment signal has at least directional value.
  4. **Continuous alignment modulation**: Replace binary mask with cosine-similarity-weighted WD (lambda_i = lambda * max(0, cos(w_i, g_i))). If this improves over binary CWD, there is room for a better alignment-aware method.
- **Confounders**: CWD may work through a mechanism other than WD reduction or alignment (e.g., implicit regularization via sparse updates). Mitigate by tracking weight norm trajectories to confirm the effective-lambda hypothesis explains the observed norm behavior.
- **Pilot design**: CIFAR-10, ResNet-20, CWD-AdamW vs. matched-lambda-AdamW, 3 seeds, 200 epochs. ~10 min.


## Phase 3: Self-Critique

### Against Candidate A (Definitive Benchmark)

- **Confound attack**: The biggest confound is **hyperparameter tuning budget**. Choi et al. (2019) showed that search space is the single most important factor. Even with equal grid sizes, different methods have different HP sensitivities -- a method with a broader performance plateau around optimal HPs will appear better simply because it is more robust, not because its peak performance is higher. **Mitigation**: report both best performance (peak of the grid) AND expected performance (average over the HP grid), following Choi et al.'s recommendation. Also report HP sensitivity plots showing accuracy vs. each HP.
- **Statistical attack**: 5 seeds may be insufficient for detecting the small effect sizes expected (0.1-0.6% accuracy differences). Power analysis: for a paired t-test at alpha=0.05, power=0.8, to detect a 0.3% accuracy difference with expected std=0.15% across seeds, we need n = ceil((2 * 1.96 * 0.15 / 0.3)^2) ~ 4 seeds. So 5 seeds is marginally adequate. For 0.1% effects, we would need ~40 seeds, which is computationally prohibitive. **Mitigation**: be transparent about detection limits; frame results as "no effect >0.3% detected" rather than "no effect exists."
- **Benchmark attack**: CIFAR-10/100 may be saturated -- modern methods achieve >96% on CIFAR-10, leaving little room for WD improvements to manifest. ImageNet is better but expensive. The true test might be on LLM pretraining (where CWD shows its strongest results), but this is extremely expensive. **Mitigation**: include ImageNet and one LLM-scale experiment (Wikitext-103 with a 125M model) as sanity checks. Acknowledge that CIFAR results may not transfer to LLM scale.
- **Ablation completeness attack**: The effective-WD-strength ablation (comparing CWD with matched constant WD) is the most important ablation. If this single ablation is done carefully, it alone could explain CWD's results. The random mask ablation is a strong complement. These two ablations together would be definitive for the alignment-awareness question. **Verdict**: The ablation design is strong.
- **Verdict**: STRONG. This is the experiment the field actually needs. The main risk is that the null hypothesis is correct (no meaningful differences), which is scientifically valuable but may face publication resistance.

### Against Candidate B (Metric Validation)

- **Confound attack**: CSI and AIS may simply be proxies for total WD strength, in which case their correlation with accuracy is trivial (better-tuned WD -> better metrics AND better accuracy). **Mitigation**: partial out total WD applied when computing correlations. But this may remove the entire signal.
- **Statistical attack**: Spearman rank correlation across 7 methods is low-powered (max rho=1.0 but with only 7 data points, even rho=0.7 is not significant at p<0.05). **Mitigation**: include more methods (add cosine WD, linear WD, step WD, warm-restart WD) to increase the number of data points to 10+. Alternatively, compute correlation across all method-architecture-dataset combinations (7 methods x 2 architectures x 3 datasets = 42 data points).
- **Benchmark attack**: The metrics are proposed as universal but validated only on vision tasks. LLM pretraining may require entirely different metrics (e.g., perplexity-based rather than accuracy-based CSI). **Mitigation**: include at least one LLM experiment.
- **Ablation completeness attack**: The CSI component ablation is critical -- if only one component (e.g., weight norm variance) drives the correlation, the metric can be simplified. The AIS ablation across training stages is informative but may not change the overall conclusion.
- **Verdict**: MODERATE. The metric validation experiment is valuable but may produce weak results if the metrics are not truly informative. The risk is that BEM, CSI, AIS turn out to be fancy versions of "total WD applied" + "training stability" + "noise level", with no incremental predictive power over simpler proxies.

### Against Candidate C (CWD Falsification)

- **Confound attack**: The "effective lambda" calculation assumes CWD's mask ratio is approximately constant over training, which may not be true. The mask ratio evolves dynamically (CWD's paper shows it varies from ~40% to ~60% during training). The time-averaged effective lambda may miss important temporal effects. **Mitigation**: track mask ratio at each step and use the actual time-varying effective-lambda trajectory as the baseline (i.e., apply a WD schedule that matches CWD's effective-lambda curve but without the alignment-based mask selection).
- **Statistical attack**: This is a well-powered experiment because the comparison is paired (CWD vs. matched-constant on same seeds). With 5 seeds, a 0.2% effect at std=0.15% is detectable. **Verdict**: adequate.
- **Benchmark attack**: The falsification is strongest on CIFAR where CWD's improvements are smallest (~0.1-0.3%). On LLM pretraining where CWD shows larger improvements, the effective-lambda explanation may be more clearly falsified. But LLM experiments are expensive. **Mitigation**: if the CIFAR result is ambiguous, prioritize one LLM experiment.
- **Ablation completeness attack**: The random mask + inverted mask + continuous modulation ablation battery is comprehensive and would definitively establish whether alignment matters. The continuous modulation ablation (cosine-weighted WD) is particularly valuable because it tests whether CWD's binary discretization loses information compared to a continuous signal.
- **Verdict**: STRONG. This is a clean, well-powered falsification experiment that directly addresses the most important empirical question about the most recent SOTA method (CWD). The result -- whether positive or negative -- is publishable and informative.


## Phase 4: Refinement

### Dropped
- None. All three candidates address complementary aspects of the same fundamental question: "Do dynamic WD methods genuinely help, and can we measure why?"

### Strengthened and Merged

The three candidates are naturally **nested**: Candidate A provides the comprehensive benchmark infrastructure, Candidate C provides the most important single ablation within that benchmark, and Candidate B validates the proposed metrics using the same experimental data. The optimal research design merges all three into a single coherent experimental campaign:

**Merged Design: "A Rigorous Empirical Investigation of Dynamic Weight Decay Methods"**

The experimental campaign proceeds in three phases:

**Phase I (Pilot, ~2 GPU-hours):** CIFAR-10, ResNet-20, 3 methods (AdamW, CWD-AdamW, SWD), 3 seeds. Validate that the evaluation infrastructure works, confirm the effective-lambda confound for CWD, compute preliminary AIS values. Decision point: if AIS < 0.05 and effective-lambda-matched WD matches CWD, the alignment story is dead and we pivot to null-hypothesis framing.

**Phase II (Main benchmark, ~40 GPU-hours):** Full method sweep (7 methods) x 2 architectures (ResNet-20, VGG-16-BN) x 2 datasets (CIFAR-10, CIFAR-100) x 5 seeds. Equal-budget HP grids per method. All ablations from Candidate C (random mask, inverted mask, effective-lambda matching). Compute BEM, CSI, AIS for all runs. Statistical tests for all pairwise comparisons.

**Phase III (Scale validation, ~100 GPU-hours):** ImageNet with ResNet-50 (3 methods, 3 seeds) + optional ViT-Small. One LLM-scale experiment (125M model on Wikitext-103) to check whether CIFAR findings transfer. This phase confirms or refutes the scale-dependence question raised by Wen et al. (2025).

### Additional Controls Added During Refinement

1. **Learning rate interaction control**: For each WD method, sweep LR jointly with WD (2D grid) to account for the LR-WD coupling identified by Schaipp (2024) and Loshchilov & Hutter (2019). At minimum, test each WD method at 3 LR values.

2. **Training length control**: Run each experiment for 200 epochs (CIFAR) and 90 epochs (ImageNet). Also run a subset at 2x training length to test the "WD benefits diminish with longer training" hypothesis from Lewkowycz & Gur-Ari (2020).

3. **Parameter group control**: Ensure all methods exclude bias and normalization parameters from WD (following best practices). Test one configuration where all parameters receive WD to check if method rankings change.

4. **Implementation verification**: Before running experiments, verify that (a) PyTorch AdamW implementation matches the mathematical definition used in each paper, (b) CWD mask is computed correctly per the ICLR 2026 paper, (c) SWD's gradient-norm-aware scheduling uses the correct formula.

### Selected Front-Runner

**The CWD Falsification Experiment (Candidate C)** is the single highest-value experiment because:
1. CWD is the most recent SOTA (ICLR 2026) and the most widely cited alignment-aware WD method
2. The falsification is clean: one specific mechanism (alignment-awareness) tested against one specific alternative (reduced effective WD)
3. The result is informative regardless of outcome: if CWD is explained by effective-WD reduction, the field learns that alignment is uninformative; if CWD genuinely outperforms matched-WD, the field learns that alignment matters
4. The experiment requires minimal additional compute beyond the standard benchmark
5. This directly tests the most contested claim: whether gradient-weight alignment provides actionable signal for WD decisions

The comprehensive benchmark (Candidate A) and metric validation (Candidate B) provide the necessary context and infrastructure, but the CWD falsification is the single experiment that would most advance the field's understanding.


## Phase 5: Final Proposal

### Title
**"Measuring What Matters: A Rigorous Empirical Assessment of Dynamic Weight Decay Methods with Standardized Evaluation Metrics"**

### Hypothesis
Under compute-controlled, equal-tuning-effort conditions, the performance differences between dynamic WD methods (CWD, SWD, cosine WD schedule, AdamWN) and optimally tuned constant WD are smaller than reported in individual papers, and CWD's improvements can be substantially (>50%) attributed to reduced effective WD strength rather than alignment-awareness. The proposed standardized metrics (BEM, CSI, AIS) provide predictive power (Spearman rho > 0.5) for method rankings that raw accuracy alone does not capture.

### Falsification Criterion
The hypothesis is KILLED if:
1. Any dynamic WD method achieves >0.5% accuracy improvement over the best constant WD baseline with p < 0.01 (Bonferroni-corrected for multiple comparisons) across 3+ architectures, OR
2. CWD outperforms effective-lambda-matched constant WD by >0.3% with p < 0.05, AND random-mask WD with matched sparsity performs >0.3% worse than CWD (both conditions required to confirm alignment matters), OR
3. AIS values are consistently >0.2 across architectures and datasets, indicating that alignment signals are genuinely informative at practical scale.

### Method
The approach being tested is not a new algorithm but a rigorous evaluation methodology for comparing existing dynamic WD methods. The experimental framework includes:

1. **Unified evaluation codebase**: Fork the `why-weight-decay` repository (NeurIPS 2024, MIT license), which provides ResNet/VGG/ViT training infrastructure with weight/gradient norm tracking. Add implementations of CWD (one-line), SWD/AdamS, AdamWN (~20 lines), cosine WD schedule.

2. **Fair comparison protocol** (inspired by Choi et al. 2019):
   - Equal hyperparameter tuning budget per method (20 grid points)
   - Joint LR-WD sweep (5 LR x 4 WD values per method)
   - Report both peak performance and expected performance (average over grid)
   - Report HP sensitivity plots
   - All results with 5 random seeds, mean +/- std

3. **Standardized metrics computation**:
   - **BEM**: Normalize all comparisons to equal total training FLOPs. For methods with different overhead (e.g., CWD computes mask each step), account for the additional compute.
   - **CSI**: Composite metric: (a) coefficient of variation of weight norm trajectory over last 50% of training, (b) spectral condition number of weight matrices at convergence, (c) stability of effective learning rate (std/mean of eta_eff across layers). Normalize to [0,1]; lower = more stable.
   - **AIS**: Empirical alignment informativeness: fraction of training steps where the CWD mask makes a "correct" decision (defined as: applying WD to parameters whose norm exceeds the optimal norm, and not applying WD to parameters whose norm is below optimal). Estimated post-hoc from the best constant-WD run as the oracle.

4. **CWD falsification battery**:
   - CWD vs. effective-lambda-matched constant WD
   - CWD vs. random binary mask with matched sparsity
   - CWD vs. inverted (anti-alignment) mask
   - CWD vs. continuous cosine-similarity-weighted WD

### Evaluation Protocol

**Primary benchmarks (established public benchmarks):**
- CIFAR-10, CIFAR-100 (standard torchvision)
- ImageNet-1K (standard training protocol)

**Architectures:**
- CIFAR: ResNet-20, VGG-16-BN
- ImageNet: ResNet-50
- Optional extension: ViT-Small (CIFAR-100, ImageNet)

**Metrics with statistical test plan:**
- Primary: Test accuracy, mean +/- std across 5 seeds
- Pairwise comparison: Paired t-test (alpha=0.05) with Bonferroni correction for 7 methods
- Nonparametric backup: Wilcoxon signed-rank test
- Effect size: Cohen's d for each pairwise comparison
- Bootstrap 95% CI for all reported accuracy numbers
- Multiple comparison correction: Bonferroni for 21 pairwise comparisons (7 methods)

**Number of random seeds:** 5 (42, 123, 456, 789, 1024)

### Ablation Schedule

| Ablation | What it tests | Expected outcome |
|----------|--------------|-----------------|
| CWD vs. matched-lambda constant WD | Whether alignment-awareness matters beyond WD reduction | Expect matched performance (null: alignment is uninformative) |
| CWD vs. random mask (matched sparsity) | Whether the specific alignment-based mask selection matters | Expect matched performance (null: any sparsity pattern works) |
| CWD vs. inverted mask | Whether anti-alignment WD is harmful | Expect worse performance (confirming alignment has directional value, even if magnitude is small) |
| CWD vs. continuous cosine-weighted WD | Whether binary vs. continuous alignment signal matters | Uncertain: either continuous is better (supporting richer alignment use) or equivalent (supporting CWD's discretization) |
| Schedule shape ablation (cosine/linear/step/log WD) | Whether WD schedule shape matters beyond average value | Expect matched performance with same average (null: only average WD matters) |
| BN vs. no-BN | Whether WD method rankings change with normalization | Expect different rankings (confirming architecture-dependence) |
| 200 vs. 400 epoch training | Whether WD method benefits persist with longer training | Expect diminishing differences (WD benefits should decrease with training length) |
| Per-layer vs. global WD | Whether per-layer WD assignment adds value | Uncertain: per-layer (AlphaDecay-style) may help for ViTs but not ResNets |

### Control Experiments

1. **Null control**: Standard AdamW with constant WD, grid-searched over {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}. This is the baseline everything must beat.

2. **Oracle control**: Post-hoc optimal WD schedule (choose the best WD value at each epoch using a validation set). This establishes the ceiling: if the oracle barely outperforms constant WD, dynamic scheduling has no room to help.

3. **Random schedule control**: Apply WD according to a random walk schedule (lambda(t) = lambda_0 * exp(noise(t))) with same mean as best constant WD. If this matches constant WD, schedule shape is irrelevant.

4. **Effective-LR-matched control**: For methods where WD acts primarily through effective LR (D'Angelo et al. 2024), directly adjust the LR schedule to achieve the same effective LR trajectory as the dynamic WD method. If this matches the dynamic WD performance, the method is just indirect LR scheduling.

### Pilot Design
- CIFAR-10, ResNet-20, 200 epochs
- Methods: AdamW (constant WD), CWD-AdamW, SWD/AdamS
- Seeds: 42, 123, 456
- Track: test accuracy, training loss, weight norm trajectory (per-layer), gradient-weight cosine (per-layer), effective LR, CWD mask ratio over time
- Decision criteria:
  - If CWD-AdamW vs. AdamW difference < 0.1%: alignment is likely uninformative at CIFAR scale
  - If CWD mask ratio is approximately constant: effective-lambda explanation is viable
  - If AIS < 0.05: alignment signal is noise
- Time estimate: ~15 min on 1 GPU, ~2 min on 8 GPUs

### Resource Estimate
| Phase | Configuration | GPU-hours | Wall time (8x RTX PRO 6000) |
|-------|--------------|-----------|---------------------------|
| Pilot | 3 methods x 3 seeds x ResNet-20/CIFAR-10 | 1.5 | 15 min |
| Phase II | 7 methods x 5 seeds x 2 arch x 2 datasets + ablations | 40 | ~5 hours |
| Phase III (ImageNet) | 3 methods x 3 seeds x ResNet-50 | 72 | ~9 hours |
| Phase III (optional ViT) | 3 methods x 3 seeds x ViT-Small/ImageNet | 48 | ~6 hours |
| Phase III (optional LLM) | 3 methods x 3 seeds x 125M/Wikitext-103 | 24 | ~3 hours |
| **Total** | | **~185** | **~23 hours** |

Target: all core experiments (Pilot + Phase II + ImageNet) completable in ~14 hours on 8 GPUs.

### Risk Assessment

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Null hypothesis is correct (no method beats constant WD) | Medium | High (60%) | Frame as a positive result: "the field's null hypothesis is confirmed, saving future researchers from pursuing dead ends" |
| CWD falsification is ambiguous (0.1-0.2% difference, p=0.05-0.10) | High | Medium (40%) | Increase seeds to 10 for the CWD-specific comparison; use Bayesian analysis with a prior centered on zero |
| Proposed metrics (CSI, AIS) are uninformative | Medium | Medium (50%) | Report negative results transparently; propose alternative metrics or conclude that simpler proxies (weight norm, training loss) are sufficient |
| CIFAR results don't transfer to ImageNet/LLM | Medium | Medium (40%) | Always include ImageNet results; acknowledge CIFAR limitations explicitly |
| Implementation bugs create false differences | High | Low (10%) | Unit test all optimizer implementations; verify against reference repos; compare weight norm trajectories against published figures |
| HP tuning confound (despite equal budget, one method benefits more from tuning) | Medium | Medium (30%) | Report sensitivity plots; compute "HP robustness" = std(accuracy) across HP grid; use Choi et al.'s inclusion-relationship test |

### Novelty Claim

The experimental contribution answers the following empirical question for the first time:

**"Under fair, compute-controlled conditions with proper statistical testing, do any of the recently proposed dynamic weight decay methods (CWD, SWD, cosine WD, AdamWN) provide statistically significant improvement over optimally tuned constant weight decay?"**

No existing paper answers this question because:
1. CWD (ICLR 2026) compares against its own baselines but does not include SWD, AdamWN, or compute-normalized constant WD
2. SWD (NeurIPS 2023) predates CWD and AdamWN
3. D'Angelo et al. (NeurIPS 2024) study WD mechanisms but do not compare dynamic WD methods
4. No paper reports statistical significance tests (confidence intervals, p-values) for WD method comparisons
5. No paper tests the effective-lambda-reduction hypothesis for CWD

Additionally, the standardized metrics (BEM, CSI, AIS) are proposed and empirically validated for the first time, providing the community with a principled evaluation toolkit for future WD research.

The CWD falsification battery (matched-lambda, random mask, inverted mask, continuous modulation) has never been performed and directly addresses the most important mechanistic question about the most recent SOTA method.
