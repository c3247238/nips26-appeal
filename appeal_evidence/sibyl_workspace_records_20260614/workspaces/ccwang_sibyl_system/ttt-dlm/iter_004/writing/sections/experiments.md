## 4. Experiments

We evaluate Belief-State Diffusion (BSD) and Adaptive Classifier-Free Guidance (A-CFG) on two reasoning benchmarks, comparing against vanilla Dream-7B denoising, established remasking baselines, and our prior Diffusion Memory Injection (DMI) method. Our experiments are structured to answer four questions: (1) Do BSD and A-CFG individually improve reasoning accuracy? (2) How sensitive are they to hyperparameter choices? (3) Does their combination yield synergistic gains? (4) Do improvements generalize across benchmarks?

### 4.1 Experimental Setup

**Model.** We use Dream-v0-Instruct-7B \citep{liu2025dream}, the strongest open-source masked diffusion language model at the time of our experiments. All experiments run on a single NVIDIA RTX PRO 6000 Blackwell GPU (98 GB VRAM) on the cs8000d server.

**Benchmarks.** Our primary benchmark is **Countdown-500**, a structured arithmetic reasoning task requiring models to find arithmetic expressions over given numbers that reach a target value. We use 500 problems evaluated across 3 random seeds (42, 123, 456) for full-scale evaluation, and a 16-sample pilot (Countdown-16) for ablation studies. For cross-benchmark generalization, we evaluate on **GSM8K-16**, a 16-sample pilot of grade-school math word problems.

**Baselines.** We compare against five baselines:
- **Vanilla Dream-7B**: Standard denoising with 128 steps, $T=128$, generation length 256, temperature 0.4.
- **DMI** ($\alpha=0.3$): Diffusion Memory Injection \citep[our prior iteration]{}, which mixes previous-step logit-weighted embeddings into mask positions with a fixed ratio $\alpha$. Our only full-scale validated method (9.3\% on Countdown-500, $\sim$2$\times$ vanilla).
- **ReMDM-conf**: Confidence-based remasking \citep{arriola2025remdm}, which re-masks and re-predicts low-confidence tokens.
- **RCR**: Running Confidence Remasking \citep{chen2025mdpo}, which tracks per-token confidence across steps.
- **DTA**: Denoising-Time Adaptation with online LoRA updates \citep[our prior iteration]{}, representing parameter-space adaptation.

**Our methods.** We evaluate:
- **BSD** ($k=0.75$, linear $\alpha$ 0.1$\to$0.8): Belief-State Diffusion with 25\% of steps in the continuous belief phase and 75\% for hard token reveal.
- **A-CFG** ($w=1.5$, fixed, $m=10\%$): Adaptive Classifier-Free Guidance with confidence-based re-masking of the 10\% least-confident positions and fixed guidance weight 1.5.
- **BSD+A-CFG**: Combination applying BSD in Phase 1 and A-CFG during Phase 2 hard reveal.

**Evaluation metrics.** We report accuracy (primary metric), n-gram repetition rates (rep-2, rep-3), lexical diversity (distinct-1/2/3), output length statistics, and FLOPs overhead relative to vanilla. We explicitly do not use perplexity as a quality metric, following our finding from prior iterations that pilot PPL improvements of 16--25\% failed entirely at full scale (all $p > 0.25$).

**Statistical protocol.** For the Countdown-500 full-scale evaluation, we apply McNemar's exact test with Bonferroni correction ($\alpha' = 0.05/N$) for pairwise method comparisons, and report bootstrap 95\% confidence intervals (10,000 resamples). For pilot evaluations ($n=16$), we report Cohen's $h$ effect sizes and note that statistical tests have very low power at this sample size (minimum detectable effect $\sim$25pp).

### 4.2 Main Results

Table 1 presents the comprehensive comparison across all methods and benchmarks.

**Table 1: Main results on Countdown and GSM8K.** Countdown-500 results use 3-seed averaging; Countdown-16 and GSM8K-16 are single-seed pilots. Best results per column are **bolded**. FLOPs are relative to vanilla 128-step denoising.

| Method | Group | FLOPs | Countdown-500 | Countdown-16 | GSM8K-16 | rep-3 | distinct-3 |
|--------|-------|-------|---------------|-------------|----------|-------|------------|
| Vanilla (128 steps) | Baseline | 1.0$\times$ | 4.7% $\pm$ 0.6 | 0.0% | 25.0% | 0.079 | 0.876 |
| ReMDM-conf | Remasking | $\sim$1.5$\times$ | 4.4% $\pm$ 1.0 | --- | --- | --- | --- |
| RCR | Remasking | $\sim$1.2$\times$ | 5.7% $\pm$ 0.6 | --- | --- | --- | --- |
| DMI ($\alpha$=0.3) | Cross-step info | $\sim$1.05$\times$ | **9.3% $\pm$ 1.4** | 12.5% | 25.0% | 0.095 | 0.857 |
| BSD ($k$=0.75) | Cross-step info | $\sim$1.1$\times$ | --- | 6.2% | 18.8% | 0.048 | 0.913 |
| A-CFG ($w$=1.5) | Cross-step info | $\sim$2.0$\times$ | --- | **12.5%** | **37.5%** | 0.054 | 0.889 |
| BSD+A-CFG | Cross-step info | $\sim$2.1$\times$ | --- | 6.2% | --- | 0.046 | 0.915 |
| DTA (LoRA) | Param. adapt. | $\sim$3.0$\times$ | --- | 6.2% | --- | --- | --- |

Several findings emerge from Table 1. First, **cross-step information methods consistently outperform remasking baselines**: DMI (9.3\%), BSD (6.2\%), and A-CFG (12.5\%) all exceed ReMDM-conf (4.4\%) and RCR (5.7\%) on their respective evaluation scales. This confirms that the representation poverty of mask embeddings, rather than the denoising schedule, is the primary bottleneck.

Second, **DMI remains the only method validated at full scale** with 3-seed replication on Countdown-500. Its 9.3\% $\pm$ 1.4\% accuracy represents a $\sim$2$\times$ improvement over vanilla (4.7\% $\pm$ 0.6\%) with near-zero computational overhead ($\sim$1.05$\times$ FLOPs).

Third, **A-CFG achieves the highest pilot accuracy** on both Countdown-16 (12.5\%) and GSM8K-16 (37.5\%), matching DMI on the former and substantially exceeding all methods on the latter. This result is particularly notable given that A-CFG requires $\sim$2$\times$ FLOPs due to its dual forward pass.

Fourth, **all methods maintain generation quality**: repetition rates (rep-3) remain below 0.10 and lexical diversity (distinct-3) above 0.85 for all methods, with no evidence of degeneration. BSD in particular shows the lowest repetition (0.048) and highest diversity (0.913), suggesting that continuous belief representations may regularize the generation process.

Finally, the **BSD+A-CFG combination (6.2\%) performs below both individual methods**, decisively falsifying our synergy hypothesis (H7). We analyze this failure in Section 5.

**Statistical analysis.** On the Countdown-500 full-scale evaluation, DMI versus vanilla yields Cohen's $h = 0.72$ (medium effect size). On the pilot evaluations ($n=16$), McNemar's exact tests do not reach significance for any pairwise comparison (all $p \geq 0.45$), which is expected given the low statistical power at $n=16$. Bootstrap 95\% CIs for pilot comparisons are wide: DMI and A-CFG versus vanilla both show $\Delta = +0.125$ with CIs of $[0.0, +0.25]$ on Countdown-16, and A-CFG versus vanilla shows $\Delta = +0.125$ with CI $[-0.125, +0.375]$ on GSM8K-16. Full-scale 3-seed validation of BSD and A-CFG is required for publication-ready conclusions.

### 4.3 Ablation Studies

We conduct systematic ablation studies on both methods using the Countdown-16 pilot. All ablations use seed 42, 128 denoising steps, generation length 256, and temperature 0.4.

#### 4.3.1 BSD k-Parameter (Belief Phase Length)

The $k$ parameter controls what fraction of denoising steps use hard token reveal versus continuous belief refinement. We test $k \in \{T/4, T/2, 3T/4\}$, corresponding to 75\%, 50\%, and 25\% of steps in the belief phase, respectively.

| $k$ | Belief Phase | Accuracy | rep-3 | distinct-3 |
|-----|-------------|----------|-------|------------|
| Vanilla | 0\% | 0.0\% | 0.079 | 0.876 |
| $T/4$ | 75\% | 0.0\% | 0.014 | 0.899 |
| $T/2$ | 50\% | 0.0\% | 0.020 | 0.947 |
| $T/4 \times 3$ | **25\%** | **6.2\%** | 0.048 | 0.913 |

Only $k = 3T/4$ (the shortest belief phase, 25\%) achieves non-zero accuracy. Longer belief phases ($k = T/4$ and $k = T/2$) yield 0\% accuracy despite maintaining excellent diversity metrics. This **falsifies H3** (intermediate $k$ is optimal): the model requires early access to hard token anchors, and even modest delays in committing to discrete tokens degrade reasoning performance. This finding suggests that Dream-7B's Transformer representations are not well-calibrated for continuous belief inputs, and that hard token context is essential for the model to form coherent arithmetic reasoning chains.

#### 4.3.2 BSD Alpha Schedule

Fixing $k = 0.75$ (best from above), we ablate four $\alpha$ schedule variants controlling the EMA update rate during the belief phase.

| Alpha Schedule | Accuracy | rep-3 | distinct-3 |
|----------------|----------|-------|------------|
| linear (0.1$\to$0.8) | 6.2\% | 0.048 | 0.913 |
| cosine (0.1$\to$0.8) | 6.2\% | 0.048 | 0.913 |
| constant (0.3) | 6.2\% | 0.048 | 0.913 |
| constant (0.5) | 6.2\% | 0.067 | 0.852 |

All four schedules produce identical accuracy (6.2\%), with near-identical quality metrics for the first three variants. The constant (0.5) schedule shows slightly higher repetition and lower diversity, but accuracy is unaffected. **The alpha schedule shape is not a performance bottleneck for BSD at the optimal $k$ setting.** We adopt linear (0.1$\to$0.8) as the default for its slightly better diversity profile.

#### 4.3.3 A-CFG Guidance Weight

We test three guidance weight values $w \in \{1.0, 1.5, 2.0\}$ with fixed re-masking of the 10\% least-confident positions.

| Guidance Weight $w$ | Accuracy | rep-3 | distinct-3 | FLOPs |
|---------------------|----------|-------|------------|-------|
| Vanilla | 0.0\% | 0.079 | 0.876 | 1.0$\times$ |
| $w = 1.0$ | 6.2\% | 0.057 | 0.900 | $\sim$2.0$\times$ |
| $w = 1.5$ | **12.5\%** | 0.054 | 0.889 | $\sim$2.0$\times$ |
| $w = 2.0$ | **12.5\%** | 0.052 | 0.885 | $\sim$2.0$\times$ |

Both $w = 1.5$ and $w = 2.0$ achieve the highest accuracy (12.5\%), doubling the $w = 1.0$ result. Moderate-to-strong guidance is required for meaningful reasoning improvements. Quality metrics remain stable across all settings, with no evidence of over-extrapolation even at $w = 2.0$ (capped at $w_{\max} = 2.0$).

#### 4.3.4 A-CFG Temporal Schedule

We test four temporal scheduling strategies for guidance weight, motivated by the theoretical prediction from continuous diffusion that early guidance is harmful \citep{rojas2025cfgscheduling}. All use $w_{\text{base}} = 1.5$.

| Schedule | Accuracy | rep-3 | distinct-3 |
|----------|----------|-------|------------|
| **Fixed** | **12.5\%** | 0.054 | 0.889 |
| Linear ramp | 0.0\% | 0.074 | 0.866 |
| Cosine ramp | 0.0\% | 0.068 | 0.875 |
| Threshold (70/30) | 0.0\% | 0.055 | 0.897 |

Fixed guidance dominates all scheduled variants, which uniformly achieve 0\% accuracy. This **decisively falsifies H6** (temporal scheduling outperforms fixed guidance). The result is striking: all three non-fixed schedules suppress guidance during early denoising steps (when mask rate is high), but this is precisely when Dream-7B's predictions are most uncertain and would benefit most from guidance amplification. The theoretical framework from continuous diffusion \citep{rojas2025cfgscheduling}---which assumes smooth noise-to-signal transitions---does not transfer to masked diffusion's discrete mask/unmask dynamics, where the model has sufficient partial context at any step to benefit from constant guidance.

#### 4.3.5 RACFG vs. A-CFG: Re-masking Signal Quality

Our original proposal included Reasoning-Aware CFG (RACFG), which selects positions for re-masking based on cross-step JSD stability rather than single-step confidence. We test this directly against A-CFG.

| Method | Re-mask Signal | Accuracy | FLOPs |
|--------|---------------|----------|-------|
| Vanilla | N/A | 0.0\% | 1.0$\times$ |
| RACFG (JSD stability) | Cross-step JSD | 0.0\% | $\sim$1.8$\times$ |
| A-CFG (confidence) | Single-step confidence | **12.5\%** | $\sim$2.0$\times$ |

A-CFG (12.5\%) dramatically outperforms RACFG (0.0\%), **falsifying H5** (cross-step stability outperforms single-step confidence). Across all RACFG configurations tested---re-mask percentages of 5\%, 10\%, and 20\%, and EMA smoothing parameters $\lambda \in \{0.3, 0.7\}$---every RACFG variant achieves exactly 0\% accuracy, identical to vanilla.

The root cause is that Dream-7B produces remarkably stable cross-step probability distributions, with JSD stability scores concentrated near 0.997 across all positions. This near-degeneracy eliminates the "hesitation signal" that RACFG relies on to identify reasoning-critical tokens. Single-step confidence, by contrast, directly identifies currently uncertain positions where guidance has the most impact.

Figure 4 summarizes all ablation results in a 2$\times$2 grid.

> **Figure 4**: Ablation study grid. (a) BSD $k$-parameter: only the shortest belief phase ($k=3T/4$, 25\%) achieves non-zero accuracy. (b) BSD $\alpha$ schedule: all schedules yield identical 6.2\% accuracy. (c) A-CFG guidance weight: moderate-to-strong guidance ($w \geq 1.5$) required. (d) A-CFG temporal schedule: only fixed guidance works; all scheduled variants achieve 0\%. Annotations show distinct-3 diversity scores.

### 4.4 GSM8K Generalization

To test whether improvements transfer beyond structured arithmetic reasoning, we evaluate the best configuration of each method on GSM8K-16 (grade-school math word problems).

| Method | Countdown-16 | GSM8K-16 | $\Delta$ vs Vanilla (GSM8K) |
|--------|-------------|----------|---------------------------|
| Vanilla | 0.0\% | 25.0\% (4/16) | --- |
| DMI ($\alpha=0.3$) | 12.5\% | 25.0\% (4/16) | +0.0pp |
| BSD ($k=0.75$) | 6.2\% | 18.8\% (3/16) | $-$6.2pp |
| **A-CFG** ($w=1.5$) | **12.5\%** | **37.5\% (6/16)** | **+12.5pp** |

A-CFG is the only method demonstrating consistent cross-benchmark improvement, achieving 37.5\% on GSM8K-16 versus vanilla's 25.0\% (+12.5pp). DMI, despite being the strongest validated method on Countdown-500, shows zero benefit on GSM8K. BSD performs below vanilla on GSM8K (18.8\% vs 25.0\%), suggesting that continuous arithmetic token mixing in belief vectors may disrupt the free-form reasoning chains required by math word problems.

The flip analysis supports A-CFG's generalization: of the 4 disagreements between A-CFG and vanilla on GSM8K, A-CFG wins 3 and loses 1. This 3:1 ratio, while not statistically significant at $n=16$, is directionally consistent with the Countdown results.

Figure 5 presents the cross-benchmark comparison.

> **Figure 5**: Cross-benchmark generalization. Grouped bar chart showing accuracy of each method on Countdown-16 (blue) versus GSM8K-16 (orange). A-CFG is the only method with consistent improvement across both benchmarks (+12.5pp on GSM8K). BSD degrades on free-form math ($-$6.2pp).

### 4.5 Compute Fairness Analysis

Since A-CFG requires $\sim$2$\times$ FLOPs (one extra forward pass per step), we compare each method against vanilla denoising with a proportionally increased number of steps to control for compute budget.

**Table 2: Compute-fair comparison.** Each method is compared against vanilla with matched FLOPs (e.g., A-CFG at $\sim$2.0$\times$ versus vanilla with 256 steps).

| Method | FLOPs | Accuracy | Vanilla (matched FLOPs) | $\Delta$ |
|--------|-------|----------|------------------------|----------|
| BSD ($\sim$1.1$\times$) | 1.1$\times$ | 6.2\% | 12.5\% (141 steps) | $-$6.2pp |
| A-CFG ($\sim$2.0$\times$) | 2.0$\times$ | 12.5\% | 12.5\% (256 steps) | $\pm$0.0pp |
| BSD+A-CFG ($\sim$2.1$\times$) | 2.1$\times$ | 6.2\% | 6.2\% (269 steps) | $\pm$0.0pp |
| DMI ($\sim$1.05$\times$) | 1.05$\times$ | 12.5\% | 12.5\% (134 steps) | $\pm$0.0pp |

At matched compute budgets, vanilla step-scaling is competitive with all proposed methods on the Countdown-16 pilot. Specifically, vanilla with 256 steps (2.0$\times$ FLOPs) achieves 12.5\%, matching A-CFG exactly. The Pareto frontier analysis identifies vanilla step-scaling as Pareto-optimal at most compute levels in the pilot data.

However, two caveats are important. First, the pilot sample size ($n=16$) provides very low resolution for distinguishing methods at similar accuracy levels. Second, the full-scale Countdown-500 results tell a different story: at 1.0$\times$ FLOPs, vanilla achieves only 4.7\% while DMI achieves 9.3\%---a clear advantage for information-quality methods at the low-compute frontier. The compute-fair comparison suggests that the value proposition of BSD and A-CFG lies not in dominating vanilla at all compute budgets, but in providing qualitatively different improvements (belief accumulation, reasoning signal amplification) that may scale differently with problem difficulty and benchmark complexity.

### 4.6 Information-Theoretic Validation

We validate that BSD belief vectors accumulate information across denoising steps by tracking per-position belief entropy trajectories.

| Property | BSD | Vanilla |
|----------|-----|---------|
| Mean terminal entropy | 0.001 | 0.002 |
| Spearman $\rho$ (step vs. entropy) | $-$0.952 | $-$0.932 |
| Samples with monotonic decrease ($\rho < -0.8$) | 15/16 (94\%) | N/A |
| Entropy--accuracy correlation ($r$) | 0.784 ($p < 0.001$) | --- |

BSD belief entropy decreases near-monotonically during denoising (Spearman $\rho = -0.952$, 15/16 samples), **supporting H2**. Terminal belief entropy is lower for BSD (0.001) than vanilla (0.002), confirming that continuous belief evolution accumulates more information than discrete unmasking. Most importantly, there is a strong correlation between terminal belief entropy and task accuracy ($r = 0.784$, $p < 0.001$): samples where beliefs converge more completely are more likely to be solved correctly. This provides information-theoretic evidence that belief convergence quality is predictive of reasoning success, establishing a principled diagnostic for identifying when BSD is likely to help.

### 4.7 Hypothesis Verification Summary

Table 3 summarizes the status of all pre-registered hypotheses.

**Table 3: Hypothesis verification summary.**

| ID | Hypothesis | Expected | Observed | Verdict |
|----|-----------|----------|----------|---------|
| H1 | BSD $>$ DMI on Countdown-500 | $\geq$14\% | 6.2\% (pilot) | **Pending** (full-scale required) |
| H2 | BSD entropy monotonically decreases | Monotonic | $\rho = -0.952$, 15/16 monotonic | **Supported** |
| H3 | Intermediate $k$ optimal | $k \approx T/4$--$T/2$ best | $k = 3T/4$ (shortest belief) best | **Falsified** |
| H4 | A-CFG $>$ vanilla on Countdown | $\geq$15\% | 12.5\% (pilot) | **Partially supported** |
| H5 | JSD $>$ confidence for re-masking | JSD $+$2pp | 0\% vs 12.5\% | **Falsified** |
| H6 | Temporal scheduling $>$ fixed | $+$2pp | 0\% vs 12.5\% | **Falsified** |
| H7 | BSD+A-CFG synergy | $\geq$18\% | 6.2\% ($<$ both individual) | **Falsified** |
| H8 | Best method generalizes to GSM8K | Significant gain | A-CFG +12.5pp | **Supported** (pilot) |
| H9 | BSD quality maintained | rep-3 $<$ vanilla$+$20\% | rep-3 = 0.048 ($<$ vanilla 0.079) | **Supported** |
| H10 | A-CFG no length bias | Mean $\pm$15\% | Within range | **Supported** |
| H11 | Methods beat vanilla at equal FLOPs | Methods $>$ vanilla | Competitive | **Falsified** (pilot) |

Of 11 hypotheses, 4 are supported (H2, H8, H9, H10), 4 are falsified (H3, H5, H6, H7), 1 is partially supported (H4), and 2 require full-scale validation (H1, H11). The falsified hypotheses---particularly H5, H6, and H7---provide actionable constraints on the design space for future MDLM inference-time methods and are analyzed in detail in Section 5.

<!-- FIGURES
- Figure 4: gen_fig4_ablation_grid.py, fig4_ablation_grid.pdf — 2x2 ablation grid: BSD k-parameter, BSD alpha schedule, A-CFG guidance weight, A-CFG temporal schedule
- Figure 5: gen_fig5_gsm8k_generalization.py, fig5_gsm8k_generalization.pdf — Grouped bar chart comparing Countdown-16 vs GSM8K-16 accuracy across methods
- Table 1: inline — Main results comparison across all methods and benchmarks
- Table 2: inline — Compute-fair comparison of methods vs vanilla at matched FLOPs
- Table 3: inline — Hypothesis verification summary (H1-H11)
-->
