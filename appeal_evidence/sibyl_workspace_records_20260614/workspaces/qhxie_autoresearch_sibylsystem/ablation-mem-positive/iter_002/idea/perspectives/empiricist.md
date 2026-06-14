# Empiricist Perspective

## Phase 1: Literature Survey

### Key Methodology Papers

1. [Chanin et al., 2024. A is for Absorption: Studying Feature Splitting and Absorption in SAEs. arXiv:2409.14507] — First systematic absorption quantification using ablation-based metric; identifies hierarchical feature co-occurrence as root cause. Limitation: metric unreliable past layer 17; only validates on spelling task.

2. [Karvonen et al., 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532] — 8-metric evaluation suite including probe projection absorption metric that works across all layers. Critical insight: projection-based metric avoids ablation reliability issues.

3. [Costa et al., 2025. From Flat to Hierarchical: MP-SAE. arXiv:2506.03093] — Matching Pursuit SAE reduces absorption via conditional orthogonality. Key insight: residual-guided greedy selection recovers hierarchical structure standard SAEs miss.

4. [Cui et al., ICLR 2026. On the Limits of Sparse Autoencoders. arXiv:2506.15963] — Proves full disentanglement is mathematically impossible under realistic sparsity. Critical context: high absorption may be an inevitable consequence of representational interference.

5. [Gao et al., 2024. Scaling and Evaluating SAEs. arXiv:2406.04093] — k-sparse autoencoders; establishes scaling laws. Provides baseline SAE training methodology for comparison.

6. [Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders. arXiv:2509.22033] — 65% absorption reduction via orthogonality penalty. Demonstrates absorption is reducible but not eliminable.

7. [Basu et al., 2026. Interpretability without Actionability. arXiv:2603.18353] — 98.2% probe AUROC but zero output change via steering. Critical warning: internal detection does not guarantee intervention efficacy.

8. [Bussmann et al., 2025. Matryoshka SAEs. arXiv:2503.17547] — Nested dictionaries reduce absorption; superior sparse probing. Shows architectural modification can address absorption.

### Experimental Landscape

**What has been properly tested:**
- Absorption existence and detection on Gemma-2-2B (Chanin et al.) — early layers only
- Cross-architecture comparison of absorption rates (SAEBench) — aggregate metrics
- Theoretical limits of SAE feature recovery (Cui et al.)

**What is accepted without evidence:**
- That absorption rates generalize across layer depths (prior work limited to layers 0-17)
- That first-letter spelling task results generalize to other hierarchical features
- That ablation-based absorption detection is reliable (SAEBench shows projection is more robust)

**Methodological gaps:**
- No systematic cross-layer quantification with projection-based metrics
- No rigorous statistical comparison of absorption detection methods (ablation vs projection)
- No controlled experiment isolating absorption from feature frequency confounds
- No quantification of absorption impact on established interpretability benchmarks

---

## Phase 2: Initial Candidates

### Candidate A: Projection-Based Cross-Layer Absorption Quantification

- **Core hypothesis**: Absorption rates measured via probe projection (SAEBench method) will show a non-monotonic pattern across layers, with mid-layer peaks (layers 5-9) reflecting optimal trade-off between feature specificity and hierarchical encoding density.
- **Falsification criterion**: If absorption rates are monotonic with layer depth (either increasing or decreasing throughout), the hypothesis is disproven. If rates are uniform across layers (no significant variation), the hypothesis is disproven.
- **Evaluation protocol**:
  - Primary benchmark: SAEBench probe projection metric on GPT-2 residual stream SAEs (layers 0, 3, 6, 9, 11)
  - Metric: absorption rate (% of features with significant absorption), mean absorption score
  - Statistical test: Kruskal-Wallis H-test across layers; post-hoc pairwise Wilcoxon tests with Bonferroni correction
  - Minimum 3 random seeds for candidate pair selection
- **Ablation plan**:
  - Ablate layer as unit → compare projection vs ablation metric reliability
  - Vary candidate selection threshold (0.3, 0.5, 0.7) → assess threshold sensitivity
  - Compare JumpReLU (GemmaScope) vs TopK (LlamaScope) architectures → assess architecture dependence
- **Confounders identified**:
  - Feature frequency: High-frequency features may show spuriously high absorption scores due to better probe quality
  - Dead feature ratio: Layers with high dead ratios may bias toward remaining active features
  - Dictionary size: Different SAE widths (16k vs 65k) may show different absorption scales
- **Pilot design**: Run projection-based absorption on GPT-2 layer 6 SAE (24,576 features, 99.9% alive) comparing top-100 candidate pairs by decoder cosine similarity. Expected time: 10 min. Early signal: whether projection scores correlate with decoder geometry.

### Candidate B: Absorption Detection Method Comparison (Ablation vs Projection)

- **Core hypothesis**: Probe projection (SAEBench) and ablation (Chanin et al.) absorption metrics measure different phenomena — projection captures representational contribution while ablation captures causal output sensitivity — and the discrepancy is informative about absorption mechanism.
- **Falsification criterion**: If both metrics agree on absorption status for >95% of candidate pairs, the hypothesis is disproven (they measure the same thing). If the correlation is near-zero (r < 0.1), the hypothesis is disproven (metrics are both noise).
- **Evaluation protocol**:
  - Primary benchmark: Run both metrics on same candidate pairs (n=100) across 3 layers
  - Metric: Correlation (Pearson, Spearman), classification agreement at thresholds (0.3, 0.5, 0.7)
  - Statistical test: McNemar's test for classification disagreement patterns
- **Ablation plan**:
  - Vary probe quality (by number of examples) → assess probe reliability effect
  - Vary layer depth → assess metric reliability degradation with layer
  - Compare high vs low frequency candidates → assess frequency confound
- **Confounders identified**:
  - Probe quality differences between parent and child features
  - Non-linear effects in output space that ablation captures but projection misses
  - Different sensitivity to absorbing vs non-absorbing child latents
- **Pilot design**: Run both metrics on 50 candidate pairs from GPT-2 layer 6, computing correlation. Expected time: 15 min. Early signal: whether metrics agree at all.

### Candidate C: Absorption Impact on Sparse Probing Fidelity

- **Core hypothesis**: Absorbed features will show systematically lower sparse probing accuracy than non-absorbed features of comparable frequency, because probing must rely on absorbing latents rather than the true parent feature.
- **Falsification criterion**: If absorbed and non-absorbed features show equal sparse probing accuracy (within 2% CI), the hypothesis is disproven. If accuracy difference is explained entirely by feature frequency, the hypothesis is disproven.
- **Evaluation protocol**:
  - Primary benchmark: K-sparse probing (Gurnee et al., 2023) on absorbed vs matched control features
  - Metric: Probe AUROC, probe calibration error, per-class precision-recall
  - Statistical test: Paired t-test (matched pairs) or ANCOVA controlling for frequency
  - Sample: 50 absorbed features (identified by projection score > 0.5) + 50 frequency-matched controls
- **Ablation plan**:
  - Ablate absorbed feature → measure probe accuracy change
  - Ablate absorbing child feature → measure probe accuracy change
  - Compare intervention response profiles between absorbed and control features
- **Confounders identified**:
  - Feature frequency (primary confound — absorbed features may be lower frequency)
  - Feature activation sparsity (denser features may have better probes regardless of absorption)
  - Layer-specific effects (mid-layers may show absorption impact differently than early/late)
- **Pilot design**: Train linear probes for 20 absorbed vs 20 control features on GPT-2 layer 6. Expected time: 12 min. Early signal: whether absorbed features show measurably different probe quality.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Confound attack**: The E4 results from prior experiments showed co-occurrence correlates negatively with absorption score (r=-0.52). If co-occurrence is a proxy for hierarchical feature relationship, this suggests the current scoring formula is confounded. The projection-based metric avoids this but still depends on probe quality, which varies with feature frequency. *Need to control for feature frequency in all cross-layer comparisons.*
- **Statistical attack**: With 5 layers and 100 candidates each, we have limited power for non-parametric tests. The Bonferroni correction for 10 pairwise comparisons (5 layers choose 2) reduces power substantially. *Consider FDR correction (Benjamini-Hochberg) instead of Bonferroni, or increase candidates per layer.*
- **Benchmark attack**: SAEBench's projection metric was validated on Gemma-2-2B SAEs. Its reliability on GPT-2 SAEs is assumed but not proven. Different model architectures (residual stream vs MLP-only) may affect metric validity. *Need to validate projection metric on GPT-2 with a ground-truth probe task.*
- **Ablation completeness attack**: Even with projection, we're only measuring one aspect of absorption (representational). If absorption also manifests as causal insensitivity (Basu et al.), we need output-based metrics too. *Consider adding logit lens or direct logit change measurement.*
- **Verdict**: STRONG — Candidate A directly addresses the cross-layer gap (Gap 1) and uses the more robust projection metric. The confound concerns are addressable with proper frequency matching.

### Against Candidate B

- **Confound attack**: The two metrics (projection and ablation) have different failure modes. Projection depends on probe quality; ablation depends on output sensitivity. If probe quality and output sensitivity are both correlated with absorption, the metrics will agree spuriously. *Need to independently validate probe quality and output sensitivity.*
- **Statistical attack**: McNemar's test requires binary classification decisions. With continuous scores, the threshold choice (0.3, 0.5, 0.7) is arbitrary and affects agreement statistics. *Consider using continuous correlation measures (ICC) as primary outcome.*
- **Benchmark attack**: The comparison is methodologically valid but may not produce a publishable finding — "two metrics disagree" is descriptive, not causal. The field already knows both metrics have limitations. *Need to connect disagreement to a downstream consequence (e.g., circuit analysis implications).*
- **Ablation completeness attack**: If the metrics measure different aspects of absorption (representational vs causal), disagreement is expected and informative. But without a ground truth, we cannot determine which is "correct." *Need to use synthetic data (SynthSAEBench) as ground truth reference.*
- **Verdict**: MODERATE — The comparison is methodologically valuable but may not stand alone as a paper. Should be combined with downstream impact analysis.

### Against Candidate C

- **Confound attack**: Feature frequency is the primary confound for sparse probing accuracy. Absorbed features are hypothesized to be lower-frequency (by definition of absorption pattern). If lower frequency causes both apparent absorption and lower probe accuracy, the result is confounded. *Must use frequency-matched controls, not just any control features.*
- **Statistical attack**: ANCOVA with frequency as covariate assumes linear relationship between frequency and accuracy. If the relationship is non-linear or threshold-based, ANCOVA is inappropriate. *Use non-parametric rank ANCOVA or stratify by frequency bins.*
- **Benchmark attack**: K-sparse probing is itself a SAEBench metric. If absorption affects SAEBench metrics broadly, the comparison may not be informative. *Consider using a non-SAE probing method (e.g., activation patching-based probing) as orthogonal validation.*
- **Ablation completeness attack**: Even if absorbed features show lower probing accuracy, we haven't established causation — absorption could be a symptom rather than cause. *Need intervention experiment: if we "fix" absorption (e.g., via MP-SAE re-encoding), does probe accuracy improve?*
- **Verdict**: MODERATE — The hypothesis is plausible and addresses a key gap, but requires careful matching and non-parametric analysis to avoid frequency confounding.

---

## Phase 4: Refinement

### Dropped Candidates

- **Candidate B (Method Comparison)**: Dropped because while methodologically valid, it does not produce a standalone publishable finding. The comparison would be most valuable as a component within Candidates A or C, not as an independent paper. We retain the insight that projection and ablation measure different phenomena for use in analysis.

### Strengthened Survivors

**Candidate A (Cross-Layer Quantification)** strengthened:
- Prior E2 results show layer 6 has highest absorption rate (0.549%) and most fragmented absorption graph (9 components). This is corroborating evidence for the mid-layer peak hypothesis.
- Prior E4 showed decoder cosine similarity (geometric signal) is more reliable than co-occurrence (confounded). Use decoder cosine as primary candidate selection criterion.
- Add frequency bin stratification: analyze absorption separately for high, medium, and low-frequency features to control for frequency confound.
- Add SynthSAEBench validation: use synthetic features with known ground-truth absorption to validate projection metric reliability on GPT-2.

**Candidate C (Sparse Probing Impact)** strengthened:
- Prior E5 shows absorbed features have significantly lower CV (1.07 vs 1.46, p=0.005). This is corroborating evidence that absorbed features have different activation profiles.
- The CV finding suggests absorbed features are more "stable" — if probing relies on activation patterns, stability may affect probe reliability in ways that go beyond frequency.
- Change from frequency-matched controls to CV-matched controls: match on coefficient of variation rather than raw frequency, since CV captures the stability dimension that may mediate absorption-probing relationship.
- Add intervention arm: compare probe accuracy on original SAE vs MP-SAE re-encoding of same features.

### Additional Controls

For all experiments, add:
1. **Dead feature ratio reporting**: Include dead feature % as covariate in all analyses (prior E2 showed this varies 0.02% to 17.3% across layers)
2. **Probe quality control**: Only include features where probe achieves >0.7 AUROC on held-out set (ensures probe reliability)
3. **Multiple correction**: Report both Bonferroni and FDR-corrected p-values for transparency

### Selected Front-Runner

**Candidate A (Projection-Based Cross-Layer Absorption Quantification)** is selected as the front-runner because:

1. It directly addresses **Gap 1** (systematic quantification across layers) — the most concrete gap with clear experimental methodology
2. It uses the more robust **projection-based metric** (SAEBench) rather than the degenerate ablation metric from prior experiments
3. The **pilot is fast** (10-15 min) and provides immediate signal about projection score distributions
4. **Novelty claim is clear**: No prior work has done systematic cross-layer absorption quantification using projection metrics across multiple model architectures
5. **Risk is manageable**: Even null results (no cross-layer variation) are publishable as a correction to the field's assumptions about absorption uniformity

Candidate C is retained as a secondary direction that can be addressed in the downstream impact section of the paper if Candidate A produces strong results.

---

## Phase 5: Final Proposal

### Title

**Layer-Dependent Feature Absorption in Sparse Autoencoders: A Projection-Based Quantification Across Model Architectures**

### Hypothesis

Feature absorption rates in pretrained SAEs follow a non-monotonic pattern across network depth, with peak absorption in mid-layers (layers 5-9) where feature representations transition from general to specialized. This pattern is robust across model architectures (GPT-2, Gemma-2-2B) and SAE configurations (TopK, JumpReLU) when measured using probe projection metrics, but differs from ablation-based estimates in both magnitude and layer profile.

### Falsification Criterion

The hypothesis is DISPROVEN if:
1. Absorption rates are statistically uniform across all layers tested (Kruskal-Wallis p > 0.05, no post-hoc significant pairs)
2. OR absorption rates show monotonic increase/decrease with layer depth without mid-layer peak
3. OR cross-model comparison shows no significant correlation in absorption layer profiles (r < 0.3)

### Method

**Absorption Measurement via Probe Projection (SAEBench protocol)**:
- For each candidate feature F, train a linear probe to detect F's activation from SAE latent activations
- Compute probe projection contribution: proportion of F's representational subspace accounted for by absorbing latents vs main latents
- Absorption score = contribution_from_absorbing_latents / (contribution_from_absorbing + contribution_from_main)
- Threshold for "absorbed": score > 0.5

**Candidate Selection**:
- Filter to alive features with activation frequency > 0.001 (to ensure reliable probes)
- Select top-100 candidates by decoder cosine similarity to other features (geometric signal)
- Stratify by layer and frequency bin for analysis

### Evaluation Protocol

**Primary Benchmarks**:
- GPT-2 residual stream SAEs (SAELens pretrained): layers 0, 3, 6, 9, 11
- GemmaScope JumpReLU SAEs: layers 0, 6, 12, 18 (for cross-model validation)

**Metrics**:
- Absorption rate per layer: % of candidates with absorption score > 0.5
- Mean absorption score with 95% bootstrap CI (10,000 resamples)
- Absorption score distribution (continuous, not just thresholded)

**Statistical Analysis**:
1. Kruskal-Wallis H-test: Do absorption scores differ across layers?
2. Post-hoc pairwise Wilcoxon tests with Benjamini-Hochberg FDR correction
3. Effect size (rank-biserial correlation) for each pairwise comparison
4. Cross-model correlation: Pearson r between GPT-2 and Gemma absorption layer profiles

**Number of Random Seeds**:
- 3 random seeds for candidate pair selection per layer
- Report mean and std across seeds

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| Metric comparison | Projection vs ablation reliability | Ablation underestimates deep-layer absorption; projection shows fuller picture |
| Frequency bin stratification | Frequency confound in absorption measurement | Absorption-high bins show similar patterns after stratification |
| Architecture comparison | TopK vs JumpReLU absorption profiles | Similar layer profiles but different absolute rates |
| Cross-model validation | GPT-2 vs Gemma-2-2B generalization | Correlated but not identical profiles |

### Control Experiments

1. **Synthetic ground-truth validation (SynthSAEBench)**: Run projection metric on synthetic features with known ground-truth absorption. Validate that projection recovers known absorption rates within 10%.
2. **Shuffle baseline**: Shuffle candidate labels and compute "absorption scores" to establish null distribution (expect mean ~0.3 by construction).
3. **Frequency-matched comparison**: For each absorbed candidate, compute a frequency-matched control to verify absorption signal is not purely a frequency artifact.

### Pilot Design (10-15 min)

Run projection-based absorption on GPT-2 layer 6 (the "hotspot" from prior E2):
- Load GPT-2 residual stream SAE layer 6 (24,576 features, ~99.9% alive)
- Select top-100 candidates by decoder cosine similarity
- Train linear probes for each candidate (500 examples, 80/20 train/val split)
- Compute projection-based absorption score for each
- Report distribution: mean, median, std, min/max, histogram

Expected pilot signal: If prior E2 findings are real, we should see mean absorption score > 0.35 (above the 0.3 shuffle baseline) with at least 15% of candidates above 0.5 threshold.

### Resource Estimate

| Resource | Specification | Time |
|----------|--------------|------|
| Model | GPT-2-small (85M) via SAELens | — |
| SAE | Residual stream, layers 0,3,6,9,11 (5 SAEs) | — |
| Compute | Single A100 GPU | ~45 min total |
| Candidates | 100 per layer × 5 layers = 500 total | — |
| Probe training | ~5 min per feature × 500 = 2,500 min (parallelized) | — |
| Expected runtime | With 10x parallelization: ~25 min | — |

Override justification: Project spec allows longer experiments. This is a comprehensive study with multiple layers and cross-model validation.

### Risk Assessment

**Biggest Threat 1: Projection metric unreliable on GPT-2 SAEs**
- SAEBench was validated on GemmaScope SAEs. GPT-2 may show different probe reliability.
- *Mitigation*: Use SynthSAEBench ground-truth validation before reporting GPT-2 results. Report probe AUROC as quality control metric.

**Biggest Threat 2: Frequency confound not resolved**
- Absorption signal may be explained by feature frequency alone.
- *Mitigation*: Stratified analysis by frequency bin. Frequency-matched control comparison.

**Biggest Threat 3: Basu et al. (2026) negative result relevance**
- If absorbed features produce no output change, why does absorption quantification matter?
- *Mitigation*: Acknowledge explicitly. Focus on what absorption tells us about SAE representation structure, not intervention efficacy. The paper's contribution is understanding SAE limitations, not fixing them.

### Novelty Claim

This is the **first systematic cross-layer quantification of feature absorption using probe projection metrics across multiple model architectures**. Prior work (Chanin et al., 2024) used ablation metrics limited to early layers (0-17). SAEBench introduced projection metrics but did not perform systematic cross-layer analysis. We provide:

1. **Layer absorption profiles** for GPT-2 and Gemma-2-2B with statistical comparisons
2. **Metric validation**: Ablation vs projection comparison with ground-truth synthetic data
3. **Frequency-controlled absorption rates**: First analysis to stratify by feature frequency
4. **Cross-architecture comparison**: TopK (GPT-2) vs JumpReLU (Gemma) absorption patterns

The experimental contribution answers the empirical question: **At which layers does absorption occur most severely, and does this vary across model architectures?** This is a foundational measurement that subsequent work on absorption mitigation (architectural or post-hoc) will reference.
