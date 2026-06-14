# Methodology: Systematic Analysis and Quantification of Feature Absorption in Sparse Autoencoders

## 1. Experimental Overview

This iteration tests four hypotheses (H1a-c, H2a-c, H3a-b, H4a-b) across four research questions using a pilot-first design. All experiments use ground-truth synthetic data (SynthSAEBench) supplemented by validation on GPT-2 small pretrained SAEs. The design prioritizes honest negative results, data integrity, and scope discipline.

**Key constraint from lessons learned**: Maximum 3-4 variants per iteration. No definitive rankings from incomplete data. Small-n correlations reported as exploratory observations only.

---

## 2. Experimental Framework

### 2.1 Ground-Truth Foundation: SynthSAEBench

We use synthetic data with known feature hierarchies to eliminate probe-based confounds:
- Each synthetic feature has explicit parent-child structure
- Absorption detection uses exact ground-truth labels (no logistic regression probes)
- Enables precise quantification without probe calibration bias

### 2.2 Validation Layer: GPT-2 Small

Pretrained GPT-2 small (124M) layer 8 residual stream SAEs from SAELens provide real-LLM validation:
- Use SAEBench absorption eval for comparability with prior work
- Cross-check synthetic findings against real model behavior

### 2.3 Data Integrity Pipeline (Mandatory)

Every experiment must pass:
1. **Feature count validation**: num_features == declared value
2. **Convergence verification**: loss curve plateaued, not early-stopped at spike
3. **Cross-seed independence**: MD5 hash of metrics across seeds to detect duplication
4. **Output file audit**: every planned experiment has a result file
5. **Numerical provenance**: every paper number traceable to single source file

---

## 3. RQ1: L0-Matched Architecture Comparison (H1a-c)

### 3.1 Variants

| Variant | Architecture | Core Mechanism | Prior Claim |
|---------|-------------|----------------|-------------|
| Baseline L1 | Standard ReLU + L1 | Sparse penalty | Reference (high absorption) |
| TopK | TopK selection | Explicit k-sparse | Worse than Baseline at low L0 |
| Matryoshka | Nested multi-scale | Multi-resolution dictionary | ~90% reduction (Bussmann et al.) |
| OrtSAE | Orthogonality penalty | Decoder orthogonality | ~65% reduction (claim) |

**Scope note**: 4 variants maximum per iteration. Gated and JumpReLU deferred to next iteration if pilot succeeds.

### 3.2 L0-Matching Protocol

1. Train each variant to target L0 = 50 (sparse regime) and L0 = 200 (moderate regime)
2. For Baseline, sweep lambda_L1 to match each variant's achieved L0
3. Compare absorption rates at matched L0

**Controls**:
- Random SAE (untrained dictionary): validates metric discrimination
- Shuffled feature labels: validates absorption detection is not artifactual

### 3.3 Metrics

- Absorption rate (ground-truth, synthetic)
- L0 (average active features)
- Explained variance (reconstruction quality)
- Dead feature ratio

### 3.4 Statistical Analysis

- Welch's t-test for each variant vs. L0-matched Baseline
- Cohen's d with pooled standard deviation
- Bonferroni correction for 4 comparisons

---

## 4. RQ2: Absorption-Downstream Causality (H2a-c)

### 4.1 Design: Two Independent Manipulations

**Manipulation A (Architectural)**: Use variants from RQ1 with naturally different absorption rates (~5% to ~50%).

**Manipulation B (Sparsity-induced)**: Fix architecture (Baseline L1) and vary lambda_L1:
- lambda_L1 = 5e-5 (low sparsity, low absorption)
- lambda_L1 = 8e-5 (medium)
- lambda_L1 = 1.2e-4 (high sparsity, high absorption)
- lambda_L1 = 1.6e-4 (very high)
- lambda_L1 = 2.0e-4 (extreme)

### 4.2 Downstream Metrics

1. **Sparse probing F1**: Linear probes on SAE latents for synthetic concept classification
2. **Steering efficacy**: Ablation effect size when steering with parent vs. child features
3. **Circuit-tracing precision**: Fraction of true parent-child edges identified by attribution

### 4.3 Causal Inference

If both Manipulation A and B show the same absorption-downstream correlation, this strengthens causal interpretation. If only B shows correlation, the effect is sparsity-driven, not absorption-specific.

### 4.4 Statistical Analysis

- Pearson r between absorption rate and each downstream metric (individual replicates, n >= 20)
- Linear regression controlling for architecture
- Report 95% CI for all correlations

---

## 5. RQ3: Mutual Coherence Theory (H3a-b)

### 5.1 Computation

For each trained SAE, compute decoder mutual coherence:
```
mu(W_dec) = max_{i != j} |cosine_similarity(W_dec[:, i], W_dec[:, j])|
```

### 5.2 Predictions

- H3a: Plot mu(W_dec) vs. absorption rate across all variants and seeds
- H3b: Test if mu < 1/(2k-1) predicts absorption onset (k = average L0)

### 5.3 Statistical Analysis

- Pearson r for H3a
- Logistic regression or piecewise linear model for H3b threshold
- Report AUC for threshold-based classification

---

## 6. RQ4: Task Generalization (H4a-b)

### 6.1 First-Letter Tasks

Use SAEBench absorption eval on GPT-2 small (standard benchmark).

### 6.2 Semantic Tasks

Design 3 semantic feature categories on synthetic data:
1. **Syntactic**: Part-of-speech features (noun, verb, adjective)
2. **Factual**: Country-capital relationships
3. **Safety**: Harmful request detection patterns

### 6.3 Comparison

Compute absorption rates for each category and test correlation with first-letter absorption.

### 6.4 Statistical Analysis

- Pearson r for H4a
- One-way ANOVA across feature categories for H4b

---

## 7. Expected Visualizations

### Tables
- **Table 1**: Main architecture comparison (variant x absorption rate x L0 x explained variance)
- **Table 2**: Dose-response results (absorption level x downstream metric)
- **Table 3**: Mutual coherence correlation summary

### Figures
- **Figure 1**: Architecture diagram showing overall method pipeline
- **Figure 2**: L0-matched absorption comparison (bar chart per variant)
- **Figure 3**: Dose-response curves (absorption vs. downstream metrics)
- **Figure 4**: Mutual coherence vs. absorption scatter plot with regression line
- **Figure 5**: Ablation study (bar chart per component removal)
- **Figure 6**: Training dynamics (loss curves per variant)

---

## 8. Baselines and Controls

### 8.1 Baselines

- **Random SAE**: Untrained dictionary to validate metric discrimination
- **L0-matched Baseline**: For each variant, a Baseline SAE trained to the same L0
- **Shuffled labels**: Feature labels randomly permuted to detect artifactual absorption

### 8.2 Ablation Studies

One ablation per proposed component:
- Remove Matryoshka nesting (flat dictionary)
- Remove OrtSAE orthogonality penalty (standard decoder)
- Remove TopK constraint (ReLU + L1 equivalent)

---

## 9. Software and Versions

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.0+ | Deep learning framework |
| SAELens | latest | SAE training and analysis |
| SAEBench | latest | Absorption evaluation benchmark |
| TransformerLens | latest | Activation extraction |
| NumPy | latest | Numerical computing |
| SciPy | latest | Statistical tests |
| Matplotlib | latest | Plotting |
| Seaborn | latest | Statistical visualization |

---

## 10. Reproducibility

- **Seed**: 42 for all experiments
- **Hardware**: Single GPU (NVIDIA A100 or equivalent, 40GB+ VRAM)
- **Checkpointing**: Save model checkpoints every 1000 steps
- **Logging**: Log all hyperparameters, metrics, and random seeds
- **Code versioning**: Git commit hash recorded with each experiment

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| No causal link found | Frame as important negative result; contrarian was right |
| Mutual coherence theory fails | Report negative result; theory may need refinement |
| Semantic detection unreliable | Use multiple detection methods; report method-dependent variance |
| Training time exceeds budget | Use pretrained SAELens SAEs where possible; SynthSAEBench is training-free |
| First-letter vs. semantic correlation weak | This is itself a finding; community needs to know |

---

## 12. Pre-registered Analysis Plan

### Primary Analyses (must report regardless of outcome)
1. H1a-c: Welch's t-test for each architecture vs. L0-matched Baseline
2. H2a-b: Pearson correlation with 95% CI between absorption and downstream metrics
3. H3a: Pearson correlation between mutual coherence and absorption

### Secondary Analyses (report if significant, note if not)
4. H2c: Linear regression of downstream metrics on absorption, controlling for architecture
5. H3b: Threshold model fit for mu = 1/(2k-1)
6. H4a-b: Cross-task correlation and ANOVA

### Exploratory Analyses (report with caveats)
7. Interaction effects: Does architecture moderate the absorption-downstream relationship?
8. Layer-wise patterns: Does absorption vary across model layers?
9. Dead latent confounding: Does dead latent rate correlate with absorption?
