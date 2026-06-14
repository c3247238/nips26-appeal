# Methodology: Feature Absorption and Downstream SAE Reliability

## Overview

This study provides the first systematic, quantitative bridge between feature absorption detection and downstream interpretability task performance. The methodology is entirely training-free, leveraging pre-trained SAEs from Gemma Scope and SAEBench. All experiments use public benchmarks and include rigorous baselines, controls, and ablation studies.

## Research Questions & Pre-registered Hypotheses

| ID | Hypothesis | Falsification Criterion | Type |
|---|---|---|---|
| H1 | Features with higher absorption rates exhibit proportionally lower steering effectiveness: S = S_0 * (1 - k * A) | Pearson |r| < 0.30 with p > 0.05 for n >= 100 features | Primary |
| H2 | Features with higher absorption rates exhibit lower sparse probing F1 scores (monotonic relationship) | Spearman rho < 0.20 with p > 0.05 | Secondary |
| H3 | The absorption-degradation relationship is consistent across layers/dictionary sizes within a model family | Significant interaction effect (p < 0.05) in ANOVA | Exploratory |

## Power Analysis

| Effect Size | n needed for 80% power | n needed for 90% power | Our n |
|---|---|---|---|
| |r| = 0.30 (small) | 84 | 112 | 100+ (pilot), 260+ (full) |
| |r| = 0.40 (medium) | 46 | 62 | Adequate |
| |r| = 0.50 (large) | 28 | 38 | Highly adequate |

**Interpretation**: Our pilot (n=100) is adequately powered for medium-to-large effects. The full study (n=260+) is powered for small effects. Observed null results will be interpreted in the context of this power analysis.

## Experimental Design

### Phase 1: Absorption Detection

For each SAE configuration, we:
1. Load pre-trained SAE via SAELens
2. Run the Chanin et al. absorption metric (via SAEBench or sae-spelling)
3. On the first-letter task: for each letter A-Z, detect if "starts with X" feature is absorbed
4. Record: absorption rate per feature, absorbing latent IDs, ablation effect magnitudes
5. Classify features into: HIGH absorption (>50%), MEDIUM (10-50%), LOW (<10%)

**SAE Configurations (Primary)**:
- Gemma-2-2B, Gemma Scope SAEs, layers 8, 12, 16, dictionary sizes 16K and 65K
- Pythia-160M, SAEBench SAEs, layer 8, dictionary size 32K (validation)

### Phase 2: Feature Steering Experiment

For each "starts with X" feature identified in Phase 1:
1. Extract feature direction from SAE decoder: `direction = sae.W_dec[feature_id]`
2. Generate test prompts containing words starting with target letter (100 prompts per letter)
3. Run steering at strengths: [1.0, 2.0, 5.0, 10.0]
4. Measure: does steering increase probability of target-letter words?
5. Success metric: % increase in target-letter token probability vs. unsteered baseline

**Controls (CRITICAL - addressing iteration 0 gaps)**:
- **Random feature baseline**: Steer 26 randomly selected SAE latents, compare success rates
- **Null steering baseline**: Steering strength = 0 (no effect expected)
- **Shuffled control**: Shuffle absorption labels, recompute correlation (should be ~0)
- **Match HIGH and LOW absorption features for base activation strength** where possible

### Phase 3: Sparse Probing Experiment

For each SAE:
1. Train k-sparse linear probe (k=1, 5, 10) on first-letter classification
2. Measure probe F1 score using: (a) only main feature latents, (b) all latents
3. Compare F1 degradation for absorbed vs. non-absorbed features

### Phase 4: Correlation Analysis

1. Compute Pearson/Spearman correlation between absorption rate and steering effectiveness
2. Compute correlation between absorption rate and probing F1 degradation
3. Fit linear model: `task_degradation = beta * absorption_rate + epsilon`
4. Report R^2, confidence intervals, and p-values
5. Test for consistency across layers, dictionary sizes, and model families

## Baselines

1. **No-absorption baseline**: Features with <5% absorption rate
   - Expected: High steering effectiveness (>80% success), high probing F1 (>0.9)

2. **Random feature baseline**: Randomly selected SAE latents
   - Expected: Near-zero steering effectiveness, random probing F1

3. **Full-activation baseline**: Using all SAE latents for probing
   - Expected: Higher F1 than k-sparse probing; measures recoverability

4. **Shuffled label control**: Randomly permute absorption labels
   - Expected: Correlation ~0; validates that relationship is not spurious

## Evaluation Benchmarks

| Benchmark | Purpose | Metric |
|---|---|---|
| First-letter classification (A-Z) | Primary task for steering and probing | Steering success rate, probing F1 |
| Random feature steering | Control for non-specific steering effects | Success rate vs. first-letter features |
| Null steering | Control for baseline generation behavior | No effect expected |
| Shuffled label correlation | Control for spurious correlations | Correlation ~0 |

## Metrics

| Metric | Description | Target |
|---|---|---|
| Steering Success Rate | % increase in target-letter token probability | Baseline: ~0; Good: >50% |
| Probing F1 | F1 score for first-letter classification | Baseline: random; Good: >0.8 |
| Absorption Rate | Chanin differential correlation metric | Range: 0-1 |
| Pearson r | Linear correlation (absorption vs. degradation) | Significant if |r| > 0.30, p < 0.05 |
| Spearman rho | Rank correlation | More robust to outliers |
| R^2 | Variance explained by linear model | Higher = stronger relationship |

## Expected Visualizations

- **Figure 1**: Architecture diagram showing absorption detection pipeline and downstream task evaluation
- **Table 1**: Main results comparison (feature set x absorption rate x steering success x probing F1)
- **Figure 2**: Scatter plot - absorption rate vs. steering success (with regression line, R^2, p-value)
- **Figure 3**: Scatter plot - absorption rate vs. probing F1 degradation
- **Figure 4**: Ablation study - bar chart showing effect of each control/baseline
- **Figure 5**: Layer comparison - absorption-degradation relationship across layers (if H3 tested)
- **Figure 6**: Power analysis table - sample size vs. effect size vs. power

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No absorbed features found in selected set | Low | High | Use Chanin validated first-letter task; most SAEs show absorption |
| Steering effectiveness is noisy | Medium | Medium | Multiple prompts per feature; statistical testing; random baseline |
| Absorption doesn't correlate with degradation | Medium | High | Core hypothesis; null result is still publishable with proper power analysis |
| SAEBench absorption metric too slow | Low | Low | Use sae-spelling directly; same metric |
| Gemma Scope SAEs unavailable | Low | High | Fallback to SAEBench collection or Pythia SAEs |
| First-letter task too narrow | Medium | Medium | Supplement with semantic hierarchy features (WordNet) in follow-up |
| Steering bypasses encoder (robustness confound) | High | High | Explicitly discuss encoder-vs-decoder distinction; frame findings accordingly |

## Resource Requirements

| Item | Estimate |
|---|---|
| GPU | Single 24GB GPU (RTX 3090/4090 or A10) |
| Pilot | ~3 GPU-hours |
| Full study | ~25 GPU-hours |
| Wall-clock (with parallelization) | ~2-3 days |
| Model sizes | Gemma-2-2B (primary), Pythia-160M (validation) |
| Storage | <10GB for models + SAEs |

All experiments are training-free, using only pre-trained models and SAEs.
