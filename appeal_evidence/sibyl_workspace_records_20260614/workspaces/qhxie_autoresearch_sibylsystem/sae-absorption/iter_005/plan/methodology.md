# Experiment Methodology: Iteration 5

## Overview

This iteration executes a three-contribution study on SAE feature absorption:
1. **Confound Resolution**: Epidemiological causal inference to disentangle absorption from width/L0
2. **Cross-Domain Generalization**: First absorption measurement on RAVEL knowledge hierarchies
3. **Absorption Scaling Surface**: 2D phase map across 200+ SAEs with interaction testing

All analyses are training-free, operating on existing pre-trained SAEs (Gemma Scope, GPT-2 Small SAEs via SAELens) and precomputed SAEBench results.

## Phase 1: Confound Resolution (Zero GPU)

### Setup
- **Data**: iter_004 54-SAE dataset (`C3A_saebench_corr.json`) with absorption scores, quality metrics (sparse probing F1, SCR, RAVEL TPP, unlearning), width, layer, arch_class, and L0
- **Software**: Python 3.12 with numpy, scipy, pandas, statsmodels, pingouin (partial correlations), scikit-learn (propensity matching)
- **Critical**: Some SAEs in the dataset have `l0: null` (canonical SAEs without reported L0). These must be excluded from L0-controlled analyses or imputed via the SAEBench metadata lookup.

### Step 1.1: L0 as Covariate (GO/NO-GO)
- Compute `partial_corr(absorption, quality_metric | log_width, layer, arch_class, log_L0)` for all four quality metrics using pingouin
- Compare with iter_004 partial correlations (which controlled log_width, layer, arch_class but NOT L0)
- **Go criterion**: At least one quality metric retains |partial_r| > 0.2 after L0 control
- **No-go**: All four drop below |0.2| -> H1 falsified, pivot to characterizing L0 as the true driver

### Step 1.2: Width-Stratified Analysis
- Partition SAEs into width groups: 16k (~15 SAEs), 65k (~15 SAEs), 1M (~18 SAEs)
- Within each stratum: Spearman correlation between absorption and each quality metric
- BCa bootstrap 95% CIs (10,000 resamples) per stratum
- Report per-stratum n, effect sizes, CIs transparently
- Note: 1M stratum has the widest L0 range (9-207), giving best statistical power

### Step 1.3: Mediation Analysis
- Path model: L0 -> Absorption -> Quality (controlling for width, layer)
- Total effect: quality ~ log(L0) + log(width) + layer
- Direct effect: quality ~ log(L0) + absorption + log(width) + layer
- Indirect effect = total - direct; proportion mediated = indirect / total
- Sobel test + bootstrap CI (10,000 resamples) on indirect effect
- Software: statsmodels OLS with bootstrap; optionally pingouin.mediation

### Step 1.4: Rosenbaum Sensitivity Analysis
- Match high-absorption (>0.3) and low-absorption (<0.1) SAEs on width (exact) and L0 (nearest neighbor, caliper 0.2 SD)
- Compute quality difference for matched pairs
- Wilcoxon signed-rank test on differences
- Compute Rosenbaum Gamma at which test becomes non-significant
- Interpretation: Gamma > 1.5 = moderate robustness; Gamma > 2.0 = strong

### Step 1.5: SCR Suppression Variable Diagnosis
- Sequentially add covariates: width-only, layer-only, arch-only, L0-only
- Track how SCR partial correlation changes from -0.431 (bivariate) to -0.677 (full partial)
- Identify which covariate produces the suppression effect

### Step 1.6: Clustered SE Regression (C2C PMI)
- Rerun iter_004 C2C PMI regression with letter-level clustering (26 clusters)
- Report HC3 and clustered SE side by side
- Assess whether beta regression or zero-inflated model is needed (skewness=5.186)

### Baselines
- Null model: absorption uncorrelated with quality after controlling confounds
- Alternative baseline: L0 alone explains all quality variation (absorption is epiphenomenon)

### Metrics
- Partial correlation coefficients with CIs
- Proportion mediated with bootstrap CIs
- Rosenbaum Gamma sensitivity parameter
- Within-stratum Spearman rho with bootstrap CIs

## Phase 2: Cross-Domain Absorption (GPU, ~4-6 hours)

### Setup
- **Model**: Gemma 2 2B via TransformerLens (`google/gemma-2-2b`)
- **Data**: RAVEL city dataset (3000+ cities with country, continent, language attributes)
- **SAEs**: Gemma Scope 16k and 65k at layers 8, 12, 17 (loaded via SAELens `SAE.from_pretrained`)
- **Software**: transformer-lens, sae-lens, torch, sklearn (logistic regression probes)

### Step 2.1: Probe Training
- Load Gemma 2 2B with TransformerLens
- Construct prompt templates for each attribute:
  - City-Country: `"The city of {City} is located in"` (extract residual stream at last token)
  - City-Continent: `"The city of {City} is on the continent of"`
  - City-Language: `"The primary language spoken in {City} is"`
- Train multi-class logistic regression probes at layers 8, 12, 17
  - 3 seeds each, 80/20 train/test split
  - Use sklearn LogisticRegression(max_iter=1000, C=1.0)
- **Quality gate**: Probe accuracy >= 85% required to proceed; reject layers/attributes below threshold
- Save probe weights for downstream absorption measurement

### Step 2.2: Absorption Measurement (Adapted Chanin Metric)
- For each (layer, width, attribute) combination:
  1. **k-sparse probing**: Find k feature splits for each attribute value (e.g., "France", "Germany") using SAE features
  2. **False-negative identification**: Find tokens where all k split latents fail to activate but probe correctly classifies
  3. **Integrated-gradients ablation**: On false-negative tokens, identify which SAE latent has highest ablation effect
  4. **Absorption detection**: Highest-ablation latent has cosine > 0.025 with parent probe AND dominance > 1.0 over second-highest
- Threshold sweep: cosine similarity in {0.01, 0.025, 0.05, 0.1}; dominance in {0.5, 1.0, 2.0}
- Compute absorption rate per attribute per layer per SAE width

### Step 2.3: Controls
- **Shuffled hierarchy**: Randomize city-attribute mappings, re-measure absorption (expect < 5%)
- **Random probe direction**: Use random unit vectors as "probe" directions (expect < 5%)
- **First-letter baseline**: Replicate Chanin first-letter measurement on same SAEs for direct comparison
- **Dead feature exclusion**: Report separately for dead vs. alive features

### Step 2.4: Cross-Domain Comparison
- Side-by-side absorption rates: first-letter vs. country vs. continent vs. language
- Compute hierarchy sharpness: mutual information between entity and attribute
- Correlate absorption severity with hierarchy sharpness (Spearman)
- Early/late absorption taxonomy on knowledge features (tau sweep 0.2-0.4)

### Baselines
- Shuffled-hierarchy control (expect < 5% absorption)
- Random probe direction control (expect < 5%)
- First-letter spelling absorption as reference point (15-35% from literature)

### Metrics
- Absorption rate per attribute per layer per SAE width
- Absorption rate ratio: real / shuffled control
- Hierarchy sharpness (MI) per attribute
- Correlation between absorption rate and hierarchy sharpness

## Phase 3: Absorption Scaling Surface (Zero GPU, ~1 hour)

### Setup
- **Data**: SAEBench precomputed results from HuggingFace (`adamkarvonen/sae_bench_results`)
- Load via `datasets.load_dataset("adamkarvonen/sae_bench_results")` or direct JSON/parquet download
- Extract: absorption score, L0, width, layer, architecture class for 200+ SAEs
- **Software**: pandas, numpy, scipy, pyGAM (or statsmodels GAM), matplotlib

### Step 3.1: Data Collection & Cleaning
- Download SAEBench results for all available Gemma Scope SAEs
- Merge absorption scores with SAE metadata (width, L0, layer, architecture)
- Filter to SAEs with both absorption score and L0 available
- Report: total N, breakdown by width x L0 x layer

### Step 3.2: Phase Surface Construction
- Plot absorption rate as 2D heatmap in (log2(width), log2(L0)) space
- Fit GAM: `absorption ~ s(log_width) + s(log_L0) + ti(log_width, log_L0) + layer`
  - `s()`: smooth univariate terms
  - `ti()`: tensor interaction (tests whether absorption depends on joint width-L0 structure)
- Test interaction term significance (p < 0.05)
- Report deviance explained by additive model vs. interaction model
- Visualize as contour plot with labeled regions

### Step 3.3: Phase Boundary Detection
- Compute gradient magnitude of the GAM surface at grid points
- Identify ridges (regions of maximal gradient = candidate phase boundaries)
- If sharp boundary exists: characterize the transition zone (width range, L0 range)
- If no sharp boundary: fit smooth scaling law with confidence bands
- Three predicted regimes from proposal: hedging (low W), absorption (intermediate W, low L0), recovery (high W, high L0)

### Baselines
- Additive model (no interaction): absorption ~ s(log_width) + s(log_L0)
- Linear model: absorption ~ log_width + log_L0

### Metrics
- GAM interaction term p-value and deviance explained
- Gradient surface ridge characteristics
- R-squared of linear vs. additive vs. interaction models
- Absorption rate contour plot

## Phase 4: Taxonomy Correction (GPU, ~1-2 hours)

### Setup
- **Data**: iter_004 taxonomy results (`C2D_taxonomy.json`) with per-letter classification
- **Model**: GPT-2 Small (open-model anchor for validation)
- **SAEs**: GPT-2 Small SAEs from SAELens

### Step 4.1: Proper Comparison Tokens
- For each letter with n_comparison_tokens=0, use tokens from the same log-frequency band
- Frequency-matched sampling: for letter X with target tokens at frequency f, sample comparison tokens from frequency band [0.5f, 2f] that do not start with X
- Alternative: use sae-spelling ground-truth parent feature IDs from Chanin labels

### Step 4.2: Recomputed Classification
- Rerun Type II classification with corrected baselines
- Report: Type I validated rate, Type II corrected rate, Chanin replication rate
- Mark original 92.3% as "upper bound (uncorrected)"
- Compute corrected combined rate with bootstrap CI

### Baselines
- Original taxonomy rates from iter_004 (92.3% combined)
- Frequency-matched comparison tokens

### Metrics
- Corrected Type II rate
- Corrected combined absorption rate with 95% bootstrap CI
- Comparison with original rates

## Expected Visualizations

- **Table 1**: Main confound resolution results -- partial correlations before/after L0 control, per quality metric
- **Table 2**: Width-stratified correlations (16k, 65k, 1M) with bootstrap CIs
- **Table 3**: Mediation analysis results -- total, direct, indirect effects and proportion mediated
- **Table 4**: Cross-domain absorption rates (first-letter vs. country vs. continent vs. language)
- **Figure 1**: Absorption scaling surface -- 2D contour plot in (log(width), log(L0)) space
- **Figure 2**: Phase boundary detection -- gradient magnitude surface with ridge overlay
- **Figure 3**: Cross-domain comparison -- bar chart of absorption rates with shuffled controls
- **Figure 4**: Mediation path diagram with standardized coefficients
- **Figure 5**: Rosenbaum sensitivity plot -- Gamma vs. p-value for matched pair analysis
- **Figure 6**: Corrected taxonomy -- stacked bar chart comparing original vs. corrected rates

## Software Dependencies

```
transformer-lens>=2.0.0
sae-lens>=4.0.0
torch>=2.0
numpy
scipy
pandas
statsmodels
pingouin
scikit-learn
pyGAM
matplotlib
seaborn
datasets  # HuggingFace datasets for SAEBench
```

## Reproducibility

- All random seeds fixed at 42 (pilot), 42/123/456 (full)
- All SAE IDs and HuggingFace release names logged
- Probe training uses fixed train/test splits with stratification
- Bootstrap resampling uses fixed seed sequences
- All threshold sweeps use pre-registered values
