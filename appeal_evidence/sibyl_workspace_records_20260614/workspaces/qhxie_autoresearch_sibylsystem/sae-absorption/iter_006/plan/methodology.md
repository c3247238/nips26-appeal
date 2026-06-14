# Methodology: Cross-Domain Absorption Characterization with Rate-Distortion Diagnostics and Confound Decomposition

## Iteration Context

This is iteration 6, synthesis round 2. Candidate: `cand_cross_domain_ratedistortion`.

Iterations 1-5 established:
- GPT-2 Small is insufficient for cross-domain absorption (98% dead features on city prompts)
- EDA (1 - encoder-decoder cosine) is identical to SAEBench's existing metric (not novel)
- ITAC evaluated on synthetic activations only; needs real-activation validation
- Proxy labels inflate AUROC (L12-65k pilot AUROC=0.853 collapsed to 0.468 with full evaluation)
- Cross-domain H3 was FALSIFIED with proper shuffled control on GPT-2 Small
- Three-subtype taxonomy is threshold-sensitive (tau=0.2: 32% early; tau=0.3: 72% early)
- Within-width null (Rosenbaum Gamma=1.0) means absorption-quality associations vanish when width held constant

Iteration 6 pilot results:
- First-letter: 13.4% absorption (within published 15-35% range)
- City-country: 0.0% (probe F1 = 0.602, below gate)
- City-continent: 6.5% (CI: 0-11.5%, probe F1 = 0.795)
- City-language: 6.6% (probe F1 = 0.745)
- Animal-class: 1.4% (probe F1 = 0.696)
- ITAC: no clear separation (median 1.35 vs random 1.14, not significant)
- Confound: 96.9% hierarchy-driven at L0=22

**Key revisions from idea_debate**:
1. Rate-distortion theory elevated to co-primary contribution (novelty 8/10)
2. Unsupervised detection de-prioritized (ITAC pilot negative)
3. H1 threshold lowered from 10% to 5% based on pilot
4. Impossibility theorem dropped (saturated landscape)
5. Lateral inhibition bifurcation analysis added (H7)
6. Domain-dependent absorption rates framed as primary finding

---

## Stage 0: Environment Setup and Smoke Test

### 0.1 Environment Setup
- Install dependencies: `sae-lens>=4.0`, `transformer-lens`, `torch>=2.1`, `scikit-learn`, `scipy`, `pygam`
- Download Gemma 2 2B model weights (HuggingFace, ~5GB)
- Download Gemma Scope SAEs: layers 10, 12, 20; widths 16k and 65k (residual stream)
- Multiple L0 operating points for layer 12: L0={22, 42, 82, 163} (for multi-L0 confound decomposition)
- Verify SAELens can load each SAE config via `SAE.from_pretrained()`
- Smoke test: encode 100 tokens through model+SAE, verify non-zero activations and <20GB VRAM

### 0.2 SAELens Configuration Notes
- Use `SAE.from_pretrained(release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_22")` for loading
- Gemma Scope SAEs are JumpReLU architecture (hard threshold) -- critical for H7 bifurcation analysis
- L1-trained SAEs (if available via SAELens `gpt2-small-res-jb` or custom training) needed for H7 comparison
- Decoder matrix accessed via `sae.W_dec` (shape: [d_sae, d_model])
- Encoding: `sae.encode(activations)` returns sparse features

---

## Stage 1: Cross-Domain Absorption Characterization (H1, H6)

### 1.1 First-Letter Validation (Baseline, Improved)
Reproduce and improve Chanin et al. first-letter absorption measurement on Gemma 2 2B.

**Protocol**:
1. Construct ICL prompts for 26 first-letter classes, **50+ words per letter** (up from 25 in pilot)
2. Train k-sparse (k=5) logistic regression probes per letter on SAE latent activations
3. Probe quality gate: F1 > 0.85 per letter (exclude letters failing gate; report separately)
4. Identify false-negative tokens: all k split latents fail to activate, but probe correctly classifies
5. Apply Chanin absorption criteria (cosine > 0.025, magnitude gap >= 1.0)
6. Compute per-letter and aggregate absorption rate with bootstrap 95% CI (10,000 resamples, seed=42)
7. Compare against published 15-35% rates; flag if differs >3x

**Controls**:
- C1: Random probe (random direction) -- target < 2%
- C2: Shuffled labels (random letter assignment) -- investigate pilot's 59.5% rate
- C3: Dense probe ceiling (LR on raw model activations)
- C4: Untrained SAE (if feasible)

**SAE configs**: L12 {16k, 65k}; L10 16k; L20 16k

### 1.2 Cross-Domain Measurement

Five hierarchy domains measured on Gemma 2 2B + Gemma Scope:

| Domain | Parent Feature | N_parents | Source | Pilot Rate |
|--------|---------------|-----------|--------|-----------|
| First-letter | "starts with X" | 25 | Chanin et al. | 13.4% |
| City -> Country | "in France" | 28 | RAVEL | 0.0% |
| City -> Continent | "in Europe" | 6 | RAVEL | 6.5% |
| City -> Language | "French-speaking" | 18 | RAVEL | 6.6% |
| Animal -> Class | "mammal" | 5+ | WordNet | 1.4% |

**Per domain, per SAE**:
1. Construct ICL prompts adapted for each domain
2. Train k-sparse probes with quality gate (F1 > 0.85); stratify by quality tier (>0.85 vs 0.70-0.85)
3. Measure absorption rate via Chanin et al. metric
4. Threshold sensitivity: full 5x4 grid (cosine {0.01, 0.02, 0.025, 0.03, 0.05} x gap {0.5, 1.0, 1.5, 2.0})
5. Report per-parent and per-domain absorption rates with bootstrap 95% CIs
6. Shuffled control per domain

**Exclusion criteria**: Countries with < 5 cities OR probe F1 < 0.50 excluded from absorption claims.

### 1.3 Hierarchy Predictor Analysis (H6)

For each domain, compute:
- Parent-child co-occurrence frequency ratio (from 1M-token corpus)
- Fan-out (children per parent)
- Hierarchy depth
- Parent feature frequency

Correlate each predictor with measured absorption rate across domains:
- Spearman rho with Bonferroni correction (4 predictors x N domains)
- Linear model: absorption_rate ~ co_occurrence_ratio + fan_out + parent_frequency + L0
- Partial correlations controlling for L0

---

## Stage 2: Confound Decomposition and Multi-L0 Profiling (H2)

This is a primary contribution. Extends pilot evidence (96.9% hierarchy-driven at L0=22).

### 2.1 Multi-L0 False-Negative Decomposition

For first-letter task on Gemma 2 2B L12 16k, at L0 = {22, 42, 82, 163}:
1. Identify all false-negative tokens at each L0
2. Classify each false negative as:
   - **(a) Hedging**: Token recoverable at higher L0 (parent latent fires at different L0 operating point)
   - **(b) Reconstruction error**: Co-occurring with high reconstruction error (residual norm > 2 sigma above mean)
   - **(c) Hierarchy-driven**: Persistent across L0 values AND low reconstruction error -- genuine absorption
3. Compute fraction of each type at each L0
4. Test prediction: hierarchy-driven fraction increases at optimal L0 (~22), decreases at too-low L0 (hedging dominates) and too-high L0 (reconstruction error dominates)

### 2.2 Threshold Sensitivity Analysis

Full 5x4 grid (cosine threshold x magnitude gap) on first-letter:
- Report CV across thresholds per SAE config
- Report Kendall tau rank stability (do rankings of letters change with thresholds?)
- CV < 0.3 = stable metric; CV > 0.5 = brittle

### 2.3 L0 Covariate Analysis

Across all available Gemma Scope SAEs:
- Partial correlations: absorption rate vs quality metrics (SP-F1, SCR, TPP, Unlearning)
- Controlling for: log(L0), log(width), architecture_class
- Report both FDR-corrected (Benjamini-Hochberg) and uncorrected p-values

### 2.4 Rosenbaum Matching

Two matching strategies:
- Mahalanobis matching (cross-width): standard confound control
- Within-width matching: test whether absorption-quality associations survive when width is constant
- Report Gamma divergence explicitly for both
- Within-width null (Gamma ~1.0) is expected and reported as a primary finding about confound structure

---

## Stage 3: Rate-Distortion Theory and Information-Theoretic Diagnostics (H3, H7)

This is the proposal's most novel contribution (novelty score: 8/10).

### 3.1 Theoretical Framework

**Theorem sketch** (Successive Refinement, Equitz & Cover 1991):

Let X be the LLM residual stream activation. For a parent-child feature pair with hierarchical co-occurrence, define CMI = I(X; w_parent | f_child). Then:
- Absorption is rate-distortion optimal when CMI < lambda / c(w_P, w_C)
- Absorption is suboptimal (destroys information) when CMI exceeds this threshold
- Absorption is lossless iff X -- f_child -- f_parent forms a Markov chain (CMI = 0)

### 3.2 CMI-Absorption Correlation (Primary Test for H3)

For all 25 first-letter features on Gemma 2 2B L12 16k:
1. Compute I(X; w_parent | f_child) using k-NN MI estimator (Kraskov et al., 2004)
2. Project onto decoder-direction subspace (d' = {10, 20, 30, 50}) for reliable estimation
3. Correlate with observed absorption rate
4. Target: Spearman rho < -0.3 (letters with higher CMI resist absorption)
5. Report at multiple subspace dimensions to assess estimation stability

### 3.3 ITAC-CMI Connection (Theory-Practice Bridge)

Validate that ITAC is a scalar proxy for CMI:
- ITAC = Var(residual projection | child active, parent inactive) / Var(same | neither)
- Compute both CMI and ITAC for same 25 first-letter features
- Target: Spearman correlation between CMI and ITAC > 0.5
- This bridges the information-theoretic diagnostic (CMI) with the practical metric (ITAC)

### 3.4 Geometric Constant Validation

Compute c(w_P, w_C) = ||w_P||^2 * (1 - cos^2(w_P, w_C)) from decoder weights for all first-letter pairs:
- Test whether c modulates the absorption threshold beyond CMI alone
- Target: CMI/c correlates more strongly with absorption than CMI alone
- Can be computed directly from `sae.W_dec` without additional forward passes

### 3.5 Lateral Inhibition Check (Per-Token Prediction)

For 100k tokens:
1. Compute inhibition ratio q = G_PC * a_C / (theta_P - b_P) per token
2. G_PC = decoder weight interaction, a_C = child activation, theta_P = parent threshold
3. Test: q > 1 predicts per-token false negatives
4. Target: precision@50 > 0.3 for the q > 1 predictor

### 3.6 Lateral Inhibition Bifurcation Analysis (H7)

Compare absorption patterns between JumpReLU and L1 SAEs:
- JumpReLU (Gemma Scope): hard threshold -- predicted to show binary (all-or-nothing) absorption
- L1+ReLU (if available via SAELens/custom): soft threshold -- predicted to show continuous absorption distribution
- Test: distribution of per-parent absorption rates for JumpReLU should be bimodal vs continuous for L1
- KS test for distribution difference; target p < 0.1
- Note: Requires either finding L1-trained SAEs for Gemma 2 2B or training small custom L1 SAE

### 3.7 Phase Transition Prediction

From theory, compute predicted critical L0 value where absorption becomes rate-distortion optimal:
- Compare theoretical prediction with observed L0 ~ 7-14 transition from iteration 5 scaling analysis
- If match, this validates the theory's quantitative predictive power

### 3.8 Cross-Domain CMI Extension

If H3 is supported (rho < -0.3 on first-letter):
- Extend CMI computation to city-continent and city-language hierarchies
- Test whether CMI predicts cross-domain absorption rate differences
- This bridges the theory (Stage 3) with the empirical characterization (Stage 1)

---

## Stage 4: Scaling Surface (H5)

### 4.1 Multi-SAE Absorption Surface

Measure absorption rate across all available Gemma Scope SAE configurations:
- Multiple widths: 16k, 32k, 65k, 131k (where available)
- Multiple L0 operating points per width
- Multiple layers: 10, 12, 20
- Record architecture type (JumpReLU for all Gemma Scope; flag if others found)

### 4.2 GAM Analysis with Architecture Covariate

Generalized Additive Model:
- Response: absorption rate (bounded [0,1])
- Predictors: s(log_width) + s(log_L0) + te(log_width, log_L0) + arch_class
- Family: Beta regression (Beta-GAM)
- **Critical fix from iteration 5**: Include architecture as covariate (360 standard + 54 JumpReLU + 6 unknown)
- Report: interaction p-value, explained deviance, within-architecture subset analysis
- Software: pyGAM, report version and spline specification

### 4.3 Phase Transition Detection

- Identify critical L0 where absorption transitions
- Compare with theoretical prediction from Stage 3.7
- Cross-domain scaling comparison if sufficient data

---

## Stage 5: Unsupervised Detection (H4, Secondary)

Given pilot evidence of ITAC weakness, this is de-prioritized.

### 5.1 Decoder Geometry Analysis

For Gemma Scope L12 16k:
1. Pairwise cosine similarity matrix (16k x 16k)
2. Conditional cosine similarity for pairs with global cosine > 0.1
3. Firing rate asymmetry filter (100k tokens)
4. Hierarchical clustering dendrogram

### 5.2 ITAC on Real Activations

Critical improvement over iteration 5:
- Compute ITAC on real model activations (100k tokens from Pile), NOT synthetic decoder columns
- For each candidate pair from 5.1: compute residual projection variance ratio in 4 groups
- ITAC null test: random pairs should yield ITAC ~1.0
- Try larger corpus (500k tokens) and stricter candidate filtering

### 5.3 Unsupervised Validation

Validate against first-letter gold standard:
- Spearman rho, AUROC, Precision@50
- Component ablation
- **Pre-registered decision**: If rho < 0.3, report as negative result. Do not deploy cross-domain.

---

## Expected Visualizations

- **Figure 1**: Architecture diagram -- absorption mechanism and three-pronged study overview
- **Figure 2**: Cross-domain absorption rates (bar chart per domain with CI, first-letter reference)
- **Table 1**: Main cross-domain results (domain x absorption rate x probe F1 x controls)
- **Figure 3**: CMI vs absorption rate scatter (25 first-letter features, with regression line)
- **Figure 4**: Multi-L0 confound decomposition (stacked bar: hedging / reconstruction / hierarchy-driven)
- **Table 2**: Confound decomposition across L0 values
- **Figure 5**: Scaling surface heatmap (width x L0 x absorption rate)
- **Figure 6**: Lateral inhibition bifurcation (absorption rate distributions: JumpReLU vs L1)
- **Table 3**: Threshold sensitivity grid (cosine x gap x absorption rate)
- **Table 4**: Hierarchy predictors (domain x predictor x absorption rate x rho)
- **Figure 7** (appendix): Unsupervised detection ROC curves (if H4 pursued)
- **Figure 8** (appendix): Probe quality per domain (bar charts by parent)

---

## Statistical Methodology

### Pre-registered corrections
- **FDR**: Benjamini-Hochberg across all hypothesis tests
- **Bootstrap CIs**: 10,000 resamples, seed=42
- **Minimum sample sizes**: n >= 100 for absorption measurement (pilot showed n=16 causes signal reversal)
- **Within-width analysis**: Reported prominently, not buried

### Language discipline
- "Statistical association" not "causal effect" for observational data
- "Statistical mediation" not "causal mediation" for observational decomposition
- Within-width null reported in Results section with equal prominence
- Negative results (H4 unsupervised, H7 if KS non-significant) reported explicitly

### Decision gates
- **After Exp 1a**: Controls calibrated? (shuffled < 20% after refinement) -- if not, investigate before proceeding
- **After Exp 3a**: CMI-absorption rho < -0.2? -- if not, de-prioritize rate-distortion framing
- **After Exp 5c**: Unsupervised rho > 0.3? -- if not, report as negative result
- **Overall**: H1+H2+H3 supported = strong paper; H1+H2 only = moderate; H3 only = pivot to theory-first paper

---

## Resource Requirements

- **Model**: Gemma 2 2B (~5GB VRAM inference)
- **SAEs**: Gemma Scope pre-trained (HuggingFace); multiple L0 configs for L12 16k
- **GPU**: Single GPU with >= 24GB VRAM. Gemma 2 2B + 16k SAE ~10GB; 65k SAE ~16-20GB
- **Batch size**: 32-64 sequences for A100 80GB; 16-32 for 24GB GPUs. Inference only (no training).
- **Disk**: ~25GB (model + SAEs + cached activations)
- **Software**: SAELens >= 4.0, TransformerLens, PyTorch >= 2.1, scikit-learn, scipy, pyGAM
- **Total GPU-hours**: ~8-10 hours across all experiments
- **Training**: Zero for main pipeline. May need ~1-2 hours if training custom L1 SAE for H7 bifurcation.
