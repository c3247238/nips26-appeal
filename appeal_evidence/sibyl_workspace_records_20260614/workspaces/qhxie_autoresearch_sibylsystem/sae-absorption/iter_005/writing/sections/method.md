# 3 Method

This study comprises four analysis phases: (1) confound resolution via epidemiological causal inference applied to 48 Gemma Scope SAEs, (2) cross-domain absorption measurement on knowledge hierarchies using GPT-2 Small, (3) absorption scaling surface construction across 420 SAEs with formal interaction testing, and (4) taxonomy correction for the original absorption classification. All analyses are training-free, operating on existing pre-trained SAEs and precomputed SAEBench results. Figure 2 illustrates the mediation path model central to Phase 1.

## 3.1 Confound Resolution via Epidemiological Causal Inference

### 3.1.1 Dataset

The confound resolution analysis uses 48 Gemma Scope SAEs (Gemma 2 2B) from SAEBench (Karvonen et al., 2025) with known $L_0$ values. Six canonical SAEs lacking reported $L_0$ are excluded. Each SAE has a precomputed absorption score (Chanin et al., 2024) and four downstream quality metrics: sparse probing F1 ($\text{SP-F1}$), spurious correlation removal ($\text{SCR}$), RAVEL true positive proportion ($\text{TPP}$), and unlearning ($\text{UL}$). SAE dictionary widths span 16,384 to 1,048,576 and $L_0$ ranges from 9 to 297. After filtering to SAEs with $L_0$ available, all 48 SAEs share a single architecture class (JumpReLU), so architecture is dropped as a covariate.

### 3.1.2 Step 1: L0 as Covariate (Go/No-Go)

We compute partial Pearson correlations between absorption rate $R_{\text{abs}}$ and each quality metric, controlling for $\log(\text{width})$, layer, and $\log(L_0)$:

$$r_{\text{partial}}(R_{\text{abs}}, Q_i \mid \log m, \ell, \log L_0)$$

where $Q_i \in \{\text{SP-F1}, \text{SCR}, \text{TPP}, \text{UL}\}$, $m$ is dictionary width, and $\ell$ is layer index. This extends the prior analysis (which controlled for $\log m$ and $\ell$ but not $L_0$) by adding the critical confound identified in Section 2.4.

**Go/no-go criterion**: At least one quality metric must retain $|r_{\text{partial}}| > 0.2$ after $L_0$ control. If all four drop below this threshold, the absorption-quality association is an artifact of the width/$L_0$ confound, and H1 is falsified.

Collinearity is assessed via variance inflation factors (VIF). With $\text{VIF}(\log L_0) = 1.09$ and $\text{VIF}(\log m) = 1.08$, multicollinearity is not a concern.

### 3.1.3 Step 2: Width-Stratified Analysis

SAEs are partitioned into three width strata: 16k ($n = 15$), 65k ($n = 15$), and 1M ($n = 18$). Within each stratum, we compute Spearman rank correlations $\rho$ between $R_{\text{abs}}$ and each quality metric, with BCa bootstrap 95% confidence intervals (10,000 resamples). This tests whether the absorption-quality association holds within fixed width or is driven entirely by cross-width variation.

### 3.1.4 Step 3: Baron-Kenny Mediation Analysis

We test the causal path $\log(L_0) \to R_{\text{abs}} \to Q_i$ using the Baron-Kenny four-step procedure, controlling for $\log m$ and $\ell$:

1. **Path $a$**: $R_{\text{abs}} = \beta_0 + a \cdot \log(L_0) + \gamma_1 \log m + \gamma_2 \ell + \epsilon$
2. **Path $b$**: $Q_i = \beta_0 + b \cdot R_{\text{abs}} + c' \cdot \log(L_0) + \gamma_1 \log m + \gamma_2 \ell + \epsilon$
3. **Total effect $c$**: $Q_i = \beta_0 + c \cdot \log(L_0) + \gamma_1 \log m + \gamma_2 \ell + \epsilon$
4. **Indirect effect**: $ab = c - c'$

Statistical significance of the indirect effect is assessed via the Sobel test and 10,000-resample bootstrap confidence intervals. Full mediation is declared when all four Baron-Kenny steps are met and $c'$ is non-significant; partial mediation when $c'$ is reduced but remains significant.

### 3.1.5 Step 4: Rosenbaum Sensitivity Analysis

High-absorption ($R_{\text{abs}} > 0.3$) and low-absorption ($R_{\text{abs}} < 0.1$) SAEs are matched on width (exact) and $L_0$ (nearest-neighbor Mahalanobis distance matching). For each matched pair, quality differences are tested via the Wilcoxon signed-rank test. The Rosenbaum sensitivity parameter $\Gamma$ is computed -- the odds ratio of a hypothetical hidden confounder at which the matched-pair result becomes non-significant. $\Gamma > 1.5$ indicates moderate robustness; $\Gamma > 2.0$ indicates strong robustness to unmeasured confounders.

Two matching strategies are employed: propensity score matching (6 pairs) and Mahalanobis matching (17 pairs). Within-width matching strategies (median split: 23 pairs; tertile: 16 pairs) serve as additional controls.

### 3.1.6 Step 5: SCR Suppression Variable Diagnosis

To identify which covariate produces the suppression effect on $\text{SCR}$, covariates are added sequentially: width-only, layer-only, architecture-only, $L_0$-only. The partial correlation between absorption and $\text{SCR}$ is tracked at each step, isolating which variable generates the change from the bivariate $r = -0.431$ to the full partial $r$.

## 3.2 Cross-Domain Absorption Measurement

### 3.2.1 Model and SAE

Cross-domain experiments use GPT-2 Small (124M parameters; Radford et al., 2019) with the gpt2-small-res-jb SAE (24,576 latents), accessed via SAELens (Bloom et al., 2024) and TransformerLens (Nanda et al., 2022). Gemma 2 2B was the intended target, but HuggingFace access restrictions at the time of experimentation necessitated the GPT-2 Small fallback. This limits generalizability (Section 5.4) but provides the cleanest available open-model anchor for cross-domain measurement.

### 3.2.2 Probe Training

Logistic regression probes are trained on residual stream activations at `hook_resid_pre` (matching the SAE input hook point) for five attribute types across three layers (5, 8, 11):

| Probe Type | Granularity | Classes | Training Samples |
|:-----------|:------------|:--------|:-----------------|
| Country (US) | Binary | US / non-US | 3,552 cities |
| Language (English) | Binary | English / non-English | 3,552 cities |
| Continent | 6-class | Africa, Asia, Europe, N. America, Oceania, S. America | 3,552 cities |
| Country top-10 | 10-class | 10 most frequent countries | 1,523 cities |
| Language top-10 | 10-class | 10 most frequent languages | 2,297 cities |

Probes are trained with scikit-learn `LogisticRegression(max_iter=1000, C=1.0)` using 80/20 stratified train/test splits, averaged over 3 random seeds (42, 123, 456). Prompt template: `"The city of {City} is located in"` with activations extracted at the last token position. A quality gate of 85% accuracy is applied to binary probes; multiclass probes are included with explicit accuracy reporting.

### 3.2.3 Absorption Measurement (Adapted Chanin Metric)

The absorption measurement adapts the Chanin et al. (2024) protocol to knowledge hierarchies via four steps:

1. **$k$-sparse probing**: For each attribute value (e.g., "France", "English"), identify $k$ split latents -- SAE latents with selectivity $\geq 3.0$ (activation rate $\geq$ 5% on the target class and $\geq 3\times$ higher than on other classes).

2. **False-negative identification**: Find tokens in the set $\text{FN}(f)$ -- inputs where all $k$ split latents for feature $f$ fail to activate but the linear probe correctly classifies the token.

3. **Dominance-based detection**: On false-negative tokens, the SAE latent with the highest activation magnitude is identified. A token is classified as absorbed if this dominant latent's activation exceeds the second-highest by a dominance ratio $\tau_{\text{dom}} \geq 1.0$.

4. **Cosine-calibrated detection**: As an alternative, absorption is declared only when the dominant latent's decoder direction $\mathbf{d}_j$ has $\text{cos}(\mathbf{p}, \mathbf{d}_j) > \tau_{\text{cos}}$ with the probe direction $\mathbf{p}$, requiring the absorbing latent to align with the target concept direction.

The absorption rate $R_{\text{abs}}$ is reported for each (layer, probe type) combination.

### 3.2.4 Controls

Three control conditions validate the absorption metric:

- **Shuffled hierarchy**: City-attribute mappings are randomized (5 trials per probe per layer), destroying real hierarchical structure. If the metric captures genuine absorption, shuffled rates should be near 0%.
- **Random probe direction**: Random unit vectors replace trained probe directions (5 trials per layer). Rates should be near 0% if the metric is probe-direction-specific.
- **First-letter baseline**: The standard Chanin et al. first-letter spelling absorption measurement on the same SAE provides a within-model reference point (literature range: 15--35% on Gemma Scope).

### 3.2.5 Threshold Sweep

Absorption rates are computed across a grid of detection thresholds: cosine similarity $\tau_{\text{cos}} \in \{0.05, 0.10, 0.15, 0.20, 0.30\}$ and dominance ratio $\tau_{\text{dom}} \in \{1.0, 2.0\}$. This sweep characterizes metric sensitivity and identifies threshold-dependent artifacts.

## 3.3 Absorption Scaling Surface

### 3.3.1 Dataset

The scaling surface uses 420 SAEs from SAEBench spanning 9 release families of Gemma 2 2B SAEs. Dictionary widths range from 2,304 to 1,048,576 ($\log_2$ range: 11.2--20.0); $L_0$ ranges from 9.3 to 8,277 ($\log_2$ range: 3.2--13.0). Architecture breakdown: 360 standard (L1), 54 JumpReLU, 6 unknown. Three layers are represented (5, 12, 19), with 140 SAEs per layer.

### 3.3.2 GAM Fit and Model Comparison

Three regression models of increasing flexibility are compared:

**Linear model**:
$$R_{\text{abs}} = \beta_0 + \beta_1 \log_2 m + \beta_2 \log_2 L_0 + \beta_3 \ell + \epsilon$$

**Additive GAM**:
$$R_{\text{abs}} = s(\log_2 m) + s(\log_2 L_0) + \beta_3 \ell + \epsilon$$

**Interaction GAM**:
$$R_{\text{abs}} = s(\log_2 m) + s(\log_2 L_0) + \text{ti}(\log_2 m, \log_2 L_0) + \beta_3 \ell + \epsilon$$

where $s(\cdot)$ denotes smooth univariate spline terms and $\text{ti}(\cdot, \cdot)$ is a tensor interaction term that captures joint dependence of absorption on width and $L_0$ beyond their additive contributions. The significance of the interaction term (H3) is tested via its $p$-value in the interaction GAM. Model selection uses $R^2$ and AIC.

### 3.3.3 Phase Boundary Detection

The gradient magnitude of the fitted GAM surface is computed on a dense grid in $(\log_2 m, \log_2 L_0)$ space. Ridges -- regions of maximal gradient magnitude -- are identified as candidate phase boundaries where absorption transitions sharply between regimes. Ridge points are defined as grid cells where gradient magnitude exceeds the 70th percentile of the overall distribution.

Per-layer Spearman rank correlations between absorption and each of $\log_2 m$ and $\log_2 L_0$ characterize the marginal effects.

## 3.4 Taxonomy Correction

### 3.4.1 Motivation

The original Chanin et al. (2024) absorption taxonomy classifies 92.3% of letters as exhibiting some form of absorption on GPT-2 Small. The Type II classification (88.5% of letters) relies on a magnitude ratio comparing a parent feature's activation in letter-specific contexts versus comparison contexts. For letters with zero comparison tokens (n_comparison_tokens = 0), the original implementation falls back to a global mean-when-active baseline, potentially inflating the Type II rate.

### 3.4.2 Frequency-Matched Correction

For each of the 26 letters, we apply a three-strategy comparison baseline:

- **Strategy A (preferred)**: Non-letter-context activations of the same parent feature on WikiText-103 (7,991 sentences). For a parent feature identified by the selectivity heuristic, we compare its mean activation magnitude on tokens starting with the target letter versus tokens not starting with that letter.
- **Strategy B (fallback)**: Global when-active baseline from the WikiText-103 corpus when Strategy A yields insufficient data.
- **Strategy C (ground truth)**: The Chanin absorption rate (fraction of false-negative tokens per letter) provides a direct measure of absorption independent of the magnitude-ratio threshold.

The Type II classification threshold remains at magnitude ratio $< 0.5$. The corrected combined rate is computed with BCa bootstrap 95% confidence intervals (10,000 resamples).

## 3.5 Pre-Registered Hypotheses and Falsification Criteria

**H1** (Absorption-Quality Causal Chain): After controlling for $\log(L_0)$, the partial correlation between absorption and at least one quality metric retains $|r_{\text{partial}}| > 0.2$ ($p < 0.05$). Absorption mediates $> 30\%$ of $L_0$'s total effect on at least one downstream quality metric (bootstrap CI excluding 0).

*Falsified if*: All four partial correlations drop below $|0.2|$, all width-stratified correlations are non-significant, mediation indirect effect bootstrap CIs include 0, and $\Gamma < 1.2$.

**H2** (Cross-Domain Absorption Existence): Absorption rate on knowledge hierarchies exceeds 10% and exceeds $3\times$ the shuffled-hierarchy baseline for at least one domain-layer combination.

*Falsified if*: No hierarchy shows absorption exceeding $3\times$ the shuffled baseline after the probe quality gate.

**H3** (Absorption Scaling Surface): The tensor interaction term $\text{ti}(\log_2 m, \log_2 L_0)$ in the GAM is significant ($p < 0.05$), indicating that absorption depends on the joint structure of width and $L_0$.

*Falsified if*: The interaction term is non-significant ($p > 0.10$) and the additive GAM explains absorption as well as the interaction GAM.

## 3.6 Software and Reproducibility

All analyses use Python 3.12 with numpy, scipy, pandas, statsmodels, pingouin (partial correlations), scikit-learn (probes and matching), pyGAM (generalized additive models), and matplotlib/seaborn (visualization). Model access uses TransformerLens $\geq$ 2.0 and SAELens $\geq$ 4.0. Random seeds are fixed at 42 (primary), 123, and 456 (replication seeds). All threshold sweeps use pre-registered values. SAE identifiers and HuggingFace release names are logged for full reproducibility.

> Table 1 (Section 4.1) reports per-metric partial correlations before and after $L_0$ control. Figure 1 (Section 4.1) visualizes the bar chart comparison. Figure 3 (Section 4.2) shows cross-domain absorption rates alongside the shuffled control line.

<!-- FIGURES
- None
-->
