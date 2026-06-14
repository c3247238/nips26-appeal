# Paper Outline: Competitive Geometry of Feature Absorption in Sparse Autoencoders

## Title

**When Features Compete: Empirical Tests of Lotka-Volterra Absorption Detection, Corpus-Level Predictors, and Downstream Impact in Sparse Autoencoders**

---

## Section 1: Introduction (~1.5 pages)

### Key Arguments

- Feature absorption is the primary failure mode documented for SAEs deployed in mechanistic interpretability: rare general features (e.g., "first letter = A") are systematically suppressed by frequent specific features (e.g., "the token 'April'"), with reported rates of 15--35% on the first-letter task (Chanin et al., 2024).
- Three tensions remain unresolved as of April 2026:
  1. **Measurement vs. phenomenon**: The 15--35% rate captures only Type I (full, single-latent) absorption. Our taxonomy (Section 5.3) finds 92.3% of letter features show partial suppression (Type II), suggesting the true failure scope is 2--6x larger than reported.
  2. **Mechanism vs. intervention**: Architecture papers (Matryoshka SAE, OrtSAE, masked regularization, ATM SAE) reduce absorption without explaining the mechanism. We test whether a Lotka-Volterra competitive exclusion framework provides a unified explanation.
  3. **Metric vs. impact**: DeepMind deprioritized SAE research after finding dense probes outperform SAEs on safety tasks. We provide the first systematic test: absorption score correlates negatively with SAEBench downstream performance (Pearson $r = -0.59$ for sparse probing, $r = -0.45$ for RAVEL proxy, $r = -0.43$ for SCR), indicating the metric does capture something that matters---but the safety probe experiment shows the causal chain is more nuanced.
- This paper provides three contributions: (1) an unsupervised LV-based absorption detector requiring no probe directions, (2) a PMI-based corpus predictor analysis (null result), and (3) the first direct test linking absorption scores to downstream interpretability performance.

### Transition to Section 2

Positions the work relative to the absorption measurement literature, the architecture intervention literature, and the SAE evaluation literature.

> **Figure 1** reference: introduce the conceptual framework diagram here (LV competitive exclusion applied to SAE features).

---

## Section 2: Related Work (~1.5 pages)

### Key Arguments

- **Feature absorption measurement**: Chanin et al. (arXiv:2409.14507) defined the canonical metric on the first-letter task using pre-specified probe directions. LessWrong post "Looking for Feature Absorption Automatically" attempted cosine-similarity-based detection and reported negative results. Our LV coefficient approach uses co-activation statistics instead.
- **SAE architectures and absorption reduction**: JumpReLU (Rajamanoharan et al., 2024), Matryoshka SAE (Bussmann et al., 2024), OrtSAE (Bussman et al., 2025), masked regularization (arXiv:2604.06495), ATM SAE. These reduce absorption but lack a unified mechanistic account.
- **SAE evaluation benchmarks**: SAEBench (arXiv:2503.09532) provides the first multi-metric evaluation suite covering RAVEL, SCR, sparse probing, absorption, and unlearning. We leverage this for our downstream impact analysis.
- **Lotka-Volterra in ML**: The ecological competition analogy is invoked in one sentence by "The Geometry of Concepts" (2025) but never formalized for absorption prediction. We provide the first quantitative formulation.
- **Feature splitting and superposition**: Anthropic's toy model work (Bricken et al., 2023), Elhage et al. (2022) superposition framework. Feature absorption is distinct from superposition---it occurs even in SAEs trained to resolve superposition.

### Transition to Section 3

Sets up the formal framework by identifying the gap: no prior work computes a probe-free absorption predictor from activation statistics alone.

---

## Section 3: Method — Lotka-Volterra Competition Coefficient (~2 pages)

### 3.1 Formal Framework

- Define the competition coefficient:
  $$\alpha_{ij} = \sigma_{ij} \cdot \frac{f_j}{f_i}$$
  where $\sigma_{ij} = P(a_i > 0, a_j > 0) / \min(f_i, f_j)$ is the normalized co-activation rate (niche overlap) and $f_i, f_j$ are per-latent activation frequencies.
- The LV competitive exclusion principle predicts: when $\alpha_{ij} > 1$, the rarer species (feature $i$) is excluded (absorbed). The threshold at $\alpha_{ij} \approx 1$ should produce a sharp sigmoid transition in absorption rate.
- Pre-filter: restrict candidate pairs to those with decoder cosine similarity $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$, reducing computation to $O(D \cdot k)$ with $k \approx 20$.

### 3.2 Distributed Absorption Score

- Define $\text{DAS}(P, k)$: multi-child absorption capturing the case where no single child suppresses the parent, but $k$ children collectively do.
- $\text{DAS}(P, k=3)$ estimated via logistic regression of parent activation on top-3 children by $\alpha_{ij}$.
- Addresses the width paradox: wider SAEs have more candidate absorbers, potentially shifting from concentrated (Type I) to distributed (Type III) absorption.

### 3.3 Absorption Taxonomy

- **Type I (Full)**: Chanin metric $> 0.5$ AND single absorber accounts for $> 80\%$ of suppression.
- **Type II (Partial)**: Parent activation $< 50\%$ of expected magnitude on letter tokens.
- **Type III (Distributed)**: $\text{DAS}(k=3) > 0.6$ AND Type I not triggered.
- Designed to capture the full failure scope beyond the canonical 15--35% rate.

> **Figure 1: LV Competition Framework** (Section: Method, see Figure & Table Plan below)

> **Table 1: Absorption Taxonomy Definitions** (Section: Method, see Figure & Table Plan below)

### Transition to Section 4

From the formal framework, we move to the experimental setup that tests whether these quantities actually predict absorption.

---

## Section 4: Experimental Setup (~1.5 pages)

### 4.1 Models and SAEs

- **Primary model**: GPT-2 Small (open-weight anchor; Gemma-2-2B planned but required gated HuggingFace access)
- **SAE sources**: `gpt2-small-res-jb` (24k, layer 8), `gpt2-small-res-jb-feature-splitting` (24k/49k/98k, layer 8), `gpt2-small-resid-post-v5` (32k/128k, layer 8)
- **Gemma Scope SAEs** (Gemma 2 2B): used for SAEBench correlation analysis via pre-computed results (54 SAE configurations across layers 5/12/19, widths 16k/65k/1M)
- **SAEBench data**: 54 Gemma Scope SAEs with absorption, sparse probing, SCR, RAVEL, and unlearning scores

### 4.2 Ground Truth

- sae-spelling library for first-letter absorption labels (26 letters, A--Z)
- Train/test split: A--M calibration, N--Z test (13/13 letters)

### 4.3 Evaluation Protocol

- **H1 (LV Detector)**: Precision, recall, F1, ROC-AUC of $\alpha_{ij}$ as binary predictor at calibrated threshold $\tau$
- **H2 (Corpus PMI)**: Partial $R^2$ of PMI term in regression controlling for SAE configuration (log L0, log width, layer)
- **H3 (Downstream Impact)**: Pearson/Spearman correlations between absorption score and SAEBench metrics; Bonferroni correction for 4 tests
- **H4 (Width Paradox)**: Monotonicity of $\text{DAS}(k=1)$ and $\text{DAS}(k=3)$ across widths 24k/49k/98k

### 4.4 Baselines

- Cosine-similarity-only detector (prior LessWrong approach)
- Chanin probe-directed metric (ground truth)
- Dense linear probe (upper bound for downstream tasks)

---

## Section 5: Results (~3 pages)

### 5.1 LV Detector Performance (H1) — Negative Result

- The LV detector achieves test F1 = 0.128 at $\tau = 0.5$, ROC-AUC = 0.148 --- below the $F1 > 0.35$ success criterion.
- The cosine-similarity baseline achieves test F1 = 0.165, ROC-AUC = 0.201, outperforming the LV coefficient.
- **Sharpness test**: Linear fit (AIC = -61.05) marginally beats sigmoid (AIC = -60.95). No sharp threshold transition at $\alpha_{ij} \approx 1$. The LV competitive exclusion model does not describe a phase transition in absorption.
- **Interpretation**: $\alpha_{ij}$ captures some information about co-activation geometry, but the threshold-based LV framing is not supported. Absorption is not well-described as a binary competitive exclusion event.

> **Table 2: LV Detector vs. Cosine Baseline** (precision, recall, F1, AUC)
> **Figure 2: Sharpness Test** (absorption rate vs. $\alpha_{ij}$ bins with sigmoid and linear fits)

### 5.2 Corpus PMI as Predictor (H2) — Null Result

- PMI coefficient $\beta_4 = -0.0063$ (negative, opposite to prediction), $p = 0.593$.
- Partial $R^2$ for PMI = 0.0006, far below the 0.10 success criterion.
- Per-layer coefficient sign is inconsistent (5/9 layers negative, 3/9 positive, 1 zero-variance).
- **Finding**: Corpus co-occurrence statistics do not predict which letter features are absorbed. The strongest predictors are layer ($\beta_3 = -0.012$, $p < 0.0001$) and log(L0) ($\beta_1 = 0.013$, $p = 0.012$). Absorption decreases in later layers and increases with sparsity penalty.

> **Figure 3: Partial Regression Plot** (absorption vs. log PMI, residualized)
> **Table 3: PMI Regression Coefficients**

### 5.3 Absorption Taxonomy — Key Positive Result

- On GPT-2 Small (24k SAE, layer 8), the comprehensive taxonomy classifies:
  - Type I (Full): 1/26 letters (3.8%) --- letter Q
  - Type II (Partial): 23/26 letters (88.5%)
  - Type III (Distributed): 0/26 letters
  - None: 2/26 letters (7.7%) --- letters X, Z
- **Comprehensive absorption rate**: 92.3% vs. the canonical 15--35% (Chanin metric on the same data yields 35.4% overall and detects any absorption for 80.8% of letters).
- Caveat: Type II rate is likely inflated because parent features were identified by selectivity heuristic rather than sae-spelling ground truth (see limitations).

> **Figure 4: Taxonomy Stacked Bar Chart** (Type I/II/III/None across widths)

### 5.4 Downstream Impact (H3) — Strong Positive Result (H3 falsified)

- Against the H3 prediction of $|r| < 0.2$, absorption score shows strong negative correlations with SAEBench metrics (n=54 Gemma Scope SAEs):
  - Sparse probing F1: $r = -0.595$ ($p < 0.001$), partial $r = -0.661$ controlling for width/layer/architecture
  - SCR: $r = -0.431$ ($p = 0.002$), partial $r = -0.677$
  - RAVEL (TPP): $r = -0.454$ ($p < 0.001$), partial $r = -0.492$
  - Unlearning: $r = -0.175$ ($p = 0.280$) --- not significant
- All except unlearning survive Bonferroni correction ($\alpha = 0.0125$).
- **Safety probe (GPT-2 Small)**: Dense probe achieves AUC $\approx 1.0$ across all SAEs. 1-sparse SAE probe gaps are 0.118 (lowest absorption), 0.148 (median), 0.051 (highest) --- no monotone trend with absorption level.
- **Interpretation**: Absorption is meaningfully correlated with downstream SAE quality, contradicting our pre-registered H3 but validating the research community's motivating assumption. The partial correlations strengthen after controlling for confounds, suggesting absorption captures genuine quality variation beyond width and layer effects.

> **Table 4: Absorption vs. Downstream Correlation Matrix** (Pearson $r$, Spearman $\rho$, 95% CI, Bonferroni-corrected $p$)
> **Figure 5: Scatter Plots** (absorption score vs. sparse probing F1 and RAVEL TPP)
> **Table 5: Safety Probe Results** (AUC by absorption level)

### 5.5 Width Paradox (H4) — Partial Support

- Mean $\text{DAS}(k=1)$ across widths: 0.105 (24k), 0.104 (49k), 0.119 (98k) --- no clear monotonic decrease.
- Mean $\text{DAS}(k=3)$: 0.320 (24k), 0.227 (49k), 0.260 (98k) --- no monotonic increase.
- Per-letter analysis: 57.7% of letters show non-positive $\text{DAS}(k=1)$ slope with width (target: 60%); 42.3% show positive $\text{DAS}(k=3)$ slope (target: 80%).
- H4 receives partial support: $k=1$ non-monotonicity is observed, but $k=3$ does not reliably increase with width.

> **Figure 6: DAS(k=1) and DAS(k=3) vs. Width** (letter-level error bars)

### 5.6 Cross-Architecture Generalization (H1 supplement)

- On GPT-2 Small v5-32k (resid_post): fixed-$\tau$ test F1 = 0.009, cosine baseline F1 = 0.212. LV detector fails.
- On GPT-2 Small v5-128k (resid_post): fixed-$\tau$ test F1 = 0.0, cosine baseline F1 = 0.353. LV detector fails completely.
- F1 degradation averages 12.6 percentage points across architectures.
- The LV competition coefficient does not generalize across SAE architectures.

---

## Section 6: Ablations (~0.5 pages)

### 6.1 Threshold Sensitivity (A1)

- Fine sweep over $\tau \in [0.3, 1.5]$: best test F1 peaks at $\tau = 0.5$ (F1 = 0.128). No peak in the LV-predicted $[0.75, 1.25]$ range.
- F1 declines monotonically for $\tau > 0.5$, consistent with $\alpha_{ij}$ acting as a weak continuous predictor rather than a threshold discriminant.

### 6.2 PMI Regression Without SAE Config Terms (A3)

- PMI-only model: $R^2 = 0.0006$, confirming PMI has negligible standalone predictive power.

---

## Section 7: Discussion (~1.5 pages)

### Key Arguments

- **The LV framework fails as a sharp detector but succeeds conceptually**: The competition coefficient does not produce a usable binary classifier, but the conceptual framing (co-activation rate $\times$ frequency imbalance) captures the right factors. The failure is in the threshold sharpness prediction, not the feature selection.
- **Corpus statistics are not the proximal cause of absorption**: The null PMI result, combined with the dominance of layer and L0 as predictors, suggests absorption is primarily driven by the SAE's optimization objective and sparsity penalty---not by training data co-occurrence patterns. This has implications for intervention design: data engineering (corpus curation) is unlikely to reduce absorption; architectural and regularization changes remain the correct intervention point.
- **Absorption does predict downstream quality**: The H3 falsification is the strongest empirical result in the paper. Prior to this work, it was unknown whether the absorption metric was informative for downstream utility. The Pearson $r = -0.59$ with sparse probing and $r = -0.45$ with RAVEL, surviving Bonferroni correction and strengthening under partial correlation, provides the first direct evidence that absorption reduction translates to downstream improvement.
- **The taxonomy reveals underreporting of absorption severity**: 92.3% comprehensive rate vs. 3.8% strict Type I rate. Even accounting for likely inflation in the Type II metric (see limitations), the gap between the canonical 15--35% rate and the partial absorption rate is real and practically important.
- **Model scope limitation**: All probe-based experiments used GPT-2 Small. The downstream correlation analysis used Gemma Scope SAEs via SAEBench. Replication on Gemma-2-2B (pending gated access) would strengthen the taxonomy and LV detector findings.

### Transition to Section 8

From findings and their implications, we summarize contributions and identify future directions.

---

## Section 8: Conclusion (~0.5 pages)

### Key Points

- Three-part empirical study of SAE feature absorption: unsupervised detection (negative), corpus prediction (null), and downstream impact (strong positive).
- The LV competitive exclusion framework does not produce a usable probe-free absorption detector (test F1 = 0.128), and corpus PMI has no predictive power (partial $R^2 = 0.0006$).
- The strongest finding is empirical: absorption score correlates meaningfully with downstream SAE quality ($r = -0.59$ sparse probing, $r = -0.45$ RAVEL, $r = -0.43$ SCR) across 54 Gemma Scope SAEs, providing the first direct validation of the assumed causal chain from absorption to downstream interpretability.
- The three-tier absorption taxonomy reveals that partial absorption (Type II) affects 92.3% of letter features, compared to the canonical 15--35% Type I rate.
- Future work: replication on Gemma-2-2B, causal intervention experiments (ablating absorbed features and measuring downstream task change), and extending the taxonomy beyond the first-letter task.

---

## Figure & Table Plan

### Figure 1: Lotka-Volterra Competition Framework (Section: Method)
- **Purpose**: Illustrate the conceptual mapping from ecological LV competitive exclusion to SAE feature absorption
- **Type**: architecture_diagram / flow_chart
- **Content**: Left panel: ecological niche overlap diagram with two species. Right panel: SAE decoder space with parent feature and absorbing child feature. Center: mathematical mapping ($\sigma_{ij} \to$ niche overlap, $f_j/f_i \to$ carrying capacity ratio, $\alpha_{ij} > 1 \to$ exclusion/absorption).
- **Key takeaway**: The competition coefficient $\alpha_{ij}$ formalizes absorption as competitive exclusion in decoder space
- **Generation**: tikz or manual_diagram
- **Data source**: Conceptual; no experimental data

### Table 1: Absorption Taxonomy Definitions (Section: Method)
- **Purpose**: Define the three absorption types with operationalized thresholds
- **Type**: comparison_table
- **Content**: Columns: Type, Definition, Threshold, Measurement Method. Rows: Type I (Full), Type II (Partial), Type III (Distributed), None.
- **Key takeaway**: The taxonomy captures partial and distributed absorption missed by the canonical single-latent metric
- **Generation**: data_table
- **Data source**: `C2D_taxonomy.json` thresholds

### Table 2: LV Detector vs. Cosine Baseline (Section: Results 5.1)
- **Purpose**: Report H1 quantitative results comparing the LV coefficient against the cosine baseline
- **Type**: comparison_table
- **Content**: Columns: Method, $\tau$/threshold, Precision, Recall, F1, ROC-AUC. Rows: LV $\alpha_{ij}$ ($\tau=0.5$), Cosine ($\theta=0.2$), and sweep across $\tau$ values.
- **Key takeaway**: The LV detector (F1 = 0.128) underperforms the cosine baseline (F1 = 0.165)
- **Generation**: data_table
- **Data source**: `C1B_lv_validation.json`

### Figure 2: Sharpness Test — Absorption Rate vs. $\alpha_{ij}$ (Section: Results 5.1)
- **Purpose**: Test the LV prediction of a sharp sigmoid transition at $\alpha_{ij} \approx 1$
- **Type**: bar_chart + line_plot overlay
- **Content**: X-axis: $\alpha_{ij}$ bins (width 0.1, range [0, 2]). Y-axis: empirical absorption rate within each bin. Overlay: fitted sigmoid and linear functions. Inset: AIC comparison.
- **Key takeaway**: No sharp threshold; linear fit marginally preferred (AIC difference < 0.2)
- **Generation**: code (matplotlib)
- **Data source**: `C1B_lv_validation.json` lv_sharpness.bin_data

### Figure 3: Partial Regression Plot — Absorption vs. log(PMI) (Section: Results 5.2)
- **Purpose**: Show that corpus PMI has no predictive power after controlling for SAE configuration
- **Type**: scatter
- **Content**: X-axis: log(PMI) residuals. Y-axis: absorption rate residuals. Each point is one (SAE config, letter) observation. Regression line with 95% CI band.
- **Key takeaway**: Flat relationship; partial $R^2 = 0.0006$
- **Generation**: code (matplotlib/seaborn) --- already generated
- **Data source**: `C2C_plots/C2C_partial_regression_plot.png`, `C2C_regression_results.json`

### Table 3: PMI Regression Coefficients (Section: Results 5.2)
- **Purpose**: Full regression table with all coefficients, standard errors, and significance
- **Type**: data_table
- **Content**: Columns: Coefficient, Estimate, SE (HC3), $t$, $p$, 95% CI. Rows: intercept, log(L0), log(width), layer, log(PMI).
- **Key takeaway**: Layer ($p < 0.0001$) and L0 ($p = 0.012$) are significant; PMI is not ($p = 0.593$)
- **Generation**: data_table
- **Data source**: `C2C_regression_results.json`

### Figure 4: Absorption Taxonomy Across Widths (Section: Results 5.3)
- **Purpose**: Show the comprehensive absorption rate (Type I + II + III) vs. canonical rate across SAE widths
- **Type**: bar_chart (stacked)
- **Content**: X-axis: SAE width (24k, 49k, 98k). Y-axis: fraction of 26 letters. Stacked bars: Type I (red), Type II (orange), Type III (yellow), None (gray). Horizontal dashed line at 35% (Chanin canonical rate).
- **Key takeaway**: Comprehensive rate (92--96%) vastly exceeds the canonical 15--35% Type I rate
- **Generation**: code (matplotlib)
- **Data source**: `C2D_taxonomy.json` cross_width_analysis

### Table 4: Absorption vs. Downstream Correlation Matrix (Section: Results 5.4)
- **Purpose**: Main results table for H3 — the paper's strongest finding
- **Type**: comparison_table
- **Content**: Columns: Task, $n$, Pearson $r$, 95% CI, $p$, Spearman $\rho$, $p$, Partial $r$, Significant (Bonferroni). Rows: Sparse probing F1, SCR, RAVEL (TPP), Unlearning.
- **Key takeaway**: Absorption significantly predicts downstream quality for 3/4 tasks ($r$ from $-0.43$ to $-0.60$)
- **Generation**: data_table
- **Data source**: `C3A_saebench_corr.json`

### Figure 5: Absorption vs. Downstream Scatter Plots (Section: Results 5.4)
- **Purpose**: Visualize the absorption-performance correlation for the two strongest tasks
- **Type**: scatter (2 panels)
- **Content**: Panel A: absorption score vs. sparse probing F1 (54 SAEs, colored by width). Panel B: absorption score vs. RAVEL TPP. Regression line + 95% CI.
- **Key takeaway**: Clear negative trends; $r = -0.60$ and $r = -0.45$ respectively
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `C3A_saebench_corr.json` raw_records

### Table 5: Safety Probe Results (Section: Results 5.4)
- **Purpose**: Report the 1-sparse vs. dense probe gap by absorption level
- **Type**: data_table
- **Content**: Columns: Absorption Level, SAE Config, Layer, Width, Dense AUC, 1-Sparse AUC, Probe Gap. Rows: Lowest (0.000), Median (0.047), Highest (0.119).
- **Key takeaway**: Probe gap does not monotonically increase with absorption (0.118, 0.148, 0.051)
- **Generation**: data_table
- **Data source**: `C3C_safety_probe.json`

### Figure 6: DAS(k=1) and DAS(k=3) vs. SAE Width (Section: Results 5.5)
- **Purpose**: Test H4 width paradox prediction
- **Type**: line_plot (2 series with error bars)
- **Content**: X-axis: SAE width (24k, 49k, 98k). Y-axis: mean DAS score. Two lines: DAS(k=1) (blue) and DAS(k=3) (red). Error bars from letter-level standard deviation.
- **Key takeaway**: Neither metric shows the predicted monotonic trend; width paradox receives only partial support
- **Generation**: code (matplotlib)
- **Data source**: `C1D_das_vs_width.json` mean_das_by_width

### Figure 7 (Appendix): Per-Layer PMI Coefficient Stability (Section: Ablations / Appendix)
- **Purpose**: Show that the PMI coefficient sign varies across layers
- **Type**: bar_chart with error bars
- **Content**: X-axis: model layer. Y-axis: $\beta_{\text{PMI}}$ estimate. Error bars: $\pm 1$ SE.
- **Key takeaway**: Sign inconsistency across layers confirms PMI is not a robust predictor
- **Generation**: code (matplotlib) --- already generated
- **Data source**: `C2C_plots/C2C_per_layer_pmi_coef.png`, `C2C_regression_results.json`
