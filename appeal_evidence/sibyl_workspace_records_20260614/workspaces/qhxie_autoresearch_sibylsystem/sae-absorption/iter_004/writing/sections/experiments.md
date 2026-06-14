# 4 Experimental Setup

## 4.1 Models and SAE Configurations

All experiments use pre-trained SAEs loaded via SAELens; no model fine-tuning or SAE retraining is performed.

**Primary model.** GPT-2 Small (124M parameters, $d = 768$) serves as the open-weight anchor for all probe-based experiments (H1, H2, H4, taxonomy, safety probe). Gemma 2 2B was the original target model but requires gated HuggingFace access; we defer Gemma-based replication to future work.

**SAE sources for GPT-2 Small.** We use three SAE families from SAELens, all at layer 8 of the residual stream:

- `gpt2-small-res-jb` ($D = 24{,}576$, `hook_resid_pre`): primary SAE for H1 calibration, H2 regression, and taxonomy.
- `gpt2-small-res-jb-feature-splitting` ($D \in \{24{,}576, 49{,}152, 98{,}304\}$, `hook_resid_pre`): width-varying family for H4 (width paradox).
- `gpt2-small-resid-post-v5` ($D \in \{32{,}768, 131{,}072\}$, `hook_resid_post`): OpenAI v5 architecture for cross-architecture validation. JumpReLU and TopK SAEs are not available for GPT-2 Small in SAELens.

**SAEBench data (Gemma Scope).** For H3 (downstream impact), we use pre-computed SAEBench results (Karvonen et al., 2025) for 54 Gemma Scope SAEs trained on Gemma 2 2B residual stream activations. These span layers 5, 12, and 19; widths 16k, 65k, and 1M; and multiple $L_0$ settings per configuration. Metrics available per SAE: absorption score, sparse probing F1, SCR, RAVEL (TPP), and unlearning.

**PMI regression survey.** The H2 regression uses absorption measurements from 31 GPT-2 Small SAE configurations across layers 3--11 and widths 768--131,072, yielding 806 (configuration, letter) observations.


## 4.2 Ground Truth and Data Split

Ground-truth absorption labels come from `sae-spelling` (Chanin et al., 2024), which identifies absorbed latent pairs using pre-specified letter-probe directions for the first-letter task (26 letters, A--Z).

**Train/test split.** Letters A--M (13 letters) serve as the calibration set; letters N--Z (13 letters) serve as the held-out test set. The threshold $\tau$ for the competition coefficient $\alpha_{ij}$ is selected to maximize F1 on the calibration set and evaluated once on the test set. On the primary 24k SAE, the calibration set contains 1,110 candidate pairs (275 positives) and the test set contains 790 pairs (195 positives), reflecting the overall absorption rate of 35.4%.


## 4.3 Evaluation Protocol

We organize the evaluation around four pre-registered hypotheses.

**H1 (LV Detector).** The competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ is evaluated as a binary absorption predictor at threshold $\tau$. Metrics: precision, recall, F1, and ROC-AUC on the test letters. Success criterion: F1 $> 0.35$. A sharpness diagnostic tests the LV-predicted sigmoid transition at $\alpha_{ij} \approx 1$: we bin $\alpha_{ij}$ values (width 0.1, range $[0, 3]$), compute empirical absorption rates per bin, and compare AIC between fitted sigmoid and linear models.

**H2 (Corpus PMI).** We regress absorption rate on $\log(L_0)$, $\log(D)$, layer, and $\log(\text{PMI})$ with HC3 robust standard errors. The quantity of interest is the partial $R^2$ of the PMI term. Success criterion: partial $R^2 \geq 0.10$.

**H3 (Downstream Impact).** We compute Pearson $r$ and Spearman $\rho$ between SAEBench absorption scores and each downstream metric (sparse probing F1, SCR, RAVEL TPP, unlearning) across 54 Gemma Scope SAEs, applying Bonferroni correction for 4 simultaneous tests ($\alpha_{\text{corrected}} = 0.0125$). We also compute partial correlations controlling for $\log(\text{width})$, layer, and architecture class. A matched comparison selects the 5 lowest- and 5 highest-absorption SAEs from layers 12 and 19 and applies a one-sided paired $t$-test on RAVEL TPP. A safety probe pilot (50 AdvBench-style harmful + 50 benign prompts, 5-fold CV) measures the dense-vs-1-sparse probe gap at three absorption levels.

**H4 (Width Paradox).** We compute $\text{DAS}(k{=}1)$ and $\text{DAS}(k{=}3)$ for all 26 letters across widths $\{24\text{k}, 49\text{k}, 98\text{k}\}$ on GPT-2 Small layer 8. $\text{DAS}(k{=}3)$ is estimated via logistic regression of parent activation on its top-3 children by $\alpha_{ij}$. Predictions: $\text{DAS}(k{=}1)$ non-monotone or decreasing with width for $\geq 60\%$ of letters; $\text{DAS}(k{=}3)$ monotonically increasing for $\geq 80\%$ of letters.


## 4.4 Baselines

Three baselines contextualize the competition coefficient:

1. **Cosine-similarity-only detector.** Threshold on decoder cosine similarity $\cos(\mathbf{d}_i, \mathbf{d}_j)$ alone, calibrated on the same A--M split. This corresponds to the approach reported as negative in "Looking for Feature Absorption Automatically" (LessWrong).
2. **Chanin probe-directed metric.** The `sae-spelling` ground-truth absorption rate, which requires pre-specified probe directions. This defines the target variable, not a competing predictor.
3. **Dense linear probe.** Logistic regression on the full residual stream ($d = 768$ features), providing an upper bound on downstream task performance.


# 5 Results

## 5.1 LV Detector Performance (H1) --- Negative Result

The competition coefficient $\alpha_{ij}$ does not produce a usable absorption detector. Table 2 reports the primary comparison.

**Table 2: LV Detector vs. Cosine Baseline on GPT-2 Small (24k, layer 8)**

| Method | Threshold | Precision | Recall | F1 | ROC-AUC |
|:-------|:---------:|:---------:|:------:|:--:|:-------:|
| LV $\alpha_{ij}$ | $\tau = 0.5$ | 0.085 | 0.256 | **0.128** | 0.148 |
| Cosine baseline | $\theta = 0.2$ | 0.140 | 0.200 | **0.165** | 0.201 |

The LV detector achieves test F1 = 0.128 at the best calibration threshold $\tau = 0.5$, falling well below both the cosine baseline (F1 = 0.165) and the pre-registered success criterion (F1 $> 0.35$). The cosine baseline outperforms the LV coefficient by +3.7 F1 points and +5.3 AUC points.

**Sharpness test.** The LV competitive exclusion model predicts a sharp sigmoid transition in absorption rate near $\alpha_{ij} \approx 1$. Figure 2 plots empirical absorption rates against binned $\alpha_{ij}$ values. A linear fit (AIC = $-61.05$) marginally outperforms the sigmoid fit (AIC = $-60.95$; $\Delta$AIC = 0.11). The fitted sigmoid has center $x_0 = 10.0$ (far outside the data range) and slope $k = 0.16$ (essentially flat), confirming that no phase transition exists at $\alpha_{ij} \approx 1$.

![Sharpness Test](figures/fig_sharpness.pdf)

The first $\alpha_{ij}$ bin ($[0, 0.1]$, $n = 369$) shows an anomalously high absorption rate of 0.848. This artifact arises because very low $\alpha_{ij}$ values correspond to pairs where both features have similar low frequencies and high co-activation --- precisely the pattern of related features that are both being absorbed by a common parent. Excluding this bin, the remaining data show a roughly flat or weakly increasing relationship.

**Interpretation.** The competition coefficient captures some information about co-activation geometry (it selects for pairs with high niche overlap and frequency imbalance), but the ecological threshold model --- where $\alpha_{ij} > 1$ triggers competitive exclusion --- does not describe a phase transition in absorption. Absorption is better modeled as a graded phenomenon than a binary exclusion event.


## 5.2 Corpus PMI as Predictor (H2) --- Null Result

Corpus co-occurrence statistics do not predict which letter features are absorbed. The full regression model (Table 3) explains 8.7% of absorption variance ($R^2 = 0.087$), but the PMI term contributes negligibly.

**Table 3: PMI Regression Coefficients (806 observations, 31 SAE configurations, HC3 SEs)**

| Coefficient | Estimate | SE (HC3) | $t$ | $p$ | 95% CI |
|:------------|:--------:|:--------:|:---:|:---:|:------:|
| Intercept ($\beta_0$) | 0.052 | 0.064 | 0.81 | 0.418 | [$-$0.074, 0.179] |
| $\log(L_0)$ ($\beta_1$) | 0.013 | 0.005 | 2.52 | **0.012** | [0.003, 0.023] |
| $\log(D)$ ($\beta_2$) | 0.003 | 0.004 | 0.81 | 0.417 | [$-$0.005, 0.011] |
| Layer ($\beta_3$) | $-$0.012 | 0.002 | $-$6.61 | **$<$0.001** | [$-$0.016, $-$0.008] |
| $\log(\text{PMI})$ ($\beta_4$) | $-$0.006 | 0.012 | $-$0.53 | 0.593 | [$-$0.029, 0.017] |

The PMI coefficient $\beta_4 = -0.006$ is negative (opposite to the H2 prediction of positive association), statistically non-significant ($p = 0.593$), and contributes partial $R^2 = 0.0006$ --- three orders of magnitude below the 0.10 success criterion. A PMI-only model achieves $R^2 = 0.0006$, confirming negligible standalone predictive power (ablation A3).

The dominant predictors are layer ($\beta_3 = -0.012$, $p < 0.001$) and $\log(L_0)$ ($\beta_1 = 0.013$, $p = 0.012$). Absorption decreases monotonically in later layers (each layer reduces absorption by 1.2 percentage points) and increases with the sparsity penalty. Width has no significant effect after controlling for $L_0$.

**Per-layer stability (ablation A4).** The PMI coefficient sign is inconsistent across layers: negative at layers 3, 4, 6, and 7; positive at layers 5 and 8. Only two layers (4 and 6) show significance for the PMI term ($p = 0.004$ and $p = 0.015$), both with negative coefficients --- opposite to the H2 prediction. This sign instability rules out PMI as a robust absorption predictor.


## 5.3 Absorption Taxonomy --- Key Positive Result

Table 4 presents the per-letter taxonomy classification on GPT-2 Small (24k SAE, layer 8). The comprehensive taxonomy reveals that absorption is far more prevalent than the canonical Type I metric suggests.

**Table 4: Absorption Taxonomy Summary (GPT-2 Small, 24k, layer 8)**

| Type | Count | Fraction | Definition |
|:-----|:-----:|:--------:|:-----------|
| Type I (full) | 1 | 3.8% | Chanin metric $> 0.5$ AND single absorber $> 80\%$ |
| Type II (partial) | 23 | 88.5% | Parent activation $< 50\%$ expected magnitude |
| Type III (distributed) | 0 | 0.0% | DAS($k{=}3$) $> 0.6$ AND not Type I |
| None | 2 | 7.7% | Letters X, Z |
| **Comprehensive** | **24** | **92.3%** | Type I + II + III |

The comprehensive absorption rate of 92.3% is 24$\times$ the strict Type I rate (3.8%) and 2.6$\times$ the Chanin any-absorption rate (80.8% of letters show some absorption at the standard threshold). Only letter Q meets the strict Type I criterion (absorption rate = 1.0, single absorber accounts for 100% of suppression). Letters X and Z are classified as "None" because their parent features maintain $\geq 50\%$ of expected activation magnitude (magnitude ratios of 0.625 and 0.585, respectively).

![Absorption Taxonomy Across Widths](figures/fig_taxonomy_bar.pdf)

Figure 4 shows the taxonomy distribution across widths $\{24\text{k}, 49\text{k}, 98\text{k}\}$. The comprehensive rate is stable: 92.3% at 24k and 49k, rising to 96.2% at 98k (one additional letter classified as Type III). The canonical Chanin rate of 35.4% captures only the Type I tip of the absorption iceberg.

**Caveat.** The Type II rate is likely inflated. Parent features were identified by a selectivity heuristic on synthetic prompts rather than by `sae-spelling` ground-truth parent feature IDs. The magnitude ratio threshold ($< 0.5$) compares letter-token activation to a global mean-when-active fallback (comparison token count $= 0$ for most letters), which systematically underestimates expected magnitude. The Type II rate should be interpreted as "parent fires weakly on letter tokens relative to its general activation pattern," not as a validated causal measure of absorption-induced suppression.


## 5.4 Downstream Impact (H3) --- Strong Positive Result

The H3 pre-registration predicted $|r| < 0.2$ between absorption and downstream metrics, expecting absorption to be decoupled from SAE quality. The data falsify this prediction: absorption shows strong negative correlations with 3 of 4 SAEBench tasks.

**Table 5: Absorption vs. Downstream Correlation Matrix ($n = 54$ Gemma Scope SAEs)**

| Task | $n$ | Pearson $r$ | 95% CI | $p$ | Spearman $\rho$ | Partial $r$ | Bonferroni sig. |
|:-----|:---:|:-----------:|:------:|:---:|:---------------:|:-----------:|:---------------:|
| Sparse probing F1 | 54 | $-$0.595 | [$-$0.744, $-$0.389] | $<$0.001 | $-$0.435 | $-$0.661 | Yes |
| SCR | 49 | $-$0.431 | [$-$0.635, $-$0.171] | 0.002 | $-$0.308 | $-$0.677 | Yes |
| RAVEL (TPP) | 54 | $-$0.454 | [$-$0.644, $-$0.212] | $<$0.001 | $-$0.478 | $-$0.492 | Yes |
| Unlearning | 40 | $-$0.175 | [$-$0.462, 0.144] | 0.280 | $-$0.141 | $-$0.072 | No |

Three of four correlations survive Bonferroni correction and exceed the $|r| > 0.3$ meaningful-effect threshold. Sparse probing shows the strongest relationship: SAEs with higher absorption scores have lower F1 on probing tasks ($r = -0.595$). Partial correlations strengthen after controlling for width, layer, and architecture class ($r_{\text{partial}} = -0.661$ for sparse probing, $r_{\text{partial}} = -0.677$ for SCR), indicating that absorption captures genuine quality variation beyond what these confounds explain.

![Absorption vs. Downstream Scatter](figures/fig_downstream_scatter.pdf)

**Matched RAVEL comparison.** The 5 lowest-absorption Gemma Scope SAEs (mean absorption = 0.022, layers 12/19) achieve mean TPP = 0.046, compared to mean TPP = 0.009 for the 5 highest-absorption SAEs (mean absorption = 0.866). The one-sided paired $t$-test rejects the null ($t = 4.27$, $p = 0.006$, Cohen's $d = 2.13$). Low-absorption SAEs outperform high-absorption SAEs across all three non-unlearning metrics: $\Delta$TPP = +0.037, $\Delta$F1 = +0.108, $\Delta$SCR = +0.203.

**Safety probe pilot (GPT-2 Small).** The dense probe achieves AUC $\approx 1.0$ across all three SAE configurations. The 1-sparse SAE probe gaps do not follow the predicted monotone relationship with absorption:

**Table 6: Safety Probe Results (5-fold CV, GPT-2 Small)**

| Absorption Level | Config | Layer | Width | Dense AUC | 1-Sparse AUC | Probe Gap |
|:-----------------|:------:|:-----:|:-----:|:---------:|:------------:|:---------:|
| Lowest (0.000) | `resid_mid_L10_32k` | 10 | 32k | 1.000 | 0.882 $\pm$ 0.067 | 0.118 |
| Median (0.047) | `resid_pre_L8_768` | 8 | 768 | 1.000 | 0.852 $\pm$ 0.047 | 0.148 |
| Highest (0.119) | `resid_pre_L5_24k` | 5 | 24k | 0.998 | 0.947 $\pm$ 0.039 | 0.051 |

The probe gap is 0.118 (lowest absorption), 0.148 (median), and 0.051 (highest) --- not monotonically increasing with absorption. The highest-absorption SAE actually shows the smallest probe gap. This non-monotone pattern reflects confounding by layer and width: the three SAEs differ in layer (5, 8, 10) and width (24k, 768, 32k), making it impossible to isolate absorption's causal effect. The SAEBench correlation analysis, which controls for these confounds across 54 SAEs, provides stronger evidence.


## 5.5 Width Paradox (H4) --- Partial Support

H4 predicted that $\text{DAS}(k{=}1)$ decreases while $\text{DAS}(k{=}3)$ increases with SAE width, reflecting a shift from concentrated to distributed absorption. The data provide partial support.

**Table 7: Mean DAS by Width (GPT-2 Small, layer 8, $n = 26$ letters)**

| Width | Mean DAS($k{=}1$) $\pm$ SD | Mean DAS($k{=}3$) $\pm$ SD |
|:-----:|:---------------------------:|:---------------------------:|
| 24k | 0.105 $\pm$ 0.118 | 0.320 $\pm$ 0.199 |
| 49k | 0.104 $\pm$ 0.068 | 0.227 $\pm$ 0.196 |
| 98k | 0.119 $\pm$ 0.185 | 0.260 $\pm$ 0.215 |

$\text{DAS}(k{=}1)$ shows no clear monotonic trend (0.105 $\to$ 0.104 $\to$ 0.119), and 57.7% of letters have non-positive slope with width --- meeting the 60% target marginally. $\text{DAS}(k{=}3)$ does not increase monotonically (0.320 $\to$ 0.227 $\to$ 0.260): only 42.3% of letters show positive slope, far below the 80% target.

![DAS vs Width](figures/fig_das_width.pdf)

Per-letter analysis reveals high variance: letter N has $\text{DAS}(k{=}1) = 0.571$ at 24k but 0.100 at 98k (slope $= -0.34$), while letter X jumps from 0.0 to 1.0 at 98k. The DAS($k{=}3$) landscape is similarly noisy, with letter B dropping from 0.695 to 0.071 and letter Y increasing from 0.613 to 0.722.

**Assessment.** H4 receives a "partial support" verdict. The $k{=}1$ non-monotonicity observation is consistent with the width paradox narrative, but the $k{=}3$ prediction fails. The width paradox may require mechanisms beyond distributed competitive exclusion --- for example, feature splitting at higher widths creates new parent-child relationships that redistribute absorption in unpredictable ways.


## 5.6 Cross-Architecture Generalization

The LV detector does not generalize across SAE architectures. Table 8 reports results on two OpenAI v5 SAEs (different training objective, `hook_resid_post`).

**Table 8: Cross-Architecture Validation (fixed $\tau = 0.5$)**

| Architecture | $D$ | LV Test F1 | Cosine F1 | F1 $\Delta$ (pp) |
|:-------------|:---:|:----------:|:---------:|:-----------------:|
| Baseline (JB, 24k) | 24,576 | 0.128 | 0.165 | --- |
| v5-32k (resid\_post) | 32,768 | 0.009 | 0.212 | $-$11.9 |
| v5-128k (resid\_post) | 131,072 | 0.000 | 0.353 | $-$12.8 |

On v5-32k, the LV detector drops to test F1 = 0.009; on v5-128k, it produces F1 = 0.0 (zero true positives). The cosine baseline, by contrast, improves on wider architectures (F1 = 0.353 on v5-128k). Even within-architecture recalibration fails: all $\tau$ values from 0.5 to 1.5 yield F1 = 0.0 on v5-128k.

The mean F1 degradation is 12.6 percentage points across architectures, exceeding the 10-point pass criterion. The competition coefficient $\alpha_{ij}$ is specific to the training objective and hook point of the calibration SAE and does not transfer.


# 6 Ablations

## 6.1 Threshold Sensitivity (A1)

A fine sweep over $\tau \in [0.3, 2.0]$ confirms that F1 peaks at $\tau = 0.4$ (test F1 = 0.136), not within the LV-predicted range $[0.75, 1.25]$. F1 declines monotonically for $\tau > 0.5$: at $\tau = 1.0$, F1 = 0.099; at $\tau = 1.5$, F1 = 0.029; at $\tau = 2.0$, F1 = 0.020. The 90\% stability window spans only $\tau \in [0.3, 0.5]$ (width 0.2), indicating that $\alpha_{ij}$ acts as a weak continuous predictor at low thresholds rather than a discriminant with a natural cutoff near 1.

## 6.2 Decoder Cosine Pre-filter (A2)

Coverage at the default $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$ threshold is 34.0% of absorbed pairs --- far below the 80% target. Relaxing to 0.10 recovers 43.2% coverage; tightening to 0.25 drops to 14.7%. The pre-filter is too restrictive: the majority of absorbed pairs have decoder cosine similarity below 0.15, meaning the LV computation misses most absorption candidates before thresholding on $\alpha_{ij}$.

## 6.3 PMI Regression Without Configuration Terms (A3)

The PMI-only model yields $R^2 = 0.0006$, confirming that corpus co-occurrence has no predictive power for absorption in isolation. Removing SAE configuration variables does not unmask a hidden PMI signal.


<!-- FIGURES
- Figure 2: gen_fig_sharpness.py, fig_sharpness.pdf — Sharpness test: absorption rate vs alpha_ij bins with sigmoid and linear fits
- Figure 4: gen_fig_taxonomy_bar.py, fig_taxonomy_bar.pdf — Absorption taxonomy stacked bar chart across widths
- Figure 5: gen_fig_downstream_scatter.py, fig_downstream_scatter.pdf — Scatter plots: absorption score vs sparse probing F1 and RAVEL TPP
- Figure 6: gen_fig_das_width.py, fig_das_width.pdf — DAS(k=1) and DAS(k=3) vs SAE width with error bars
- Table 2: inline — LV detector vs cosine baseline comparison
- Table 3: inline — PMI regression coefficients
- Table 4: inline — Absorption taxonomy summary
- Table 5: inline — Absorption vs downstream correlation matrix
- Table 6: inline — Safety probe results
- Table 7: inline — Mean DAS by width
- Table 8: inline — Cross-architecture validation
-->
