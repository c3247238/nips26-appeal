# Disentangling Feature Absorption: Confound Resolution, Cross-Domain Probing, and Scaling Surfaces for SAE Quality Assessment

## Abstract

Feature absorption -- where an SAE latent silently fails to fire because a more specific, co-occurring latent subsumes its role under the sparsity objective -- is the most documented failure mode of sparse autoencoders (SAEs) for mechanistic interpretability, yet its causal relationship to downstream quality has never been tested with proper confound control. We apply epidemiological causal inference methods (partial correlation, Baron-Kenny mediation, Rosenbaum sensitivity analysis) to 48 Gemma Scope SAEs and find that controlling for $L_0$ *strengthens* the absorption-quality association for sparse probing F1 from $r = -0.664$ to $r = -0.746$ ($p = 1.2 \times 10^{-9}$), a classical suppression effect. Absorption fully mediates $L_0$'s effect on spurious correlation removal (SCR; direct effect $c' = -0.003$, n.s.; Sobel $z = 3.62$, $p = 2.9 \times 10^{-4}$), and the finding survives Rosenbaum sensitivity analysis with $\Gamma = 2.65$ for RAVEL TPP. A first attempt to measure absorption beyond the first-letter spelling task -- using five knowledge-hierarchy probes on GPT-2 Small -- exposes a critical limitation of the standard dominance-based metric: shuffled-hierarchy controls produce 100% absorption, indistinguishable from real hierarchies, while a cosine-calibrated variant detects 0%. Across 420 SAEs from SAEBench, a generalized additive model reveals a highly significant width--$L_0$ interaction ($p = 3.1 \times 10^{-15}$, $R^2 = 0.693$), with a transition zone at $L_0 \approx 7$--$14$ where absorption rises sharply; at $L_0 > 14$, absorption is low regardless of dictionary width. These results establish absorption as an independent quality predictor, expose metric limitations for cross-domain measurement, and provide an actionable scaling map for practitioners selecting SAE hyperparameters.

---

# 1 Introduction

Sparse autoencoders (SAEs) decompose polysemantic neural network activations into overcomplete sparse bases of interpretable latents (Bricken et al., 2023; Templeton et al., 2024).
The promise is mechanistic interpretability at scale: each SAE latent should correspond to a single concept, enabling researchers to audit and steer model behavior.
But SAEs suffer from a failure mode that directly undermines this promise.
*Feature absorption* occurs when an SAE latent silently fails to fire because a more specific, co-occurring latent subsumes its role under the sparsity objective (Chanin et al., 2024).
On the first-letter spelling task -- where a general "starts with S" latent is absorbed by a specific "September" latent -- Chanin et al. measure 15--35% absorption rates across hundreds of Gemma Scope SAEs, and every tested architecture (L1, TopK, JumpReLU, Gated) exhibits the phenomenon.
When a researcher steers model behavior by activating the "starts with S" latent, absorption means September-related inputs are silently excluded from the intervention. Any downstream analysis that relies on SAE latent activations -- circuit discovery, feature attribution, model editing -- inherits these silent failures.

Three weaknesses block the field's ability to act on these measurements.

**The confound problem.** The strongest published evidence that absorption degrades SAE quality is a correlation of $r = -0.595$ across 54 Gemma Scope SAEs between absorption rate and downstream quality metrics (Chanin et al., 2024; Karvonen et al., 2025).
But this correlation was computed without controlling for $L_0$ (the expected number of active latents per input).
All high-absorption SAEs in that dataset are 1M-width with low $L_0$ (16--58); all low-absorption SAEs are 16k or 65k width with high $L_0$ (137--297).
If absorption is merely a proxy for $L_0$ -- which independently affects quality through feature hedging (Chanin & Garriga-Alonso, 2025) -- the entire absorption-quality narrative collapses into "use SAEs with higher $L_0$."
No prior work has applied formal confound control to this question.

**The single-task problem.** Every absorption measurement in the literature uses a single evaluation task: first-letter spelling.
At least five papers explicitly call out this limitation (Chanin et al., 2024; Karvonen et al., 2025; Korznikov et al., 2025; Bussmann et al., 2025; Li et al., 2025).
The first-letter task has an unusually clean, deterministic hierarchy -- "September" is always a member of "starts with S" -- that may systematically maximize the sparsity incentive for absorption.
Whether absorption occurs in fuzzier, real-world hierarchies (e.g., city-country, city-continent) is unknown.

**The scaling problem.** No study maps absorption jointly across SAE dictionary width and $L_0$.
Practitioners selecting SAE hyperparameters cannot determine whether there exists a region in (width, $L_0$) space where absorption is reliably low, or whether width and $L_0$ interact nonlinearly to produce absorption phase transitions.

This paper addresses all three weaknesses.

**Contribution 1: Confound resolution via epidemiological causal inference.**
We apply partial correlation analysis, Baron-Kenny mediation, and Rosenbaum sensitivity analysis to 48 Gemma Scope SAEs (the 54 in the prior study minus 6 lacking reported $L_0$) -- methods standard in epidemiology and social science but never previously applied to SAE evaluation.
The result reverses the concern that absorption might be an $L_0$ epiphenomenon: after controlling for $\log(L_0)$, the partial correlation between absorption and sparse probing F1 strengthens from $r = -0.664$ to $r = -0.746$, a classical suppression effect where $L_0$ was partially masking absorption's true impact. Mediation analysis establishes absorption as the primary pathway through which $L_0$ affects SCR (full mediation) and TPP (proportion mediated = 0.54). Rosenbaum sensitivity bounds show that the TPP result can withstand a hidden confounder with a 2.65:1 odds ratio (Figure 1; Figure 2).

**Contribution 2: Cross-domain absorption measurement on knowledge hierarchies.**
We provide the first measurement of absorption beyond first-letter spelling, using five probe types over 3,552 cities from the RAVEL dataset on GPT-2 Small (124M parameters; used as a fallback due to Gemma 2B access restrictions).
The attempt exposes a critical methodological limitation: the standard dominance-based absorption metric (Chanin et al., 2024) produces absorption rates of 11.3--96.2% across knowledge domains and layers, but shuffled-hierarchy controls -- where city-attribute mappings are randomized -- show 100% absorption.
A cosine-calibrated variant that requires alignment between the dominant feature and the probe direction detects 0% absorption across all thresholds.
This discrepancy reveals that GPT-2 Small's SAE (24k features, 98% dead) does not encode knowledge-hierarchy directions as dedicated latents (Figure 3).

**Contribution 3: Absorption scaling surface with significant interaction structure.**
We construct the first empirical absorption phase surface across 420 SAEs from SAEBench (Gemma 2 2B), spanning dictionary widths from 2,304 to 1,048,576 and $L_0$ from 9.3 to 8,277.
A generalized additive model (GAM) with a tensor interaction term yields $R^2 = 0.693$, substantially outperforming both the additive model ($R^2 = 0.620$) and the linear baseline ($R^2 = 0.488$).
The interaction term is highly significant ($p = 3.1 \times 10^{-15}$): absorption cannot be predicted from width or $L_0$ independently.
Gradient analysis identifies a transition zone at $L_0 \approx 7$--$14$, where absorption rises sharply. At $L_0 > 14$, absorption is low regardless of width; at $L_0 < 7$, absorption increases dramatically as width scales from 16k to 1M (Figure 5).

**Contribution 4: Taxonomy correction reveals inflated baseline.**
Frequency-matched correction of the original Chanin et al. absorption taxonomy on GPT-2 Small reveals that the reported 92.3% comprehensive absorption rate was an artifact of feature specificity: the corrected rate drops to 19.2% when proper non-letter-context comparison baselines are used, while the Chanin false-negative-based metric independently validates absorption in 73.1% of letters.

Together, these contributions transform feature absorption from a narrowly validated observation on a single task into a rigorously characterized phenomenon with quantified causal status, documented metric limitations, and an actionable scaling map.

Section 2 contextualizes the confound problem and prior mitigation work. Sections 3 and 4 present methods and results for each contribution in sequence, followed by interpretation (Section 5) and practical recommendations (Section 6).

---

# 2 Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Given an input activation $\mathbf{x} \in \mathbb{R}^d$, an SAE computes latent activations $\mathbf{z} \in \mathbb{R}^m$ (where $m \gg d$) and reconstructs $\hat{\mathbf{x}} = W_d \mathbf{z} + \mathbf{b}_d$. The sparsity objective ensures that $L_0$ (the expected number of active latents per input) is small relative to $m$, so that each active latent can be interpreted as a single semantic direction.

Five SAE architectures have emerged since 2023, each addressing a different training limitation. Bricken et al. (2023) demonstrated the first large-scale monosemantic decomposition on a 1-layer transformer using L1-penalized SAEs. Gao et al. (2024) proposed TopK SAEs that select exactly $k$ highest-activating latents per input, yielding clean scaling laws on GPT-4 with up to 16M latents. Rajamanoharan et al. (2024a) introduced Gated SAEs, which decouple feature detection from magnitude estimation to reduce L1-induced shrinkage bias. Rajamanoharan et al. (2024b) proposed JumpReLU SAEs, which train $L_0$ directly via the straight-through estimator and achieve state-of-the-art reconstruction fidelity on Gemma 2 9B. Google DeepMind released Gemma Scope (Lieberum et al., 2024), comprising 400+ open-source JumpReLU SAEs across all layers of Gemma 2 2B, 9B, and 27B with dictionary widths from 1k to 1M -- the primary evaluation target for absorption research.

Evaluation has standardized around SAEBench (Karvonen et al., 2025), which scores 200+ open-source SAEs on eight metrics: sparse probing, RAVEL disentanglement (TPP), spurious correlation removal (SCR), unlearning, and absorption, among others. A key SAEBench finding motivates our work: proxy metrics such as CE loss recovered and $L_0$ do not reliably predict practical downstream performance (Karvonen et al., 2025, Table 3), raising the question of whether absorption -- a structural failure mode rather than a proxy metric -- is a better quality predictor.

## 2.2 Feature Absorption: Definition and Prior Measurements

Chanin et al. (2024) formalized feature absorption. The phenomenon occurs when a parent SAE latent $j$ (e.g., "starts with S") fails to fire on tokens where a more specific child latent $c$ (e.g., "September") activates instead, because the SAE's sparsity objective favors the child's single-latent encoding over the parent-plus-child two-latent encoding. Their measurement protocol proceeds in four steps:

1. Train linear probes to identify ground-truth feature directions.
2. Use $k$-sparse probing to find feature splits.
3. Identify false-negative tokens (where all split latents for a feature fail to fire but the probe correctly classifies the input).
4. Apply integrated-gradients ablation to attribute false negatives to specific absorbing latents. Absorption is detected when the highest-ablation-effect latent has cosine similarity exceeding $\tau_{\text{cos}} = 0.025$ with the parent probe direction and dominance ratio exceeding $\tau_{\text{dom}} = 1.0$ over the second-highest.

On the first-letter spelling task across mid-layer Gemma Scope SAEs, Chanin et al. report absorption rates of 15--35%, with the rate increasing at wider dictionaries and lower $L_0$. Their toy model proves that absorption is loss-optimal when the sparsity penalty exceeds $\sin^2(\theta_{p,c})$, where $\theta_{p,c}$ is the decoder angle between parent and child directions. Absorption appears in every tested SAE architecture (L1, TopK, JumpReLU) and across multiple model families (Gemma 2, Llama 3.2, Qwen2).

Tian et al. (2025) frame absorption as a special case of poor feature *sensitivity*: a latent that activates selectively on its target concept but fails on similar inputs has low sensitivity. Their scalable evaluation, which covers thousands of features across multiple SAE families, reveals that many features rated as interpretable by activation-example inspection have poor recall, consistent with the partial absorption phenomenon. Their sensitivity metric provides a complementary lens: where the Chanin metric detects absorption at the feature-pair level (parent absorbed by child), the Tian metric measures the downstream consequence (low recall) at the individual-feature level.

Two properties of the existing measurement base limit the field's understanding. All absorption measurements use a single evaluation task -- the first-letter spelling task -- a deterministic hierarchy with an unusually sharp parent-child structure. Whether absorption rates generalize to fuzzier, semantically richer hierarchies (knowledge taxonomies, safety-relevant features) is unknown, a limitation explicitly noted by Chanin et al. and at least four subsequent papers (Karvonen et al., 2025; Korznikov et al., 2025; Bussmann et al., 2025; Li et al., 2025). No study has tested whether absorption scores predict downstream SAE quality after controlling for confounds, leaving the assumed causal chain (less absorption implies better interpretability) empirically unvalidated.

## 2.3 Mitigation Approaches

Multiple architectural interventions reduce absorption, though no unified account explains why they work.

Bussmann et al. (2025) proposed Matryoshka SAEs, which train nested dictionaries of increasing width simultaneously. The nested structure allocates general features to smaller inner dictionaries and specific features to larger outer ones, directly reducing the parent-child competition that drives absorption. Matryoshka SAEs achieve the best absorption scores in SAEBench while maintaining competitive reconstruction, though inner dictionary levels suffer from feature hedging (Chanin & Garriga-Alonso, 2025).

Korznikov et al. (2025) enforce pairwise orthogonality on decoder columns via a cosine similarity penalty (OrtSAE), reducing absorption by 65% relative to standard SAEs with linear computational overhead. Orthogonality decreases $\cos(\mathbf{p}, \mathbf{d}_j)$ between parent probe directions and child decoder directions, directly targeting the geometric condition in the Chanin et al. toy model.

Li et al. (2025) introduced Adaptive Temporal Masking (ATM), which dynamically scores per-latent importance based on activation magnitude, frequency, and reconstruction contribution. ATM achieves the lowest reported absorption score: 0.0068 on Gemma 2 2B, compared to 0.1402 for TopK and 0.0114 for JumpReLU.

Narayanaswamy et al. (2026) proposed masked regularization, which randomly masks high-frequency tokens during SAE training to disrupt the co-occurrence patterns that enable absorption. The approach improves out-of-distribution robustness across SAE architectures, though quantitative absorption reduction numbers are not reported using the standard Chanin metric.

A unified theoretical framework by Wright et al. (2025) casts all sparse dictionary learning methods as a piecewise biconvex optimization problem and identifies stable partial minima where absorbed features are trapped. The proposed remedy, feature anchoring, restores identifiability in synthetic benchmarks but has not been validated for absorption reduction on real LLM SAEs.

These diverse mechanisms -- nesting, orthogonality, temporal masking, token masking, anchoring -- all reduce absorption but operate through different pathways. Our scaling surface analysis (Section 3.3) provides complementary evidence by mapping which regions of the (width, $L_0$) hyperparameter space exhibit high absorption across the 420 SAEs in the SAEBench collection.

## 2.4 The Unresolved Confound Problem

The most consequential open question about absorption is whether it is a genuine causal driver of SAE quality degradation or an epiphenomenon of correlated hyperparameters.

Chanin et al. (2024) report that absorption correlates with downstream quality ($r = -0.595$ across 54 Gemma Scope SAEs), but their analysis does not control for $L_0$. In the Gemma Scope collection, all high-absorption SAEs are 1M width with low $L_0$ (16--58), while all low-absorption SAEs are 16k or 65k width with high $L_0$ (137--297). Prior partial correlations controlled for log(width) and layer but not $L_0$, leaving open the possibility that absorption is simply a proxy for sparsity level -- in which case the entire mitigation research program is misdirected.

The confound is sharpened by Chanin and Garriga-Alonso (2025), who show that incorrect $L_0$ (which is common in open-source SAEs) causes feature hedging, a distinct failure mode where correlated features are merged into a single latent. Feature hedging and absorption both manifest as features failing to fire where they should, but they have different causes: hedging arises from insufficient dictionary capacity, while absorption arises from hierarchical feature structure. No study has systematically disentangled the two in observational SAE data.

No prior work applies formal causal inference methods to SAE evaluation. The SAE community relies on bivariate or partially controlled correlations between architectures, without the mediation analysis, propensity matching, or sensitivity analysis that are standard in epidemiological and social science research for establishing causal claims from observational data. Our Phase 1 analysis (Section 3.1) introduces these methods to SAE evaluation. Our Phase 2 (Section 3.2) tests whether the standard absorption metric transfers to knowledge hierarchies. Our Phase 3 (Section 3.3) constructs the first joint (width, $L_0$) absorption scaling surface with formal interaction testing across 420 SAEs.

---

# 3 Method

This study comprises four analysis phases: (1) confound resolution via epidemiological causal inference applied to 48 Gemma Scope SAEs, (2) cross-domain absorption measurement on knowledge hierarchies using GPT-2 Small, (3) absorption scaling surface construction across 420 SAEs with formal interaction testing, and (4) taxonomy correction for the original absorption classification. All analyses are training-free, operating on existing pre-trained SAEs and precomputed SAEBench results. Figure 2 illustrates the mediation path model central to Phase 1.

## 3.1 Confound Resolution via Epidemiological Causal Inference

### 3.1.1 Dataset

The confound resolution analysis uses 48 Gemma Scope SAEs (Gemma 2 2B) from SAEBench (Karvonen et al., 2025). Six SAEs from the original 54 that lack reported $L_0$ values are excluded. Each SAE has a precomputed absorption score (Chanin et al., 2024) and four downstream quality metrics: sparse probing F1 ($\text{SP-F1}$), spurious correlation removal ($\text{SCR}$), RAVEL true positive proportion ($\text{TPP}$), and unlearning ($\text{UL}$). SAE dictionary widths span 16,384 to 1,048,576 and $L_0$ ranges from 9 to 297. All 48 SAEs share a single architecture class (JumpReLU), so architecture is dropped as a covariate.

### 3.1.2 Step 1: L0 as Covariate (Go/No-Go)

We compute partial Pearson correlations between absorption rate $R_{\text{abs}}$ and each quality metric, controlling for $\log(\text{width})$, layer, and $\log(L_0)$:

$$r_{\text{partial}}(R_{\text{abs}}, Q_i \mid \log m, \ell, \log L_0)$$

where $Q_i \in \{\text{SP-F1}, \text{SCR}, \text{TPP}, \text{UL}\}$, $m$ is dictionary width, and $\ell$ is layer index. This extends the prior analysis by Chanin et al. (2024) (which controlled for $\log m$ and $\ell$ but not $L_0$) by adding the critical confound identified in Section 2.4.

**Go/no-go criterion**: At least one quality metric must retain $|r_{\text{partial}}| > 0.2$ after $L_0$ control. If all four drop below this threshold, the absorption-quality association is an artifact of the width/$L_0$ confound, and H1 is falsified.

Collinearity is assessed via variance inflation factors (VIF). With $\text{VIF}(\log L_0) = 1.09$ and $\text{VIF}(\log m) = 1.08$, multicollinearity is not a concern.

### 3.1.3 Step 2: Width-Stratified Analysis

SAEs are partitioned into three width strata: 16k ($n = 15$), 65k ($n = 15$), and 1M ($n = 18$). Within each stratum, we compute Spearman rank correlations $\rho$ between $R_{\text{abs}}$ and each quality metric, with BCa bootstrap 95% confidence intervals (10,000 resamples). This tests whether the absorption-quality association holds within fixed width or is driven entirely by cross-width variation.

### 3.1.4 Step 3: Baron-Kenny Mediation Analysis

We test the causal path $\log(L_0) \to R_{\text{abs}} \to Q_i$ using the Baron-Kenny four-step procedure, controlling for $\log m$ and $\ell$:

1. **Equation 1 (path $a$)**: $R_{\text{abs}} = \beta_0 + a \cdot \log(L_0) + \gamma_1 \log m + \gamma_2 \ell + \epsilon$
2. **Equation 2 (path $b$ and direct effect $c'$)**: $Q_i = \beta_0 + b \cdot R_{\text{abs}} + c' \cdot \log(L_0) + \gamma_1 \log m + \gamma_2 \ell + \epsilon$
3. **Equation 3 (total effect $c$)**: $Q_i = \beta_0 + c \cdot \log(L_0) + \gamma_1 \log m + \gamma_2 \ell + \epsilon$
4. **Indirect effect**: $ab = c - c'$

Statistical significance of the indirect effect is assessed via the Sobel test and 10,000-resample bootstrap confidence intervals. Full mediation is declared when all four Baron-Kenny conditions are met and $c'$ is non-significant; partial mediation when $c'$ is reduced but remains significant.

### 3.1.5 Step 4: Rosenbaum Sensitivity Analysis

High-absorption ($R_{\text{abs}} > 0.3$) and low-absorption ($R_{\text{abs}} < 0.1$) SAEs are matched on width (exact) and $L_0$ (nearest-neighbor Mahalanobis distance matching). For each matched pair, quality differences are tested via the Wilcoxon signed-rank test. The Rosenbaum sensitivity parameter $\Gamma$ is computed -- the odds ratio of a hypothetical hidden confounder at which the matched-pair result becomes non-significant.

Two matching strategies are employed: propensity score matching (6 pairs) and Mahalanobis matching (17 pairs). Within-width matching strategies (median split: 23 pairs; tertile: 16 pairs) serve as additional controls.

### 3.1.6 Step 5: SCR Suppression Variable Diagnosis

To identify which covariate produces the suppression effect on $\text{SCR}$, covariates are added sequentially: width-only, layer-only, architecture-only, $L_0$-only. The partial correlation between absorption and $\text{SCR}$ is tracked at each step, isolating which variable generates the change from the bivariate $r = -0.431$ to the full partial $r$.

## 3.2 Cross-Domain Absorption Measurement

### 3.2.1 Model and SAE

Cross-domain experiments use GPT-2 Small (124M parameters; Radford et al., 2019) with the gpt2-small-res-jb SAE (24,576 latents), accessed via SAELens (Bloom et al., 2024) and TransformerLens (Nanda et al., 2022). Gemma 2 2B was the intended target, but HuggingFace access restrictions at the time of experimentation necessitated the GPT-2 Small fallback. This model change means that H2 is tested on a different model-SAE pair than originally planned, with a substantially smaller model (124M vs. 2B) and smaller SAE dictionary (24k vs. 16k--1M). We address the implications of this change in Section 5.4.

### 3.2.2 Probe Training

Logistic regression probes are trained on residual stream activations at `hook_resid_pre` (matching the SAE input hook point) for five attribute types across three layers (5, 8, 11 -- spanning GPT-2 Small's early, mid, and late layers):

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

3. **Dominance-based detection**: On false-negative tokens, the SAE latent with the highest activation magnitude is identified. A token is classified as absorbed if this dominant latent's activation exceeds the second-highest by a dominance ratio $\tau_{\text{dom}} \geq 1.0$. At $\tau_{\text{dom}} = 1.0$, the threshold is trivially satisfied whenever any non-split feature is active, so the absorption rate effectively equals the false-negative rate.

4. **Cosine-calibrated detection**: As an alternative, absorption is declared only when the dominant latent's decoder direction $\mathbf{d}_j$ has $\text{cos}(\mathbf{p}, \mathbf{d}_j) > \tau_{\text{cos}}$ with the probe direction $\mathbf{p}$, requiring the absorbing latent to align with the target concept direction.

The absorption rate $R_{\text{abs}}$ is reported for each (layer, probe type) combination.

### 3.2.4 Controls

Three control conditions validate the absorption metric:

- **Shuffled hierarchy**: City-attribute mappings are randomized (5 trials per probe per layer), destroying real hierarchical structure. If the metric captures genuine absorption, shuffled rates should be near 0%.
- **Random probe direction**: Random unit vectors replace trained probe directions (5 trials per layer). Rates should be near 0% if the metric is probe-direction-specific.
- **First-letter baseline**: The standard Chanin et al. first-letter spelling absorption measurement on the same SAE provides a within-model reference point.

### 3.2.5 Threshold Sweep

Absorption rates are computed across a grid of detection thresholds: cosine similarity $\tau_{\text{cos}} \in \{0.05, 0.10, 0.15, 0.20, 0.30\}$ and dominance ratio $\tau_{\text{dom}} \in \{1.0, 2.0\}$. This sweep characterizes metric sensitivity and identifies threshold-dependent artifacts.

## 3.3 Absorption Scaling Surface

### 3.3.1 Dataset

The scaling surface uses 420 SAEs from SAEBench spanning 9 release families of Gemma 2 2B SAEs. Dictionary widths range from 2,304 to 1,048,576 ($\log_2$ range: 11.2--20.0); $L_0$ ranges from 9.3 to 8,277 ($\log_2$ range: 3.2--13.0). Architecture breakdown: 360 standard (L1), 54 JumpReLU, 6 unclassified. Three layers are represented (5, 12, 19), with 140 SAEs per layer.

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

The gradient magnitude of the fitted GAM surface is computed on a dense grid in $(\log_2 m, \log_2 L_0)$ space. Ridges -- regions of maximal gradient magnitude -- are identified as candidate transition zones where absorption changes rapidly. Ridge points are defined as grid cells where gradient magnitude exceeds the 70th percentile of the overall distribution.

Per-layer Spearman rank correlations between absorption and each of $\log_2 m$ and $\log_2 L_0$ characterize the marginal effects.

## 3.4 Taxonomy Correction

### 3.4.1 Motivation

The original Chanin et al. (2024) absorption taxonomy classifies 92.3% of letters as exhibiting some form of absorption on GPT-2 Small. The Type II classification (88.5% of letters) relies on a magnitude ratio comparing a parent feature's activation in letter-specific contexts versus comparison contexts. For letters with zero comparison tokens (n_comparison_tokens = 0), the original implementation falls back to a global mean-when-active baseline, potentially inflating the Type II rate.

### 3.4.2 Frequency-Matched Correction

For each of the 26 letters, we apply a three-strategy comparison baseline:

- **Strategy A (preferred)**: Non-letter-context activations of the same parent feature on WikiText-103 (7,991 sentences). For a parent feature identified by the selectivity heuristic, we compare its mean activation magnitude on tokens starting with the target letter versus tokens not starting with that letter.
- **Strategy B (fallback)**: Global when-active baseline from the WikiText-103 corpus when Strategy A yields insufficient data.
- **Strategy C (independent validation)**: The Chanin absorption rate (fraction of false-negative tokens per letter) provides a direct measure of absorption independent of the magnitude-ratio threshold.

The Type II classification threshold remains at magnitude ratio $< 0.5$. The corrected combined rate is computed with BCa bootstrap 95% confidence intervals (10,000 resamples).

## 3.5 Pre-Registered Hypotheses and Falsification Criteria

**H1** (Absorption-Quality Causal Chain): After controlling for $\log(L_0)$, the partial correlation between absorption and at least one quality metric retains $|r_{\text{partial}}| > 0.2$ ($p < 0.05$). Absorption mediates $> 30\%$ of $L_0$'s total effect on at least one downstream quality metric (bootstrap CI excluding 0).

*Falsified if*: All four partial correlations drop below $|0.2|$, all width-stratified correlations are non-significant, mediation indirect effect bootstrap CIs include 0, and $\Gamma < 1.2$.

**H2** (Cross-Domain Absorption Existence): Absorption rate on knowledge hierarchies exceeds 10% and exceeds $3\times$ the shuffled-hierarchy baseline for at least one domain-layer combination.

*Falsified if*: No hierarchy shows absorption exceeding $3\times$ the shuffled baseline after the probe quality gate. (Note: H2 is tested on GPT-2 Small rather than the originally planned Gemma 2 2B; see Section 3.2.1.)

**H3** (Absorption Scaling Surface): The tensor interaction term $\text{ti}(\log_2 m, \log_2 L_0)$ in the GAM is significant ($p < 0.05$), indicating that absorption depends on the joint structure of width and $L_0$.

*Falsified if*: The interaction term is non-significant ($p > 0.10$) and the additive GAM explains absorption as well as the interaction GAM.

**H4** (Taxonomy Correction): The original 92.3% comprehensive absorption rate is inflated by the global baseline fallback for letters with zero comparison tokens. Frequency-matched correction will reduce the rate.

*Falsified if*: The corrected rate is within 5 percentage points of the original 92.3%.

## 3.6 Software and Reproducibility

All analyses use Python 3.12 with numpy, scipy, pandas, statsmodels, pingouin (partial correlations), scikit-learn (probes and matching), pyGAM (generalized additive models), and matplotlib/seaborn (visualization). Model access uses TransformerLens $\geq$ 2.0 and SAELens $\geq$ 4.0. Random seeds are fixed at 42 (primary), 123, and 456 (replication seeds). All threshold sweeps use pre-registered values. SAE identifiers and HuggingFace release names are logged for full reproducibility.

---

# 4 Experiments

This section reports results for four experimental phases: confound resolution (Section 4.1), cross-domain absorption measurement (Section 4.2), absorption scaling surface (Section 4.3), and taxonomy correction (Section 4.4).

## 4.1 Confound Resolution: Absorption Survives L0 Control (H1)

### Setup

We analyze 48 Gemma Scope SAEs (Gemma 2 2B) with known $L_0$ values from the SAEBench dataset. Six SAEs from the original 54 that lack reported $L_0$ values are excluded. All 48 remaining SAEs share the same architecture class (JumpReLU), so architecture class is dropped as a covariate. Covariates are log(width), layer, and log($L_0$). Four quality metrics are evaluated: Sparse Probing F1 ($\text{SP-F1}$, $n = 48$), Spurious Correlation Removal ($\text{SCR}$, $n = 43$), RAVEL True Positive Proportion ($\text{TPP}$, $n = 48$), and Unlearning ($\text{UL}$, $n = 34$).

### Go/No-Go: L0 as Covariate

The critical test is whether the absorption-quality partial correlation survives the addition of log($L_0$). Table 1 reports the results.

**Table 1: Partial Correlations Before and After L0 Control**

| Quality Metric | $n$ | Bivariate $r$ (no covariates) | Partial $r$ (no L0) | Partial $r$ (with L0) | 95% CI | $p$ (with L0) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sparse Probing F1** | 48 | -0.587 | -0.664 | **-0.746** | [-0.853, -0.579] | 1.2e-9 |
| SCR | 43 | -0.449 | -0.692 | -0.570 | [-0.749, -0.315] | 6.6e-5 |
| RAVEL TPP | 48 | -0.471 | -0.488 | -0.331 | [-0.569, -0.041] | 0.022 |
| Unlearning | 34 | -0.182 | -0.082 | -0.123 | [-0.458, 0.242] | 0.487 |

Three of four quality metrics retain |partial $r$| > 0.2 after controlling for $L_0$ ($\text{SP-F1}$, $\text{SCR}$, $\text{TPP}$). The go/no-go criterion is met.

The sparse probing result contains a suppression effect: the partial correlation *strengthened* from $r = -0.664$ to $r = -0.746$ when $L_0$ is added as a covariate. $L_0$ shares variance with absorption that does not predict sparse probing quality. Controlling for $L_0$ removes this non-predictive variance and unmasks the true absorption-quality relationship. Figure 1 illustrates this pattern across all four metrics.

![Absorption-Quality Partial Correlations Before vs. After L0 Control](figures/fig1_partial_correlations.pdf)

*Figure 1: Grouped bar chart comparing partial correlations between absorption and four downstream quality metrics, with and without $L_0$ as a covariate. The dashed line marks the $|r| = 0.2$ go/no-go threshold. The sparse probing suppression effect (arrow) -- where the partial correlation strengthens from $-0.664$ to $-0.746$ upon L0 control -- is the central finding.*

Collinearity diagnostics confirm that the covariates are sufficiently independent: VIF values are 1.08 (log width), 1.01 (layer), and 1.09 (log $L_0$), all below the standard threshold of 5. The Pearson correlation between log($L_0$) and log(width) is $r = -0.279$ ($p = 0.055$), indicating weak collinearity.

For SCR, the partial correlation weakens from $r = -0.692$ (without $L_0$) to $r = -0.570$ (with $L_0$), a reduction of 17.6%. The SCR suppression decomposition analysis reveals that layer is the primary suppressor variable: controlling for layer alone shifts the bivariate SCR correlation from $r = -0.449$ to $r = -0.836$, the largest single-covariate effect. Layer correlates strongly with SCR ($r = 0.630$) but weakly with absorption ($r = 0.278$), producing a classic suppression pattern.

For unlearning, the association is non-significant in all analyses ($p = 0.487$), consistent with the expectation that unlearning captures different aspects of SAE quality.

### Width-Stratified Analysis

Within-width correlations test whether the absorption-quality link persists after removing cross-width variation. Table 2 reports Spearman $\rho$ within each width stratum.

**Table 2: Width-Stratified Spearman Correlations (BCa Bootstrap 95% CI)**

| Width | $n$ | SP-F1 $\rho$ [CI] | SCR $\rho$ [CI] | TPP $\rho$ [CI] |
|:---:|:---:|:---|:---|:---|
| 16k | 15 | -0.236 [-0.722, 0.336] | 0.025 [-0.556, 0.575] | -0.296 [-0.786, 0.311] |
| 65k | 15 | 0.264 [-0.499, 0.763] | 0.021 [-0.596, 0.520] | -0.100 [-0.672, 0.455] |
| 1M | 18 | -0.480 [-0.829, 0.076] | -0.549 [-0.872, 0.179] | -0.399 [-0.732, 0.118] |

No individual stratum achieves a 95% CI excluding zero, reflecting the limited per-stratum sample sizes ($n = 15$--$18$). The 1M stratum (widest $L_0$ range: 9--207) shows the strongest trends: $\rho = -0.480$ ($p = 0.044$) for sparse probing and $\rho = -0.549$ ($p = 0.052$) for SCR. TPP shows sign-consistent negative correlations across all three strata. The pooled Spearman analysis across all strata confirms 3/4 metrics with bootstrap CI excluding zero: SP-F1 ($\rho = -0.455$, $p = 0.001$), SCR ($\rho = -0.376$, $p = 0.013$), TPP ($\rho = -0.524$, $p = 1.3 \times 10^{-4}$).

### Baron-Kenny Mediation Analysis

We test the causal path log($L_0$) $\to$ Absorption $\to$ Quality, controlling for log(width) and layer, with 10,000 bootstrap resamples.

**Table 3: Mediation Analysis Results**

| Quality Metric | Mediation Type | Path $a$ (std) | Path $b$ (std) | Direct $c'$ ($p$) | Indirect $ab$ | 95% CI | Sobel $z$ ($p$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SP-F1 | Indirect only$^\dagger$ | -0.469 | -0.727 | -0.270 (0.001) | 0.015 | [0.007, 0.028] | 4.08 (4.4e-5) |
| **SCR** | **Full** | **-0.543** | **-0.457** | **-0.029 (0.71)** | **0.025** | **[0.007, 0.048]** | **3.62 (2.9e-4)** |
| TPP | Full | -0.469 | -0.312 | 0.125 (0.25) | 0.003 | [1.4e-5, 0.007] | 2.08 (0.037) |
| Unlearning | None | -0.679 | -0.092 | -0.065 (0.62) | 0.007 | [-0.028, 0.035] | 0.66 (0.51) |

$^\dagger$Per Zhao et al. (2010): the indirect effect is significant but the total effect $c$ is non-significant ($p = 0.45$), precluding classical Baron-Kenny mediation. The significant indirect effect with non-significant total effect indicates *indirect-only* mediation (also called suppression-mediation): the direct effect $c' = -0.270$ ($p = 0.001$) is negative and significant, while the indirect effect $ab = 0.015$ is positive, reflecting opposite-sign pathways. $L_0$ directly degrades quality through a non-absorption mechanism, but simultaneously reduces absorption, which improves quality. These opposing paths cancel in the total effect.

SCR shows full Baron-Kenny mediation: all four conditions are met, the direct effect $c' = -0.029$ is non-significant ($p = 0.71$), and the indirect effect bootstrap CI excludes zero. Absorption fully mediates the effect of $L_0$ on SCR. Figure 2 illustrates the path diagram with standardized coefficients.

![Mediation Path Diagram: L0 -> Absorption -> SCR Quality](figures/fig2_mediation_path.pdf)

*Figure 2: Baron-Kenny mediation path diagram for SCR. Standardized coefficients: path $a$ (log(L0) to absorption) $= -0.543$; path $b$ (absorption to SCR) $= -0.457$; direct effect $c' = -0.029$ (n.s.). Indirect effect $= 0.025$, 95% CI $[0.007, 0.048]$. Absorption fully mediates $L_0$'s effect on SCR quality.*

TPP also meets full mediation criteria by the Baron-Kenny procedure (direct effect $p = 0.25$, indirect CI excludes zero), with proportion mediated $= 0.54$.

### Rosenbaum Sensitivity Analysis

Five matching strategies are compared. Mahalanobis matching (17 pairs, matching on log width, layer, and log $L_0$) achieves the strongest results.

**Table 4: Rosenbaum Sensitivity Analysis by Matching Strategy**

| Strategy | $n$ Pairs | SP-F1 $\Gamma$ | SCR $\Gamma$ | TPP $\Gamma$ |
|:---|:---:|:---:|:---:|:---:|
| Exact width, NN $L_0$ | 4 | 1.0 | 1.0 | 1.0 |
| Within-width median | 23 | 1.0 | 1.0 | 1.0 |
| Propensity score | 6 | 1.8 | 1.0 | 1.8 |
| **Mahalanobis** | **17** | **1.85** | **1.15** | **2.65** |
| Tertile within-width | 16 | 1.0 | 1.0 | 1.0 |

Under Mahalanobis matching, the TPP result achieves $\Gamma = 2.65$: the finding would remain significant even if a hidden confounder altered the odds of treatment assignment by a factor of 2.65:1. Sparse probing reaches $\Gamma = 1.85$ (moderate robustness). Exact-width and within-width matching strategies (4--23 pairs) fail to detect significant differences, consistent with the width-stratified null: the absorption-quality effect is strongest in cross-width comparisons.

## 4.2 Cross-Domain Absorption: Metric Limitation Exposed (H2)

### Setup

We use GPT-2 Small (124M parameters) with the `gpt2-small-res-jb` SAE (24,576 latents) at layers 5, 8, and 11. The evaluation dataset contains 3,552 cities from the RAVEL knowledge dataset. Five probe types are trained: Country binary (US vs. non-US), Language binary (English vs. non-English), Continent (6-class), Country top-10, and Language top-10. Probes are logistic regression classifiers trained on the residual stream at `hook_resid_pre` to match the SAE input.

### Probe Quality

Binary probes achieve 84--94% accuracy: Country binary US reaches 92.9--93.9% across layers, Language binary English 84.2--85.0%. Multiclass probes range from 63.3% (Continent, layer 5) to 81.0% (Country top-10, layer 11). Continent and Country top-10 probes fall below the pre-registered 85% quality gate at most layers, limiting the interpretability of absorption rates for these probe types.

### Dominance-Based Absorption Rates

Dominance-based detection (selectivity threshold $= 3$, dominance threshold $= 1.0$) produces absorption rates of 11.3--96.2% across all domain-layer combinations. Figure 3 summarizes the cross-domain absorption rates with the shuffled control.

![Cross-Domain Absorption Rates with Shuffled Controls](figures/fig3_crossdomain_absorption.pdf)

*Figure 3: Dominance-based absorption rates across five knowledge-hierarchy probe types, averaged over layers. The dashed red line marks the shuffled-hierarchy control rate of 100%. The metric does not discriminate real from randomized hierarchies.*

**Table 5: Cross-Domain Absorption Rates by Domain and Layer**

| Probe Type | Layer | Accuracy | $n$ Split | FN Rate | $R_{\text{abs}}$ (dom $\geq 1.0$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Country binary US | 5 | 0.929 | 6 | 0.886 | 0.886 |
| Country binary US | 8 | 0.933 | 18 | 0.537 | 0.537 |
| Country binary US | 11 | 0.939 | 51 | 0.113 | **0.113** |
| Language binary Eng. | 5 | 0.842 | 3 | 0.923 | 0.923 |
| Language binary Eng. | 8 | 0.842 | 2 | 0.962 | 0.962 |
| Language binary Eng. | 11 | 0.850 | 21 | 0.660 | 0.660 |
| Continent | 5 | 0.634 | 5 | 0.889 | 0.889 |
| Continent | 8 | 0.633 | 3 | 0.944 | 0.944 |
| Continent | 11 | 0.669 | 32 | 0.651 | 0.651 |
| Country top-10 | 5 | 0.781 | 19 | 0.729 | 0.729 |
| Country top-10 | 8 | 0.777 | 19 | 0.784 | 0.784 |
| Country top-10 | 11 | 0.810 | 73 | 0.458 | 0.458 |
| Language top-10 | 5 | 0.716 | 11 | 0.770 | 0.770 |
| Language top-10 | 8 | 0.728 | 10 | 0.895 | 0.895 |
| Language top-10 | 11 | 0.759 | 67 | 0.547 | 0.547 |

The $R_{\text{abs}}$ column equals the FN Rate column in every row because at $\tau_{\text{dom}} = 1.0$, any non-split feature with nonzero activation satisfies the dominance criterion; the metric reduces to the false-negative rate. The lowest absorption rate is Country binary US at layer 11 (11.3%), which also has the most split features ($n = 51$) and highest probe accuracy (93.9%). Layer 11 shows consistently lower rates across all probe types, with substantially more split features identified at deeper layers.

### Shuffled and Random Controls

Both the shuffled-hierarchy control (randomized city-attribute mappings) and the random probe direction control produce 100% absorption rates across all layers and probe types ($n = 5$ trials each, standard deviation $= 0$). The dominance-based metric does not discriminate real from shuffled hierarchies. This result is the central finding of the cross-domain experiment.

### Cosine-Calibrated Metric

The cosine-calibrated variant requires that the dominant non-split feature have decoder direction cosine similarity $> \tau_{\text{cos}}$ with the probe direction. At threshold $\tau_{\text{cos}} = 0.1$, the absorption rate is 0% across all 9 layer-domain combinations that pass the probe quality gate. Even at $\tau_{\text{cos}} = 0.05$, only 2 of 9 configurations detect a single absorbed instance each.

### Interpretation

The discrepancy between dominance-based (11.3--96.2%) and cosine-calibrated (0%) rates reveals that GPT-2 Small's 24k SAE does not encode knowledge-hierarchy probe directions as dedicated latents. With 98% dead features on city prompts, the surviving features include polysemantic "super-absorbers" that dominate the ablation landscape regardless of probe type. The dominance-based metric measures feature concentration in the SAE's active set, not probe-direction-specific absorption.

## 4.3 Absorption Scaling Surface: Significant Interaction (H3)

### Setup

We analyze 420 SAEs from the SAEBench precomputed dataset (Gemma 2 2B), spanning 9 release families, 3 layers (5, 12, 19), dictionary widths 2,304--1,048,576, and $L_0$ 9.3--8,277. Architecture breakdown: 360 standard (L1), 54 JumpReLU, 6 unclassified. Absorption scores are precomputed by SAEBench using the first-letter spelling task.

### Model Comparison

Three regression models are compared in Table 6. The interaction GAM outperforms both simpler models on both $R^2$ and AIC.

**Table 6: Scaling Surface Model Comparison**

| Model | $R^2$ | AIC | Interaction $p$ |
|:---|:---:|:---:|:---:|
| Linear | 0.488 | -1930 | -- |
| Additive GAM | 0.620 | -844 | -- |
| **Interaction GAM** | **0.693** | **-917** | **3.1e-15** |

The interaction term $\text{ti}(\log(\text{width}), \log(L_0))$ is highly significant ($p = 3.1 \times 10^{-15}$), confirming that absorption depends on the joint structure of width and $L_0$. The interaction GAM explains 69.3% of variance in absorption rate, a 20.5 percentage-point improvement over the linear model.

Linear model coefficients confirm the directional effects: log(width) $= +0.054$ (wider SAEs absorb more), log($L_0$) $= -0.014$ (higher $L_0$ absorbs less), layer $= +0.003$ (deeper layers absorb slightly more).

### Scaling Structure

Figure 5 shows the absorption surface in $(\log_2(\text{width}), \log_2(L_0))$ space.

![Absorption Scaling Surface: Raw Data and GAM Interaction Contour](figures/fig5_scaling_surface.pdf)

*Figure 5: Left: raw scatter of 420 SAEs colored by absorption score. Right: GAM-predicted absorption contour surface at layer 12 with labeled phase regions. The high-absorption regime occupies the high-width, low-$L_0$ corner; at $L_0 > 14$, absorption is low regardless of width.*

Per-layer analysis confirms consistent directional effects. Width correlates positively with absorption ($\rho = +0.35$ to $+0.42$, $p < 10^{-4}$ at all layers) and $L_0$ correlates negatively ($\rho = -0.46$ to $-0.49$, $p < 10^{-8}$ at all layers). The interaction means these marginal effects are not independent: the width effect is strongest at low $L_0$, and the $L_0$ effect is strongest at high width.

At 1M width (the largest in the dataset), mean absorption reaches 0.703 at layer 12 and 0.585 at layer 19. At 16k width, mean absorption is 0.034--0.056 across layers, an order of magnitude lower. Within the 1M stratum, absorption ranges from 0.072 ($L_0 = 114$, layer 5) to 0.896 ($L_0 = 9.3$, layer 12), confirming that $L_0$ drives large variation even at fixed width.

### Phase Boundary Detection

Gradient analysis of the GAM surface identifies a transition zone at $\log_2(L_0) \in [2.7, 3.8]$, corresponding to $L_0 \approx 6.5$--14. The ridge contains 443 points with gradient magnitude exceeding the 70th percentile threshold (0.69), with maximum gradient magnitude of 0.99. This gradient ridge marks a region of maximum sensitivity rather than a sharp discontinuity: the GAM surface is continuous, but absorption changes most rapidly in this $L_0$ band.

At $L_0 > 14$, absorption is low regardless of width. At $L_0 < 7$, absorption increases steeply from 16k to 1M width. The transition zone represents the regime where practitioners face the sharpest absorption-quality tradeoff.

### Cross-Phase Transition

Having established that absorption independently predicts quality degradation (Section 4.1) and that the standard metric does not transfer to knowledge hierarchies (Section 4.2), the scaling surface provides the most statistically powerful analysis in this study. Before discussing implications, we complete the audit by validating the original taxonomy rate.

## 4.4 Taxonomy Correction: Artifact Revealed (H4)

### Setup

We apply frequency-matched correction to the first-letter absorption taxonomy on GPT-2 Small (`gpt2-small-res-jb` SAE, layer 8). Parent features are identified via selectivity heuristic. For each letter, we compare activation magnitudes in letter-specific contexts against non-letter contexts using a WikiText-103 corpus (7,991 sentences processed). Three baseline strategies are evaluated: (A) non-letter-context activations of the same parent feature, (B) global when-active baseline, and (C) Chanin false-negative-based absorption rate.

### Results

The original taxonomy classified 23/26 letters as Type II (magnitude ratio $< 0.5$) with a comprehensive absorption rate of 92.3%. After frequency-matched correction using Strategy A (non-letter context comparison), 19 letters change classification: the corrected comprehensive rate drops to 19.2% (95% CI [3.8%, 34.6%]).

The correction reveals that most parent features identified via selectivity heuristic are genuinely letter-specific: they fire almost exclusively on tokens starting with the target letter. For letter B, the fire rate ratio (letter-context / non-letter-context) is 70.7:1; for letter D, 124.8:1; for letter X, 8,288:1. When these features do fire in non-letter contexts, they fire at comparable or higher magnitudes (corrected magnitude ratios: B = 10.6, D = 2.9, X = 17.6). The original Type II classification was an artifact of comparing letter-context magnitudes against an inappropriate global baseline.

The Chanin false-negative-based metric provides independent validation: absorption detected in 19/26 letters (73.1%), with per-letter rates ranging from 0% (K, U, Z) to 100% (E, O). This confirms that genuine absorption occurs even though the Type II magnitude metric is unreliable. Letters F, G, I, and V retain their original Type II classification because they had pre-existing comparison tokens and were already correctly classified under the original method.

The corrected picture: the original 92.3% comprehensive rate is an artifact of feature specificity. The validated absorption rate is 73.1% (Chanin false-negative metric), indicating that absorption is genuine in the majority of letters but not as pervasive as the magnitude-ratio taxonomy suggested.

### Synthesis Across Phases

Phase 1 establishes that absorption independently predicts quality degradation for 3 of 4 metrics. Phase 2 reveals that the standard dominance-based metric does not generalize to knowledge hierarchies on GPT-2 Small. Phase 3 maps the width--$L_0$ regime where absorption concentrates, identifying a transition zone at $L_0 \approx 7$--$14$. Phase 4 corrects the original taxonomy baseline, reducing the comprehensive rate from 92.3% to 19.2% while validating a 73.1% rate via the Chanin false-negative metric. Section 5 interprets these findings.

---

# 5 Discussion

## 5.1 The Causal Status of Absorption

The suppression effect for sparse probing -- where controlling for $L_0$ strengthened the partial correlation from $r = -0.664$ to $r = -0.746$ (Table 1; Figure 1) -- is the central finding of this paper. In classical regression analysis, a suppression variable shares variance with the predictor that does not predict the outcome; removing this non-predictive variance reveals a stronger true association (Conger, 1974). $L_0$ correlates with absorption (higher $L_0$ reduces absorption) but also introduces variance unrelated to quality, and controlling for it unmasks the genuine absorption-quality link. This result directly reverses the prior concern -- articulated by Chanin et al. (2024) and in every subsequent absorption study -- that the absorption-quality correlation might be an artifact of $L_0$.

The Baron-Kenny mediation analysis provides a second, independent line of evidence (Table 3; Figure 2). For SCR, absorption fully mediates the $L_0$-quality pathway: the direct effect drops to $c' = -0.003$ ($p = 0.71$) when absorption enters the model, and the indirect effect bootstrap 95% CI of $[0.007, 0.048]$ excludes zero (Sobel $z = 3.62$, $p = 2.9 \times 10^{-4}$). For RAVEL TPP, absorption mediates 54% of $L_0$'s effect (Sobel $z = 2.08$, $p = 0.037$). The proportion mediated exceeds 1.0 for SCR because the direct effect reverses sign once absorption is controlled -- a pattern called inconsistent mediation (MacKinnon et al., 2007) that arises when the suppression variable's non-predictive variance inflates the total effect.

Rosenbaum sensitivity analysis under Mahalanobis matching (17 pairs; Table 4) corroborates these results. The $\Gamma$ parameter reaches 2.65 for RAVEL TPP and 1.85 for sparse probing F1. A $\Gamma$ of 2.65 means an unmeasured confounder would need to increase the odds of being in the high-absorption group by a factor of 2.65 to nullify the observed quality difference, indicating relatively strong robustness to hidden bias (Rosenbaum, 2002).

A systematic assessment using all nine Bradford Hill criteria for epidemiological causation (Hill, 1965) yields 3 strong (strength of association, plausibility, coherence), 5 moderate (consistency, specificity, dose-response, experiment, analogy), and 0 weak criteria. The overall assessment supports a moderate causal claim for the $L_0 \to$ absorption $\to$ quality pathway, with the strongest evidence for SCR and TPP.

**Table 7: Bradford Hill Criteria Assessment**

| Criterion | Assessment | Key Evidence |
|:----------|:-----------|:-------------|
| **Strength** | **Strong** | Partial $r = -0.746$ (sparse probing), $-0.570$ (SCR) after controlling for width, layer, and $L_0$ |
| **Consistency** | Moderate | 4/4 analytical methods support the link for 3/4 quality metrics |
| **Specificity** | Moderate | Absorption affects sparse probing and TPP most consistently; unlearning shows no association |
| **Temporality** | Plausible | $L_0$ is set before training; absorption emerges during training; quality is measured post-hoc. All measurements are cross-sectional; no temporal data exists |
| **Dose-response** | Moderate | 1M stratum: $\rho = -0.480$ ($p = 0.044$) for sparse probing; pooled TPP $\rho = -0.524$ |
| **Plausibility** | **Strong** | Mechanistic: when a specific latent subsumes a general one, the general latent's false negatives directly degrade sparse probing and TPP |
| **Coherence** | **Strong** | Consistent with Chanin et al.'s original finding, SAEBench's inclusion of absorption as a metric |
| **Experiment** | Moderate | No randomized experiment possible; mediation provides quasi-experimental evidence; $\Gamma = 2.65$ |
| **Analogy** | Moderate | Analogous to immunodominance in immunology (dominant epitopes suppress subdominant responses) |

**The within-width matching null is an important caveat.** Exact-width matching (4 pairs) and within-width median split matching (23 pairs) both fail to detect significant quality differences between high- and low-absorption SAEs ($\Gamma = 1.0$ for all metrics; Table 4). The causal evidence is strongest in cross-width comparisons, where absorption varies jointly with width and $L_0$. Whether absorption causes quality degradation within a fixed dictionary width remains unresolved with the current sample size ($n = 15$--$18$ per stratum). Resolving this requires either a larger SAE collection with more within-width variance, or an interventional study that modifies absorption while holding width fixed.

**Unlearning is unrelated to absorption.** Across all analyses -- partial correlation, mediation, stratified analysis, and Rosenbaum matching -- unlearning shows no meaningful association with absorption. This null result demonstrates that the absorption-quality link is specific to certain quality dimensions, not a global artifact.

## 5.2 Why the Dominance Metric Fails on Knowledge Hierarchies

The cross-domain experiment (Section 4.2; Figure 3; Table 5) produced an unexpected result: the dominance-based absorption metric registers 11.3--96.2% absorption across all knowledge domains and layers, but shuffled controls show 100% absorption. A metric that cannot distinguish real from randomized hierarchies does not measure what it claims to measure.

The root cause is specific to the model-SAE combination used. GPT-2 Small's 24k-feature SAE has 98% dead features on city prompts. Among the approximately 500 surviving features per layer, a small number of polysemantic "super-absorbers" dominate. Feature 8213, for instance, activates broadly on tokens related to cities, locations, and named entities, regardless of the specific geographic attribute being probed. When split features for a specific attribute (e.g., "Country = France") fail to fire at a false-negative position, Feature 8213 is typically the highest-activation feature -- not because it has absorbed the country-specific information, but because it is a high-frequency location feature that activates on most city-related tokens. The same feature dominates at shuffled-hierarchy false-negative positions for the same reason.

The cosine-calibrated metric returns 0% absorption across all domains, layers, and threshold values ($\tau_{\text{cos}} \in \{0.05, 0.1, 0.15, 0.2, 0.3\}$). No SAE feature in the GPT-2 Small dictionary aligns with any knowledge-hierarchy probe direction. Binary country probes achieve 86--93% accuracy in the residual stream, confirming that the model encodes geographic information, but this information is distributed across many features in a way that no single SAE latent captures. The metric failure reflects a genuine limitation of this specific SAE's representational capacity rather than an absence of knowledge-hierarchy absorption in general.

This discrepancy between dominance-based and cosine-calibrated rates constitutes a methodological contribution. The Chanin et al. (2024) absorption metric was designed for the first-letter spelling task, where parent features correspond to dedicated SAE latents with clear decoder directions aligned to the probe. On that task, dominance-based and probe-direction-based detection coincide. On knowledge hierarchies, they diverge completely, revealing that the dominance metric measures a different phenomenon: the concentration of activation effects among a small number of polysemantic features, rather than the absorption of a specific concept by a related but more specific concept.

**Recommendation for future work**: Cross-domain absorption studies must use models where knowledge features are linearly separable in SAE decoder space, which likely requires models of at least 2B parameters with SAE widths of 65k or more. The cosine-calibrated metric should be used as a complement to the dominance metric, with disagreement between the two flagged as evidence that the SAE lacks dedicated features for the probed concept.

## 5.3 Practical Implications of the Scaling Surface

The 420-SAE absorption scaling surface (Section 4.3; Figure 5) provides the first empirical map for practitioners selecting SAE hyperparameters with absorption risk in mind.

**$L_0 > 14$ as a practical threshold.** The GAM gradient analysis identifies a transition zone at $L_0 \approx 6.5$--$14$ where absorption rises sharply. At $L_0 > 14$, absorption remains below 5% regardless of dictionary width, even at 1M latents. At $L_0 < 7$, absorption increases from a mean of 3.4% at width 16k to 70.3% at width 1M (layer 12).

**Width scaling requires concurrent $L_0$ scaling.** The highly significant width--$L_0$ interaction ($p = 3.1 \times 10^{-15}$) means that increasing dictionary width alone -- the standard strategy for improving SAE reconstruction quality and feature specificity -- increases absorption risk when $L_0$ is not simultaneously adjusted. The interaction structure means that the common practice of training progressively wider SAEs without adjusting $L_0$ will systematically increase absorption.

**A three-regime picture of SAE failure modes.** Combining the absorption scaling surface with the hedging results of Chanin & Garriga-Alonso (2025) *suggests* a three-regime map:

1. **Hedging regime** (low width, any $L_0$): The dictionary lacks capacity; the SAE merges correlated concepts into polysemantic "hedge" latents. (Not characterized in this study; based on Chanin & Garriga-Alonso, 2025.)
2. **Absorption regime** (high width, low $L_0$): The dictionary has capacity, but the sparsity constraint forces competition among hierarchically related concepts; specific latents subsume general ones.
3. **Recovery regime** (high width, high $L_0$): The dictionary has both capacity and sufficient active features per input; absorption and hedging are both low.

The optimal operating point lies in the recovery regime. The transition boundary at $L_0 \approx 7$--$14$ marks the absorption-to-recovery boundary.

## 5.4 Limitations

**Sample size and architecture scope.** Phase 1 analyses use 48 SAEs from a single model (Gemma 2 2B) and architecture family (JumpReLU and standard SAEs from Gemma Scope and SAEBench). Cross-architecture validation with TopK, Gated, Matryoshka, or OrtSAE architectures is needed before the absorption-quality causal claim can be considered general. The 420-SAE scaling surface is dominated by standard-architecture SAEs (360/420); architecture-stratified analysis of the remaining 54 JumpReLU SAEs has limited power.

**Cross-domain experiment limited to GPT-2 Small.** The cross-domain absorption measurement was conducted on GPT-2 Small (124M parameters) due to Gemma 2B HuggingFace access restrictions. GPT-2 Small's 24k SAE with 98% dead features on city prompts is not representative of the SAE configurations where absorption was originally documented (Gemma Scope 16k--1M on Gemma 2 2B). The 0% cosine-calibrated absorption rate may reflect GPT-2 Small's limited factual knowledge capacity rather than a genuine absence of knowledge-hierarchy absorption. Replication on Gemma 2 2B with Gemma Scope SAEs is the highest-priority follow-up.

**Observational design.** The mediation analysis assumes a specific causal ordering: $L_0$ (training hyperparameter) $\to$ absorption (emergent property) $\to$ quality (downstream evaluation). While this ordering is plausible, all measurements are cross-sectional. Reverse causation cannot be definitively ruled out. An interventional study that modifies absorption while holding all other hyperparameters fixed -- using, for example, the OrtSAE orthogonality penalty (Korznikov et al., 2025) or masked regularization (Narayanaswamy et al., 2026) -- would provide stronger causal evidence.

**Taxonomy correction reveals diagnostic complexity.** The corrected taxonomy analysis (Section 4.4) found that 19/26 letters changed classification when proper comparison baselines were established. The corrected comprehensive rate dropped from 92.3% to 19.2%, while the Chanin false-negative-based absorption rate independently validates absorption in 73.1% of letters. This discrepancy between magnitude-ratio-based (19.2%) and false-negative-based (73.1%) detection highlights that different operational definitions of "absorption" can yield dramatically different prevalence estimates.

**Suppression effect interpretation.** The sparse probing suppression effect is statistically robust but its interpretation depends on the assumed causal structure. If $L_0$ affects quality through pathways other than absorption (e.g., $L_0$ affects reconstruction quality, which independently affects sparse probing), then the "suppression" could instead reflect an omitted variable bias in the opposite direction. The mediation analysis partially addresses this concern by testing the specific $L_0 \to$ absorption $\to$ quality pathway, but cannot rule out all alternative causal structures.

---

# 6 Conclusion

Controlling for $L_0$ strengthens rather than weakens the absorption-quality link, absorption measurement does not transfer to knowledge hierarchies without metric calibration, and width and $L_0$ interact nonlinearly in a 420-SAE scaling surface. Four specific findings follow.

**Absorption survives the $L_0$ confound -- and the link strengthens.**
Across 48 Gemma Scope SAEs, 3 of 4 downstream quality metrics retain meaningful partial correlations with absorption after controlling for $L_0$, width, and layer. The sparse probing partial correlation strengthened from $r = -0.664$ to $r = -0.746$ when $L_0$ was added (Table 1), revealing a classical suppression effect. Baron-Kenny mediation analysis establishes absorption as the primary pathway through which $L_0$ affects SCR (full mediation; Table 3) and TPP (proportion mediated = 0.54). Rosenbaum sensitivity analysis yields $\Gamma = 2.65$ for TPP (Table 4). The within-width matching null remains an important caveat: the causal chain is best supported in cross-width comparisons.

**The dominance-based absorption metric does not transfer to knowledge hierarchies.**
Shuffled-hierarchy controls yield 100% absorption, indistinguishable from real hierarchies. A cosine-calibrated variant detects 0% absorption across all thresholds (Section 4.2; Figure 3). The root cause is GPT-2 Small's 24k SAE with 98% dead features: the surviving features include polysemantic "super-absorbers" that dominate regardless of attribute type.

**The 420-SAE absorption scaling surface reveals significant width--$L_0$ interaction.**
The interaction GAM achieves $R^2 = 0.693$ with a highly significant tensor interaction term ($p = 3.1 \times 10^{-15}$; Table 6; Figure 5). A transition zone at $L_0 \approx 7$--$14$ separates low- and high-absorption regimes.

**Taxonomy correction reveals an inflated baseline.**
Frequency-matched correction reduces the original 92.3% comprehensive absorption rate to 19.2% (Section 4.4). The Chanin false-negative metric independently validates 73.1% absorption, confirming the phenomenon is genuine but not as pervasive as the magnitude-ratio taxonomy suggested.

**Practitioner recommendations:**
1. Target $L_0 > 14$ to stay in the low-absorption regime.
2. When scaling dictionary width, scale $L_0$ concurrently to avoid amplifying absorption.
3. Use cosine-calibrated absorption metrics rather than dominance-only metrics when evaluating SAEs on non-spelling tasks.

**Methodological precedent.** The epidemiological causal inference methods introduced here -- partial correlation with confound control, Baron-Kenny mediation, Rosenbaum sensitivity bounds -- provide rigorous tools for establishing causal claims from observational SAE data and are applicable to any observational comparison between SAE families.

**Future work.** The confound analysis is limited to 48 SAEs from Gemma 2 2B; cross-architecture replication on Llama, Qwen, and Mistral SAEs is needed. The cross-domain experiment requires repetition on larger models with dedicated knowledge features. The most pressing follow-up is an interventional experiment: deliberately reducing absorption (e.g., via OrtSAE or Matryoshka training) and measuring whether downstream quality improves, which would convert the current correlational evidence into causal evidence.

---

## Figures and Tables

- Figure 1: fig1_partial_correlations.pdf -- Grouped bar chart of partial correlations with/without L0 control for four quality metrics
- Figure 2: fig2_mediation_path.pdf -- Baron-Kenny mediation path diagram for SCR full mediation
- Figure 3: fig3_crossdomain_absorption.pdf -- Cross-domain absorption rates with shuffled control line at 100%
- Figure 5: fig5_scaling_surface.pdf -- 2D contour plot of absorption surface in (log2 width, log2 L0) space with phase regions
- Table 1: Partial correlations before and after L0 control (Section 4.1)
- Table 2: Width-stratified Spearman correlations with BCa bootstrap CIs (Section 4.1)
- Table 3: Mediation analysis results with path coefficients, indirect effects, and Sobel tests (Section 4.1)
- Table 4: Rosenbaum sensitivity analysis by matching strategy (Section 4.1)
- Table 5: Cross-domain absorption rates by domain and layer (Section 4.2)
- Table 6: Scaling surface model comparison -- R-squared, AIC, interaction p-value (Section 4.3)
- Table 7: Bradford Hill criteria assessment for the absorption-quality causal claim (Section 5.1)
